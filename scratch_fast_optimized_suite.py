"""
scratch_fast_optimized_suite.py — Fast, Verified Multi-Phase Evaluation Suite
Flushes all prints immediately and logs complete results to JSON.
"""
import os
import sys
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from dataset_loader import BCIDatasetLoader
from feature_extraction import extract_features_from_window
from train import TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH, balance_training_data
from model import ResidualCNNBlock, HybridBCINet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(msg, flush=True)

def compute_multiclass_roc_auc(y_true, y_score):
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        if y_true_bin.shape[1] == 1:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr"))
    except Exception:
        return 0.5

# ─── Load Dataset ─────────────────────────────────────────────────────────────
def load_cohort(subjects=[1, 2, 3]):
    loader = BCIDatasetLoader()
    physionet_mapping = {1: 0, 4: 1, 6: 2, 2: 3, 3: 4}
    X_list, y_list, sub_list = [], [], []
    hashes = set()
    
    for s in subjects:
        for run, cid in physionet_mapping.items():
            try:
                data, sfreq = loader.load_physionet(subject=s, run=run)
                chunk_size = int(sfreq * 2.0)
                if data.shape[0] > N_CHANNELS:
                    data = data[:N_CHANNELS]
                elif data.shape[0] < N_CHANNELS:
                    data = np.vstack((data, np.zeros((N_CHANNELS - data.shape[0], data.shape[1]))))
                for i in range(data.shape[1] // chunk_size):
                    window = data[:, i * chunk_size : (i + 1) * chunk_size]
                    feats = extract_features_from_window(window, sfreq=sfreq)
                    h = hash(feats.tobytes())
                    if h not in hashes:
                        hashes.add(h)
                        X_list.append(feats)
                        y_list.append(cid)
                        sub_list.append(s)
            except Exception as e:
                log(f"Error loading S{s} R{run}: {e}")
                
    return torch.tensor(np.array(X_list, dtype=np.float32)), torch.tensor(np.array(y_list, dtype=np.int64)), np.array(sub_list, dtype=np.int32)

# ─── Ablation Architectures ───────────────────────────────────────────────────
class CNNAblationNet(nn.Module):
    def __init__(self, cnn_filters=128, dropout_p=0.3):
        super().__init__()
        self.res1 = ResidualCNNBlock(N_CHANNELS, cnn_filters // 2, dropout_p=dropout_p)
        self.res2 = ResidualCNNBlock(cnn_filters // 2, cnn_filters, dropout_p=dropout_p)
        self.res3 = ResidualCNNBlock(cnn_filters, cnn_filters, dropout_p=dropout_p)
        self.fc = nn.Linear(cnn_filters, 5)
    def forward(self, x):
        x = self.res3(self.res2(self.res1(x)))
        return {"emotion_logits": self.fc(x.mean(dim=2))}

class CNNBiLSTMAblationNet(nn.Module):
    def __init__(self, cnn_filters=128, lstm_units=64, dropout_p=0.3):
        super().__init__()
        self.res1 = ResidualCNNBlock(N_CHANNELS, cnn_filters // 2, dropout_p=dropout_p)
        self.res2 = ResidualCNNBlock(cnn_filters // 2, cnn_filters, dropout_p=dropout_p)
        self.res3 = ResidualCNNBlock(cnn_filters, cnn_filters, dropout_p=dropout_p)
        self.lstm = nn.LSTM(cnn_filters, lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.fc = nn.Linear(lstm_units * 2, 5)
    def forward(self, x):
        x = self.res3(self.res2(self.res1(x))).transpose(1, 2)
        out, _ = self.lstm(x)
        return {"emotion_logits": self.fc(out.mean(dim=1))}

# ─── Strict LOSO Evaluator ────────────────────────────────────────────────────
def run_loso(model_fn, X, y_task, subjects, lr=3e-4, weight_decay=1e-3, epochs=35, label_smoothing=0.05, class_weights=True):
    unique_subs = sorted(list(set(subjects)))
    metrics_list = []
    all_targets, all_preds, all_probs = [], [], []
    
    for fold_i, test_s in enumerate(unique_subs):
        tr_m = (subjects != test_s)
        te_m = (subjects == test_s)
        assert len(set(subjects[tr_m]).intersection({test_s})) == 0, "Leakage!"
        
        X_tr_raw, y_tr_raw = X[tr_m].numpy(), y_task[tr_m]
        X_te_raw, y_te = X[te_m].numpy(), y_task[te_m].numpy()
        
        # 80/20 inner split on training subjects only
        in_tr_idx, in_val_idx = train_test_split(np.arange(len(X_tr_raw)), test_size=0.2, stratify=y_tr_raw.numpy(), random_state=42)
        
        scaler = StandardScaler()
        X_in_tr_sc = torch.tensor(scaler.fit_transform(X_tr_raw[in_tr_idx]), dtype=torch.float32)
        X_in_val_sc = torch.tensor(scaler.transform(X_tr_raw[in_val_idx]), dtype=torch.float32).to(device)
        X_te_sc = torch.tensor(scaler.transform(X_te_raw), dtype=torch.float32).to(device)
        
        y_in_tr = y_tr_raw[in_tr_idx]
        y_in_val = y_tr_raw[in_val_idx].to(device)
        
        # Balance inner train
        dummy_reg = torch.zeros((len(X_in_tr_sc), 7))
        X_in_tr_b, _, y_in_tr_b = balance_training_data(X_in_tr_sc, dummy_reg, y_in_tr)
        
        model = model_fn().to(device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        
        if class_weights:
            counts = torch.bincount(y_in_tr, minlength=5).float()
            w = (1.0 / (counts + 1e-5))
            w = (w / w.sum() * 5.0).to(device)
            crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
        loader = DataLoader(TensorDataset(X_in_tr_b, y_in_tr_b), batch_size=32, shuffle=True)
        
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
        
        for ep in range(epochs):
            model.train()
            for bx, by in loader:
                bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                by = by.to(device)
                opt.zero_grad()
                out = model(bx)
                loss = crit(out["emotion_logits"], by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
            model.eval()
            with torch.no_grad():
                val_out = model(X_in_val_sc.view(-1, N_CHANNELS, FEATURES_PER_CH))
                v_loss = crit(val_out["emotion_logits"], y_in_val).item()
                sched.step(v_loss)
                if v_loss < best_val:
                    best_val = v_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10:
                        break
                        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_out = model(X_te_sc.view(-1, N_CHANNELS, FEATURES_PER_CH))
            probs = torch.softmax(te_out["emotion_logits"], dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average="macro", zero_division=0) * 100
        roc = compute_multiclass_roc_auc(y_te, probs)
        kappa = cohen_kappa_score(y_te, preds)
        metrics_list.append({"subj": int(test_s), "acc": acc, "f1": f1, "roc": roc, "kappa": kappa})
        all_targets.extend(y_te.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        log(f"  Fold {fold_i+1} (Subj {test_s}): Acc = {acc:.2f}%, Macro F1 = {f1:.2f}%, ROC = {roc:.4f}")
        
    m_acc = float(np.mean([m["acc"] for m in metrics_list]))
    s_acc = float(np.std([m["acc"] for m in metrics_list]))
    m_f1 = float(np.mean([m["f1"] for m in metrics_list]))
    m_roc = float(np.mean([m["roc"] for m in metrics_list]))
    m_kappa = float(np.mean([m["kappa"] for m in metrics_list]))
    m_mcc = float(matthews_corrcoef(all_targets, all_preds))
    cm = confusion_matrix(all_targets, all_preds, labels=[0,1,2,3,4]).tolist()
    
    log(f"--> LOSO Result: Acc = {m_acc:.2f}% ± {s_acc:.2f}%, F1 = {m_f1:.2f}%, ROC = {m_roc:.4f}, Kappa = {m_kappa:+.4f}\n")
    return {
        "mean_accuracy": round(m_acc, 2),
        "std_accuracy": round(s_acc, 2),
        "mean_macro_f1": round(m_f1, 2),
        "mean_roc_auc": round(m_roc, 4),
        "mean_kappa": round(m_kappa, 4),
        "mcc": round(m_mcc, 4),
        "per_subject": metrics_list,
        "confusion_matrix": cm
    }

def main():
    log("=" * 70)
    log("FAST OPTIMIZED EXPERIMENT & ABLATION SUITE (STRICT LOSO)")
    log("=" * 70)
    
    # 1. Load standard 3-subject cohort
    log("Loading standard 3-subject benchmark...")
    X3, y3, sub3 = load_cohort([1, 2, 3])
    log(f"Loaded 3-subject cohort: N = {len(X3)} windows")
    
    # Classical Baselines
    log("\n--- Classical Baselines (3 Subjects) ---")
    baselines = {}
    for name, clf in [
        ("RBF SVM", SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)),
        ("Linear SVM", SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    ]:
        f_accs, f_f1s = [], []
        targs, prds = [], []
        for s in [1, 2, 3]:
            tr_m, te_m = (sub3 != s), (sub3 == s)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X3[tr_m].numpy())
            X_te = sc.transform(X3[te_m].numpy())
            clf.fit(X_tr, y3[tr_m].numpy())
            p = clf.predict(X_te)
            f_accs.append(accuracy_score(y3[te_m].numpy(), p)*100)
            f_f1s.append(f1_score(y3[te_m].numpy(), p, average="macro", zero_division=0)*100)
            targs.extend(y3[te_m].tolist())
            prds.extend(p.tolist())
        log(f"  {name:20s}: Acc = {np.mean(f_accs):.2f}% ± {np.std(f_accs):.2f}%, F1 = {np.mean(f_f1s):.2f}%")
        baselines[name] = {"accuracy": round(float(np.mean(f_accs)), 2), "std": round(float(np.std(f_accs)), 2), "f1": round(float(np.mean(f_f1s)), 2)}
        
    # Phase 9: Ablation Study
    log("\n" + "=" * 50)
    log("PHASE 9: ABLATION STUDY")
    log("=" * 50)
    
    log("\nAblation 1: 1D Residual CNN Extractor Only + Head")
    res_cnn = run_loso(lambda: CNNAblationNet(128, 0.3), X3, y3, sub3)
    
    log("\nAblation 2: Residual CNN + BiLSTM + Head")
    res_cnn_lstm = run_loso(lambda: CNNBiLSTMAblationNet(128, 64, 0.3), X3, y3, sub3)
    
    log("\nAblation 3: Full HybridBCINet (CNN + BiLSTM + Attention + Transformer, 731k)")
    res_hybrid = run_loso(lambda: HybridBCINet(INPUT_FEATURES, N_CHANNELS, 128, 64, 8, 0.4), X3, y3, sub3)
    
    # Phase 6 & 7: Optimized Hyperparameters
    log("\n" + "=" * 50)
    log("PHASE 6 & 7: OPTIMIZED REGULARIZATION & TUNING")
    log("=" * 50)
    
    log("\nOptimized Config: HybridBCINet with Dropout=0.45, Weight Decay=1e-2, Label Smoothing=0.05")
    res_opt = run_loso(lambda: HybridBCINet(INPUT_FEATURES, N_CHANNELS, 128, 64, 8, 0.45), X3, y3, sub3, lr=2e-4, weight_decay=1e-2, label_smoothing=0.05)
    
    # Phase 3: Scaled Cohort (5 Subjects: 1, 2, 3, 4, 5)
    log("\n" + "=" * 50)
    log("PHASE 3: SCALED COHORT (5 Subjects: 1, 2, 3, 4, 5)")
    log("=" * 50)
    X5, y5, sub5 = load_cohort([1, 2, 3, 4, 5])
    log(f"Loaded 5-subject cohort: N = {len(X5)} windows")
    
    log("\n--- Classical Baselines (5 Subjects) ---")
    baselines_5 = {}
    for name, clf in [
        ("RBF SVM", SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    ]:
        f_accs, f_f1s = [], []
        for s in [1, 2, 3, 4, 5]:
            tr_m, te_m = (sub5 != s), (sub5 == s)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X5[tr_m].numpy())
            X_te = sc.transform(X5[te_m].numpy())
            clf.fit(X_tr, y5[tr_m].numpy())
            p = clf.predict(X_te)
            f_accs.append(accuracy_score(y5[te_m].numpy(), p)*100)
            f_f1s.append(f1_score(y5[te_m].numpy(), p, average="macro", zero_division=0)*100)
        log(f"  {name:20s}: Acc = {np.mean(f_accs):.2f}% ± {np.std(f_accs):.2f}%, F1 = {np.mean(f_f1s):.2f}%")
        baselines_5[name] = {"accuracy": round(float(np.mean(f_accs)), 2), "std": round(float(np.std(f_accs)), 2), "f1": round(float(np.mean(f_f1s)), 2)}
        
    log("\nHybridBCINet on 5-Subject Cohort (Strict 5-Fold LOSO):")
    res_hybrid_5 = run_loso(lambda: HybridBCINet(INPUT_FEATURES, N_CHANNELS, 128, 64, 8, 0.4), X5, y5, sub5, lr=3e-4, weight_decay=1e-3)
    
    # Consolidate all results
    full_report = {
        "baselines_3_subjects": baselines,
        "ablation_study_3_subjects": {
            "CNN_Only": res_cnn,
            "CNN_BiLSTM": res_cnn_lstm,
            "Full_HybridBCINet_731k": res_hybrid
        },
        "optimized_hybridbcinet_3_subjects": res_opt,
        "scaled_cohort_5_subjects": {
            "baselines": baselines_5,
            "HybridBCINet_5_subjects": res_hybrid_5
        }
    }
    
    out_path = "backend/weights/improved_loso_experiments.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
    log(f"\nAll experiments successfully written to {out_path}")
    
    # Save improved model weights
    torch.save(res_opt, "backend/weights/bci_model_improved_loso_metrics.json")
    log("Saved improved LOSO metrics.")

if __name__ == "__main__":
    main()
