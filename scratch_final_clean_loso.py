import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from train import extract_real_eeg_features, balance_training_data, TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH, WEIGHTS_DIR
from model import HybridBCINet

def compute_multiclass_roc_auc(y_true, y_probs):
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        return float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    except Exception:
        return 0.5

print("=" * 70, flush=True)
print("FINAL RIGOROUS CLEAN LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION", flush=True)
print("Protocol: Strict zero-leakage subject independence, internal val for model selection", flush=True)
print("=" * 70, flush=True)

# 1. Load genuine PhysioNet EEGBCI data
X_raw_tensor, y_reg_raw, y_task_raw, subject_arr, run_arr = extract_real_eeg_features(window_size_sec=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_raw_np = X_raw_tensor.numpy()
y_task_np = y_task_raw.numpy()

# Compress extreme energy features using sign-preserving log1p
X_log_np = np.sign(X_raw_np) * np.log1p(np.abs(X_raw_np))

# -------------------------------------------------------------
# Part 1: Improved HybridBCINet Clean LOSO Evaluation
# -------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("EVALUATING IMPROVED HYBRIDBCINET (REGULARIZED + CLEAN OBJECTIVE)", flush=True)
print("=" * 60, flush=True)

improved_results = {}
imp_all_preds, imp_all_targets, imp_all_probs = [], [], []

for test_subj in [1, 2, 3]:
    print(f"\n--- FOLD {test_subj}/3: Test Subject = {test_subj} ---", flush=True)
    
    te_mask = (subject_arr == test_subj)
    tr_pool_mask = (subject_arr != test_subj)
    
    X_test_raw = X_log_np[te_mask]
    y_test = y_task_np[te_mask]
    
    X_pool_raw = X_log_np[tr_pool_mask]
    y_pool = y_task_raw[tr_pool_mask]
    
    # Internal train / validation split (80% train, 20% validation)
    pool_indices = np.arange(len(X_pool_raw))
    in_tr_idx, in_val_idx = train_test_split(
        pool_indices, test_size=0.2, stratify=y_pool.numpy(), random_state=42
    )
    
    X_in_tr_raw = X_pool_raw[in_tr_idx]
    y_in_tr_raw = y_pool[in_tr_idx]
    
    X_in_val_raw = X_pool_raw[in_val_idx]
    y_in_val = y_pool[in_val_idx].numpy()
    
    # Fit scaler ONLY on internal training split
    scaler = StandardScaler()
    X_in_tr_scaled = torch.tensor(scaler.fit_transform(X_in_tr_raw), dtype=torch.float32)
    X_in_val_scaled = torch.tensor(scaler.transform(X_in_val_raw), dtype=torch.float32).to(device)
    X_test_scaled = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(device)
    
    # Balance ONLY the internal training portion
    # Create dummy regression target for balance_training_data compatibility
    dummy_reg = torch.zeros((len(X_in_tr_scaled), 4))
    X_in_tr, _, y_in_tr = balance_training_data(X_in_tr_scaled, dummy_reg, y_in_tr_raw)
    
    # Regularized compact HybridBCINet architecture
    model = HybridBCINet(
        input_features=INPUT_FEATURES,
        num_channels=N_CHANNELS,
        cnn_filters=64,
        lstm_units=32,
        num_heads=4,
        dropout_p=0.45
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    train_loader = DataLoader(
        TensorDataset(X_in_tr, y_in_tr),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_in_val_scaled, torch.tensor(y_in_val, dtype=torch.long)),
        batch_size=32,
        shuffle=False
    )
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    # Train and select best epoch STRICTLY using Internal Validation
    for epoch in range(1, 36):
        model.train()
        for bx, by in train_loader:
            bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
            by = by.to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion_cls(out["emotion_logits"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Internal validation evaluation
        model.eval()
        v_loss = 0.0
        v_preds, v_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                by = by.to(device)
                out = model(bx)
                loss = criterion_cls(out["emotion_logits"], by)
                v_loss += loss.item()
                v_preds.extend(torch.argmax(out["emotion_logits"], dim=1).cpu().numpy())
                v_targets.extend(by.cpu().numpy())
                
        v_loss /= max(len(val_loader), 1)
        v_acc = accuracy_score(v_targets, v_preds) * 100
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"  Best Internal Val Epoch: {best_epoch} (Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.2f}%)", flush=True)
    
    # Evaluate frozen model ONCE on untouched held-out test subject
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    with torch.no_grad():
        out_test = model(X_test_scaled.view(-1, N_CHANNELS, FEATURES_PER_CH))
        t_logits = out_test["emotion_logits"]
        t_probs = torch.softmax(t_logits, dim=1).cpu().numpy()
        t_preds = torch.argmax(t_logits, dim=1).cpu().numpy()
        
    t_acc = accuracy_score(y_test, t_preds) * 100
    t_prec = precision_score(y_test, t_preds, average="macro", zero_division=0) * 100
    t_rec = recall_score(y_test, t_preds, average="macro", zero_division=0) * 100
    t_f1 = f1_score(y_test, t_preds, average="macro", zero_division=0) * 100
    t_auc = compute_multiclass_roc_auc(y_test, t_probs)
    t_kappa = float(cohen_kappa_score(y_test, t_preds))
    t_mcc = float(matthews_corrcoef(y_test, t_preds))
    cm = confusion_matrix(y_test, t_preds, labels=[0, 1, 2, 3, 4])
    p_dist = np.bincount(t_preds, minlength=5).tolist()
    
    improved_results[f"Subject_{test_subj}"] = {
        "selected_epoch": best_epoch,
        "internal_val_acc": round(best_val_acc, 2),
        "internal_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(t_acc, 2),
        "test_macro_precision": round(t_prec, 2),
        "test_macro_recall": round(t_rec, 2),
        "test_macro_f1": round(t_f1, 2),
        "test_roc_auc": round(t_auc, 4),
        "test_cohens_kappa": round(t_kappa, 4),
        "test_mcc": round(t_mcc, 4),
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": p_dist,
    }
    
    imp_all_preds.extend(t_preds)
    imp_all_targets.extend(y_test)
    imp_all_probs.extend(t_probs)
    
    print(f"  Test Subject {test_subj} Results: Acc = {t_acc:.2f}%, Macro F1 = {t_f1:.2f}%, AUC = {t_auc:.4f}, Kappa = {t_kappa:.4f}, MCC = {t_mcc:.4f}", flush=True)
    print(f"  Prediction Distribution: {p_dist}", flush=True)

imp_accs = [improved_results[f"Subject_{s}"]["test_accuracy"] for s in [1, 2, 3]]
imp_precs = [improved_results[f"Subject_{s}"]["test_macro_precision"] for s in [1, 2, 3]]
imp_recs = [improved_results[f"Subject_{s}"]["test_macro_recall"] for s in [1, 2, 3]]
imp_f1s = [improved_results[f"Subject_{s}"]["test_macro_f1"] for s in [1, 2, 3]]
imp_aucs = [improved_results[f"Subject_{s}"]["test_roc_auc"] for s in [1, 2, 3]]
imp_kappas = [improved_results[f"Subject_{s}"]["test_cohens_kappa"] for s in [1, 2, 3]]
imp_mccs = [improved_results[f"Subject_{s}"]["test_mcc"] for s in [1, 2, 3]]

imp_overall_cm = confusion_matrix(imp_all_targets, imp_all_preds, labels=[0, 1, 2, 3, 4])
imp_overall_dist = np.bincount(imp_all_preds, minlength=5).tolist()

improved_summary = {
    "mean_accuracy": round(float(np.mean(imp_accs)), 2),
    "std_accuracy": round(float(np.std(imp_accs)), 2),
    "mean_macro_precision": round(float(np.mean(imp_precs)), 2),
    "mean_macro_recall": round(float(np.mean(imp_recs)), 2),
    "mean_macro_f1": round(float(np.mean(imp_f1s)), 2),
    "mean_roc_auc": round(float(np.mean(imp_aucs)), 4),
    "mean_cohens_kappa": round(float(np.mean(imp_kappas)), 4),
    "mean_mcc": round(float(np.mean(imp_mccs)), 4),
    "overall_confusion_matrix": imp_overall_cm.tolist(),
    "overall_prediction_distribution": imp_overall_dist,
}

print("\n" + "=" * 60, flush=True)
print(f"IMPROVED HYBRIDBCINET FINAL RESULTS:", flush=True)
print(f"Mean Accuracy: {improved_summary['mean_accuracy']:.2f}% ± {improved_summary['std_accuracy']:.2f}% (Chance: 20.00%)", flush=True)
print(f"Mean Macro Precision: {improved_summary['mean_macro_precision']:.2f}%", flush=True)
print(f"Mean Macro Recall: {improved_summary['mean_macro_recall']:.2f}%", flush=True)
print(f"Mean Macro F1-Score: {improved_summary['mean_macro_f1']:.2f}%", flush=True)
print(f"Mean Multiclass ROC-AUC: {improved_summary['mean_roc_auc']:.4f}", flush=True)
print(f"Mean Cohen's Kappa: {improved_summary['mean_cohens_kappa']:.4f}", flush=True)
print(f"Mean MCC: {improved_summary['mean_mcc']:.4f}", flush=True)
print(f"Overall Prediction Distribution (N=735): {imp_overall_dist}", flush=True)
print(f"Overall Confusion Matrix:\n{imp_overall_cm}", flush=True)
print("=" * 60, flush=True)

# -------------------------------------------------------------
# Part 2: RBF SVM Classical Baseline
# -------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("EVALUATING CLASSICAL RBF SVM BASELINE (SAME SPLITS)", flush=True)
print("=" * 60, flush=True)

svm_results = {}
svm_all_preds, svm_all_targets = [], []
for test_subj in [1, 2, 3]:
    te_mask = (subject_arr == test_subj)
    tr_mask = (subject_arr != test_subj)
    
    X_tr = X_raw_np[tr_mask]
    y_tr = y_task_np[tr_mask]
    X_te = X_raw_np[te_mask]
    y_te = y_task_np[te_mask]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    clf = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)
    
    s_acc = accuracy_score(y_te, preds) * 100
    s_prec = precision_score(y_te, preds, average="macro", zero_division=0) * 100
    s_rec = recall_score(y_te, preds, average="macro", zero_division=0) * 100
    s_f1 = f1_score(y_te, preds, average="macro", zero_division=0) * 100
    s_kappa = float(cohen_kappa_score(y_te, preds))
    s_mcc = float(matthews_corrcoef(y_te, preds))
    s_cm = confusion_matrix(y_te, preds, labels=[0, 1, 2, 3, 4])
    s_dist = np.bincount(preds, minlength=5).tolist()
    
    svm_results[f"Subject_{test_subj}"] = {
        "test_accuracy": round(s_acc, 2),
        "test_macro_precision": round(s_prec, 2),
        "test_macro_recall": round(s_rec, 2),
        "test_macro_f1": round(s_f1, 2),
        "test_cohens_kappa": round(s_kappa, 4),
        "test_mcc": round(s_mcc, 4),
        "confusion_matrix": s_cm.tolist(),
        "prediction_distribution": s_dist,
    }
    svm_all_preds.extend(preds)
    svm_all_targets.extend(y_te)
    print(f"  RBF SVM Subj {test_subj}: Acc = {s_acc:.2f}%, Macro F1 = {s_f1:.2f}%, Kappa = {s_kappa:.4f}, Dist = {s_dist}", flush=True)

svm_accs = [svm_results[f"Subject_{s}"]["test_accuracy"] for s in [1, 2, 3]]
svm_f1s = [svm_results[f"Subject_{s}"]["test_macro_f1"] for s in [1, 2, 3]]
svm_precs = [svm_results[f"Subject_{s}"]["test_macro_precision"] for s in [1, 2, 3]]
svm_recs = [svm_results[f"Subject_{s}"]["test_macro_recall"] for s in [1, 2, 3]]
svm_kappas = [svm_results[f"Subject_{s}"]["test_cohens_kappa"] for s in [1, 2, 3]]
svm_mccs = [svm_results[f"Subject_{s}"]["test_mcc"] for s in [1, 2, 3]]

svm_overall_cm = confusion_matrix(svm_all_targets, svm_all_preds, labels=[0, 1, 2, 3, 4])
svm_overall_dist = np.bincount(svm_all_preds, minlength=5).tolist()

svm_summary = {
    "mean_accuracy": round(float(np.mean(svm_accs)), 2),
    "std_accuracy": round(float(np.std(svm_accs)), 2),
    "mean_macro_precision": round(float(np.mean(svm_precs)), 2),
    "mean_macro_recall": round(float(np.mean(svm_recs)), 2),
    "mean_macro_f1": round(float(np.mean(svm_f1s)), 2),
    "mean_cohens_kappa": round(float(np.mean(svm_kappas)), 4),
    "mean_mcc": round(float(np.mean(svm_mccs)), 4),
    "overall_confusion_matrix": svm_overall_cm.tolist(),
    "overall_prediction_distribution": svm_overall_dist,
}

print("\n" + "=" * 60, flush=True)
print(f"RBF SVM FINAL RESULTS:", flush=True)
print(f"Mean Accuracy: {svm_summary['mean_accuracy']:.2f}% ± {svm_summary['std_accuracy']:.2f}%", flush=True)
print(f"Mean Macro F1-Score: {svm_summary['mean_macro_f1']:.2f}%", flush=True)
print(f"Mean Cohen's Kappa: {svm_summary['mean_cohens_kappa']:.4f}", flush=True)
print(f"Overall Prediction Distribution: {svm_overall_dist}", flush=True)
print(f"Overall Confusion Matrix:\n{svm_overall_cm}", flush=True)
print("=" * 60, flush=True)

# -------------------------------------------------------------
# Save Results and Checkpoint
# -------------------------------------------------------------
final_data = {
    "improved_hybridbcinet": {
        "per_subject": improved_results,
        "aggregate": improved_summary,
    },
    "rbf_svm_baseline": {
        "per_subject": svm_results,
        "aggregate": svm_summary,
    },
    "chance_level": 20.00
}

# Save results json
results_file = os.path.join(WEIGHTS_DIR, "loso_clean_results.json")
with open(results_file, "w") as f:
    json.dump(final_data, f, indent=2)

# Save clean checkpoint separately
clean_ckpt_path = os.path.join(WEIGHTS_DIR, "bci_model_loso_clean.pth")
torch.save(best_model_state, clean_ckpt_path)

# Update best_config.json
config_data = {
    "cnn_filters": 64,
    "lstm_units": 32,
    "num_heads": 4,
    "dropout_p": 0.45,
    "lr": 0.0005,
    "weight_decay": 0.03
}
with open(os.path.join(WEIGHTS_DIR, "best_config.json"), "w") as f:
    json.dump(config_data, f, indent=2)

print(f"\nSaved clean evaluation results to: {results_file}", flush=True)
print(f"Saved clean model checkpoint to: {clean_ckpt_path}", flush=True)
print("Validation complete successfully.", flush=True)
