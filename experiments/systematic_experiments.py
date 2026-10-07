"""
scratch_systematic_experiments.py — Scientific Optimization & Ablation Suite
Conducts Phases 2-11 with strict Leave-One-Subject-Out (LOSO) isolation.
Zero data leakage:
- Preprocessing and scalers fitted strictly on training subjects.
- Model selection performed strictly on inner validation splits within training subjects.
- Held-out test subjects evaluated strictly once per fold on frozen checkpoints.
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def compute_multiclass_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        if y_true_bin.shape[1] == 1:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr"))
    except Exception:
        return 0.5

# ─── 1. Scalable Dataset Extraction Function ──────────────────────────────────
def extract_cohort_eeg_features(subjects: list[int] = [1, 2, 3], window_size_sec: float = 2.0, l_freq: float = 0.5, h_freq: float = 40.0):
    loader = BCIDatasetLoader()
    physionet_mapping = {1: 0, 4: 1, 6: 2, 2: 3, 3: 4}
    
    X_list, y_task_list, sub_list, run_list = [], [], [], []
    hashes = set()
    
    for subject in subjects:
        for run, class_id in physionet_mapping.items():
            try:
                data, sfreq = loader.load_physionet(subject=subject, run=run)
                chunk_size = int(sfreq * window_size_sec)
                
                if data.shape[0] > N_CHANNELS:
                    data = data[:N_CHANNELS]
                elif data.shape[0] < N_CHANNELS:
                    data = np.vstack((data, np.zeros((N_CHANNELS - data.shape[0], data.shape[1]))))
                
                n_chunks = data.shape[1] // chunk_size
                for i in range(n_chunks):
                    window = data[:, i * chunk_size : (i + 1) * chunk_size]
                    features = extract_features_from_window(window, sfreq=sfreq)
                    
                    if len(features) < INPUT_FEATURES:
                        features = np.pad(features, (0, INPUT_FEATURES - len(features)))
                    else:
                        features = features[:INPUT_FEATURES]
                    
                    h = hash(features.tobytes())
                    if h not in hashes:
                        hashes.add(h)
                        X_list.append(features)
                        y_task_list.append(class_id)
                        sub_list.append(subject)
                        run_list.append(run)
            except Exception as err:
                print(f"Failed loading subj={subject}, run={run}: {err}")
                
    X = np.array(X_list, dtype=np.float32)
    y_task = np.array(y_task_list, dtype=np.int64)
    subject_arr = np.array(sub_list, dtype=np.int32)
    run_arr = np.array(run_list, dtype=np.int32)
    
    # Continuous reference proxy targets for regression heads:
    _REG_PROXY = {
        0: [0.45, 0.45, 0.20, 0.35, 0.30, 0.65, 0.30],
        1: [0.75, 0.80, 0.35, 0.25, 0.65, 0.70, 0.60],
        2: [0.40, 0.50, 0.75, 0.55, 0.70, 0.25, 0.75],
        3: [0.30, 0.30, 0.40, 0.80, 0.45, 0.30, 0.25],
        4: [0.65, 0.70, 0.50, 0.20, 0.55, 0.75, 0.85],
    }
    rng = np.random.default_rng(seed=42)
    y_reg_base = np.array([_REG_PROXY[int(e)] for e in y_task], dtype=np.float32)
    y_reg = np.clip(y_reg_base + rng.normal(0.0, 0.05, y_reg_base.shape).astype(np.float32), 0.0, 1.0)
    
    return torch.tensor(X), torch.tensor(y_reg), torch.tensor(y_task), subject_arr, run_arr

# ─── 2. Safe Training-Only Augmentation ────────────────────────────────────────
def safe_train_augmentation(X_train: torch.Tensor, y_task: torch.Tensor, noise_lvl: float = 0.03, drop_prob: float = 0.15):
    X_aug = X_train.clone()
    # Gaussian noise
    noise = torch.randn_like(X_aug) * noise_lvl * (X_aug.std(dim=0, keepdim=True) + 1e-8)
    X_aug += noise
    
    # Channel dropout in 3D feature representation
    X_aug_3d = X_aug.view(-1, N_CHANNELS, FEATURES_PER_CH)
    if random.random() < drop_prob:
        drop_idx = torch.randperm(N_CHANNELS)[:random.randint(1, 2)]
        X_aug_3d[:, drop_idx, :] = 0.0
    return X_aug_3d.view(-1, INPUT_FEATURES), y_task

# ─── 3. Ablation Architectures ────────────────────────────────────────────────
class CNNAblationNet(nn.Module):
    def __init__(self, cnn_filters=128, dropout_p=0.3):
        super().__init__()
        self.res1 = ResidualCNNBlock(N_CHANNELS, cnn_filters // 2, dropout_p=dropout_p)
        self.res2 = ResidualCNNBlock(cnn_filters // 2, cnn_filters, dropout_p=dropout_p)
        self.res3 = ResidualCNNBlock(cnn_filters, cnn_filters, dropout_p=dropout_p)
        self.fc_emotion = nn.Linear(cnn_filters, 5)
        self.fc_focus = nn.Linear(cnn_filters, 1)
        self.fc_stress = nn.Linear(cnn_filters, 1)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x) # (B, cnn_filters, features_per_ch)
        out = x.mean(dim=2) # GAP
        return {
            "emotion_logits": self.fc_emotion(out),
            "focus": torch.sigmoid(self.fc_focus(out)),
            "stress": torch.sigmoid(self.fc_stress(out))
        }

class CNNBiLSTMAblationNet(nn.Module):
    def __init__(self, cnn_filters=128, lstm_units=64, dropout_p=0.3):
        super().__init__()
        self.res1 = ResidualCNNBlock(N_CHANNELS, cnn_filters // 2, dropout_p=dropout_p)
        self.res2 = ResidualCNNBlock(cnn_filters // 2, cnn_filters, dropout_p=dropout_p)
        self.res3 = ResidualCNNBlock(cnn_filters, cnn_filters, dropout_p=dropout_p)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_p)
        embed_dim = lstm_units * 2
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_emotion = nn.Linear(embed_dim, 5)
        self.fc_focus = nn.Linear(embed_dim, 1)
        self.fc_stress = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = self.ln(lstm_out.mean(dim=1))
        return {
            "emotion_logits": self.fc_emotion(out),
            "focus": torch.sigmoid(self.fc_focus(out)),
            "stress": torch.sigmoid(self.fc_stress(out))
        }

# ─── 4. Unified Strict LOSO Evaluation Function ───────────────────────────────
def evaluate_model_loso(model_fn, X: torch.Tensor, y_reg: torch.Tensor, y_task: torch.Tensor, 
                        subject_arr: np.ndarray, config: dict, use_aug: bool = False, 
                        label_smoothing: float = 0.0, class_weights: bool = False):
    unique_subjects = sorted(list(set(subject_arr)))
    n_folds = len(unique_subjects)
    
    fold_test_metrics = []
    all_targets = []
    all_preds = []
    all_probs = []
    
    print(f"\n--- Running Strict LOSO Evaluation across {n_folds} subjects: {unique_subjects} ---")
    
    for fold_idx, test_subj in enumerate(unique_subjects):
        # 1. Leakage assertion
        tr_mask = (subject_arr != test_subj)
        te_mask = (subject_arr == test_subj)
        assert np.sum(tr_mask) > 0 and np.sum(te_mask) > 0, "Empty train or test partition!"
        assert len(set(subject_arr[tr_mask]).intersection(set([test_subj]))) == 0, "Subject leakage detected!"
        
        X_tr_raw = X[tr_mask].numpy()
        y_reg_tr_raw = y_reg[tr_mask]
        y_tr_raw = y_task[tr_mask]
        
        X_te_raw = X[te_mask].numpy()
        y_te = y_task[te_mask].numpy()
        
        # 2. Inner train / validation split on training subjects only (80/20)
        pool_idx = np.arange(len(X_tr_raw))
        in_tr_idx, in_val_idx = train_test_split(
            pool_idx, test_size=0.2, stratify=y_tr_raw.numpy(), random_state=42
        )
        
        X_in_tr_raw = X_tr_raw[in_tr_idx]
        y_reg_in_tr_raw = y_reg_tr_raw[in_tr_idx]
        y_in_tr_raw = y_tr_raw[in_tr_idx]
        
        X_in_val_raw = X_tr_raw[in_val_idx]
        y_reg_in_val = y_reg_tr_raw[in_val_idx].to(device)
        y_in_val = y_tr_raw[in_val_idx].to(device)
        
        # 3. Fit scaler ONLY on inner training split
        scaler = StandardScaler()
        X_in_tr_scaled = torch.tensor(scaler.fit_transform(X_in_tr_raw), dtype=torch.float32)
        X_in_val_scaled = torch.tensor(scaler.transform(X_in_val_raw), dtype=torch.float32).to(device)
        X_te_scaled = torch.tensor(scaler.transform(X_te_raw), dtype=torch.float32).to(device)
        
        # 4. Balance ONLY inner training split
        X_in_tr_b, y_reg_in_tr_b, y_in_tr_b = balance_training_data(
            X_in_tr_scaled, y_reg_in_tr_raw, y_in_tr_raw
        )
        
        # 5. Build model
        model = model_fn().to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=config.get("lr", 3e-4), weight_decay=config.get("weight_decay", 1e-4)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        if class_weights:
            cls_counts = torch.bincount(y_in_tr_raw, minlength=5).float()
            weights = (1.0 / (cls_counts + 1e-5))
            weights = (weights / weights.sum() * 5.0).to(device)
            criterion_cls = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_reg = nn.MSELoss()
        
        train_ds = TensorDataset(X_in_tr_b, y_reg_in_tr_b, y_in_tr_b)
        train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True)
        
        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(model.state_dict())
        patience = 0
        max_patience = config.get("early_stopping_patience", 15)
        epochs = config.get("epochs", 50)
        
        # 6. Training Loop (Model selection via INNER VALIDATION ONLY)
        for epoch in range(epochs):
            model.train()
            for bx, by_reg, by_task in train_loader:
                if use_aug:
                    bx, by_task = safe_train_augmentation(bx, by_task)
                bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                by_reg = by_reg.to(device)
                by_task = by_task.to(device)
                
                optimizer.zero_grad()
                out = model(bx)
                loss_cls = criterion_cls(out["emotion_logits"], by_task)
                loss_reg = criterion_reg(out["focus"], by_reg[:, 0:1]) + criterion_reg(out["stress"], by_reg[:, 2:3])
                total_loss = loss_cls + 0.1 * loss_reg
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Inner validation evaluation
            model.eval()
            with torch.no_grad():
                val_bx = X_in_val_scaled.view(-1, N_CHANNELS, FEATURES_PER_CH)
                val_out = model(val_bx)
                val_loss = criterion_cls(val_out["emotion_logits"], y_in_val).item()
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        break
                        
        # 7. Evaluate frozen model on held-out test subject EXACTLY ONCE
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            te_bx = X_te_scaled.view(-1, N_CHANNELS, FEATURES_PER_CH)
            te_out = model(te_bx)
            probs = torch.softmax(te_out["emotion_logits"], dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
        acc = accuracy_score(y_te, preds) * 100
        prec = precision_score(y_te, preds, average="macro", zero_division=0) * 100
        rec = recall_score(y_te, preds, average="macro", zero_division=0) * 100
        f1 = f1_score(y_te, preds, average="macro", zero_division=0) * 100
        roc = compute_multiclass_roc_auc(y_te, probs)
        kappa = cohen_kappa_score(y_te, preds)
        mcc = matthews_corrcoef(y_te, preds)
        
        fold_test_metrics.append({
            "subject": int(test_subj),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "kappa": kappa,
            "mcc": mcc,
            "confusion_matrix": confusion_matrix(y_te, preds, labels=[0,1,2,3,4]).tolist()
        })
        all_targets.extend(y_te.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        print(f"  Fold {fold_idx+1} (Subj {test_subj}): Test Acc = {acc:.2f}%, Macro F1 = {f1:.2f}%, ROC-AUC = {roc:.4f}")
        
    mean_acc = np.mean([m["accuracy"] for m in fold_test_metrics])
    std_acc = np.std([m["accuracy"] for m in fold_test_metrics])
    mean_f1 = np.mean([m["f1"] for m in fold_test_metrics])
    mean_roc = np.mean([m["roc_auc"] for m in fold_test_metrics])
    mean_kappa = np.mean([m["kappa"] for m in fold_test_metrics])
    mean_mcc = np.mean([m["mcc"] for m in fold_test_metrics])
    
    print(f"==> OVERALL LOSO: Acc = {mean_acc:.2f}% ± {std_acc:.2f}%, Macro F1 = {mean_f1:.2f}%, ROC-AUC = {mean_roc:.4f}, Kappa = {mean_kappa:+.4f}")
    return {
        "mean_accuracy": round(float(mean_acc), 2),
        "std_accuracy": round(float(std_acc), 2),
        "mean_macro_f1": round(float(mean_f1), 2),
        "mean_roc_auc": round(float(mean_roc), 4),
        "mean_kappa": round(float(mean_kappa), 4),
        "mean_mcc": round(float(mean_mcc), 4),
        "per_subject": fold_test_metrics,
        "overall_confusion_matrix": confusion_matrix(all_targets, all_preds, labels=[0,1,2,3,4]).tolist()
    }

# ─── 5. Classical Baselines Evaluator ─────────────────────────────────────────
def evaluate_classical_baselines(X: torch.Tensor, y_task: torch.Tensor, subject_arr: np.ndarray):
    X_np = X.numpy()
    y_np = y_task.numpy()
    unique_subjects = sorted(list(set(subject_arr)))
    
    baselines = {
        "RBF SVM (Balanced, C=1.0)": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True),
        "Linear SVM (Balanced, C=0.1)": SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Logistic Regression (C=1.0)": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    baseline_results = {}
    print("\n" + "=" * 60)
    print("EVALUATING CLASSICAL BASELINES UNDER STRICT LOSO")
    print("=" * 60)
    
    for name, clf in baselines.items():
        fold_accs, fold_f1s = [], []
        all_targets, all_preds = [], []
        for test_subj in unique_subjects:
            tr_mask = (subject_arr != test_subj)
            te_mask = (subject_arr == test_subj)
            
            X_tr = X_np[tr_mask]
            y_tr = y_np[tr_mask]
            X_te = X_np[te_mask]
            y_te = y_np[te_mask]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_te_sc)
            
            fold_accs.append(accuracy_score(y_te, preds) * 100)
            fold_f1s.append(f1_score(y_te, preds, average="macro", zero_division=0) * 100)
            all_targets.extend(y_te.tolist())
            all_preds.extend(preds.tolist())
            
        m_acc = np.mean(fold_accs)
        s_acc = np.std(fold_accs)
        m_f1 = np.mean(fold_f1s)
        kappa = cohen_kappa_score(all_targets, all_preds)
        print(f"  {name:30s} -> Acc: {m_acc:.2f}% ± {s_acc:.2f}%, Macro F1: {m_f1:.2f}%, Kappa: {kappa:+.4f}")
        baseline_results[name] = {
            "mean_accuracy": round(float(m_acc), 2),
            "std_accuracy": round(float(s_acc), 2),
            "mean_macro_f1": round(float(m_f1), 2),
            "kappa": round(float(kappa), 4)
        }
    return baseline_results

