import sys
import os
import shutil
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, os.path.abspath('backend'))
from train import extract_real_eeg_features, balance_training_data, TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH, WEIGHTS_PATH
from model import HybridBCINet

# 1. Preserve existing checkpoint
backup_path = 'backend/weights/bci_model_before_improvement.pth'
if os.path.exists(WEIGHTS_PATH) and not os.path.exists(backup_path):
    shutil.copyfile(WEIGHTS_PATH, backup_path)
    print(f'Preserved existing checkpoint to {backup_path}')
elif os.path.exists(backup_path):
    print(f'Backup already exists at {backup_path}')

# 2. Extract features
X, y_reg, y_task, subject_arr, run_arr = extract_real_eeg_features(window_size_sec=2)
X_np = X.numpy()
y_np = y_task.numpy()

print('\n=== PHASE 1.1: LABEL ENCODING AUDIT ===')
for cid, info in TASK_CLASSES.items():
    print(f'Class {cid}: {info["name"]} (Alias: {info["ui_alias"]}) -> PhysioNet Run {info["run"]}')

print('\n=== PHASE 1.2: FEATURE STATS & NORMALIZATION AUDIT ===')
print(f'Feature array shape: {X_np.shape} (Channels: {N_CHANNELS}, Features/Ch: {FEATURES_PER_CH}, Total: {INPUT_FEATURES})')
print(f'Feature value range: min={X_np.min():.4f}, max={X_np.max():.4f}, mean={X_np.mean():.4f}, std={X_np.std():.4f}')
print(f'Per-feature mean range: min={X_np.mean(axis=0).min():.4f}, max={X_np.mean(axis=0).max():.4f}')
print(f'Per-feature std range: min={X_np.std(axis=0).min():.4f}, max={X_np.std(axis=0).max():.4f}')
print(f'Any NaN in features: {np.isnan(X_np).any()}, Any Inf: {np.isinf(X_np).any()}')

# Check feature scale variance across features (e.g., band powers vs wavelets vs stats)
print('Sample feature scales (Channel 0 Fp1 features 0-26):')
print(f'  PSD Band powers (0-4): {X_np[:, 0:5].mean(axis=0).round(4)}')
print(f'  Hjorth (5-7): {X_np[:, 5:8].mean(axis=0).round(4)}')
print(f'  Entropy (8): {X_np[:, 8].mean():.4f}')
print(f'  Stats (9-11): {X_np[:, 9:12].mean(axis=0).round(4)}')
print(f'  Wavelet mean (12,15,18,21,24): {X_np[:, [12,15,18,21,24]].mean(axis=0).round(4)}')

print('\n=== PHASE 1.3: CLASS DISTRIBUTION AUDIT ===')
total = len(y_np)
print(f'Total dataset windows: {total}')
for cid in range(5):
    cnt = np.sum(y_np == cid)
    print(f'  Class {cid} ({TASK_CLASSES[cid]["name"]}): {cnt} windows ({cnt/total*100:.2f}%)')

for s in [1, 2, 3]:
    s_mask = (subject_arr == s)
    s_cnts = [np.sum(y_np[s_mask] == c) for c in range(5)]
    print(f'Subject {s} (Total={np.sum(s_mask)}): {dict(enumerate(s_cnts))}')

# 3. Test multiple classical baseline classifiers using LOSO
print('\n=== PHASE 1.6: CLASSICAL BASELINE LOSO AUDIT ===')
models = {
    'Logistic Regression (C=1.0)': LogisticRegression(max_iter=1000, C=1.0),
    'Logistic Regression (C=0.1, Balanced)': LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced'),
    'Linear SVM (C=0.1, Balanced)': SVC(kernel='linear', C=0.1, class_weight='balanced'),
    'RBF SVM (C=1.0, Balanced)': SVC(kernel='rbf', C=1.0, class_weight='balanced'),
    'Random Forest (100 trees, Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Ridge Classifier (Balanced)': RidgeClassifier(class_weight='balanced')
}

for name, clf in models.items():
    loso_accs, loso_f1s = [], []
    all_preds_loso, all_targets_loso = [], []
    for test_subj in [1, 2, 3]:
        tr_mask = (subject_arr != test_subj)
        te_mask = (subject_arr == test_subj)
        
        X_tr = X_np[tr_mask]
        y_tr = y_np[tr_mask]
        X_te = X_np[te_mask]
        y_te = y_np[te_mask]
        
        # Standardize features using TRAINING STATISTICS ONLY
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        clf.fit(X_tr_sc, y_tr)
        preds = clf.predict(X_te_sc)
        
        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average='macro', zero_division=0) * 100
        loso_accs.append(acc)
        loso_f1s.append(f1)
        all_preds_loso.extend(preds)
        all_targets_loso.extend(y_te)
        
    mean_acc = np.mean(loso_accs)
    mean_f1 = np.mean(loso_f1s)
    pred_dist = np.bincount(all_preds_loso, minlength=5)
    print(f'{name}:')
    print(f'  Mean LOSO Accuracy: {mean_acc:.2f}% | Mean Macro F1: {mean_f1:.2f}%')
    print(f'  Folds: Subj1={loso_accs[0]:.2f}%, Subj2={loso_accs[1]:.2f}%, Subj3={loso_accs[2]:.2f}%')
    print(f'  Total Prediction Distribution (N=735): {pred_dist}')
    print(f'  Overall Confusion Matrix:\n{confusion_matrix(all_targets_loso, all_preds_loso)}\n')
