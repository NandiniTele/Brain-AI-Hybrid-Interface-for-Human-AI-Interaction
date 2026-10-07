import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from train import extract_real_eeg_features, balance_training_data, TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH
from model import HybridBCINet

X, y_reg, y_task, subject_arr, run_arr = extract_real_eeg_features(window_size_sec=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_loso(model_factory, use_log1p=False, lr=0.0003, weight_decay=1e-2, epochs=35, batch_size=32, loss_mode="cls_only", label_smoothing=0.0):
    fold_accs = []
    fold_f1s = []
    
    for test_subj in [1, 2, 3]:
        te_mask = (subject_arr == test_subj)
        tr_pool_mask = (subject_arr != test_subj)
        
        X_test_raw = X[te_mask].numpy().copy()
        y_test = y_task[te_mask].numpy().copy()
        
        X_pool_raw = X[tr_pool_mask].numpy().copy()
        y_reg_pool = y_reg[tr_pool_mask]
        y_pool = y_task[tr_pool_mask]
        
        if use_log1p:
            # Apply log1p to non-negative features (energy, variance, band powers)
            X_test_raw = np.sign(X_test_raw) * np.log1p(np.abs(X_test_raw))
            X_pool_raw = np.sign(X_pool_raw) * np.log1p(np.abs(X_pool_raw))
            
        pool_indices = np.arange(len(X_pool_raw))
        in_train_idx, in_val_idx = train_test_split(
            pool_indices, test_size=0.2, stratify=y_pool.numpy(), random_state=42
        )
        
        X_in_train_raw = X_pool_raw[in_train_idx]
        y_reg_in_train_raw = y_reg_pool[in_train_idx]
        y_in_train_raw = y_pool[in_train_idx]
        
        X_in_val_raw = X_pool_raw[in_val_idx]
        y_reg_in_val = y_reg_pool[in_val_idx]
        y_in_val = y_pool[in_val_idx].numpy()
        
        scaler = StandardScaler()
        X_in_train_scaled = torch.tensor(scaler.fit_transform(X_in_train_raw), dtype=torch.float32)
        X_in_val_scaled = torch.tensor(scaler.transform(X_in_val_raw), dtype=torch.float32).to(device)
        X_test_scaled = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(device)
        
        X_in_train, y_reg_in_train, y_in_train = balance_training_data(
            X_in_train_scaled, y_reg_in_train_raw, y_in_train_raw
        )
        
        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_reg = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(X_in_train, y_reg_in_train, y_in_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_in_val_scaled, y_reg_in_val, torch.tensor(y_in_val, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False
        )
        
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            model.train()
            for bx, by_reg, by in train_loader:
                bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                by = by.to(device)
                by_reg = by_reg.to(device)
                
                optimizer.zero_grad()
                out = model(bx)
                loss_cls = criterion_cls(out["emotion_logits"], by)
                if loss_mode == "with_reg":
                    loss_reg = criterion_reg(out["focus"], by_reg[:, 0:1]) + criterion_reg(out["stress"], by_reg[:, 2:3])
                    loss = loss_cls + 0.10 * loss_reg
                else:
                    loss = loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            
            # Internal validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by_reg, by in val_loader:
                    bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                    by = by.to(device)
                    out = model(bx)
                    v_loss = criterion_cls(out["emotion_logits"], by)
                    val_loss += v_loss.item()
            val_loss /= max(len(val_loader), 1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        # Evaluate frozen model ONCE on test subject
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        with torch.no_grad():
            out_test = model(X_test_scaled.view(-1, N_CHANNELS, FEATURES_PER_CH))
            test_preds = torch.argmax(out_test["emotion_logits"], dim=1).cpu().numpy()
            
        test_acc = accuracy_score(y_test, test_preds) * 100
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0) * 100
        fold_accs.append(test_acc)
        fold_f1s.append(test_f1)
        pred_dist = np.bincount(test_preds, minlength=5).tolist()
        print(f"  Subj {test_subj} (Best Ep {best_epoch}): Acc = {test_acc:.2f}%, F1 = {test_f1:.2f}%, Dist = {pred_dist}")
        
    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    print(f"==> MEAN LOSO: Acc = {mean_acc:.2f}% ± {np.std(fold_accs):.2f}%, F1 = {mean_f1:.2f}%\n")
    return mean_acc, mean_f1

print("--- Testing Baseline Config (128 filters, 64 lstm, 0.4 dropout) ---")
def make_baseline():
    return HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4)
evaluate_loso(make_baseline, use_log1p=False, loss_mode="with_reg")

print("--- Testing Regularized Compact Config (64 filters, 32 lstm, 4 heads, 0.5 dropout, weight_decay=0.05) ---")
def make_compact():
    return HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=64, lstm_units=32, num_heads=4, dropout_p=0.5)
evaluate_loso(make_compact, use_log1p=False, lr=0.0005, weight_decay=0.05, loss_mode="cls_only")

print("--- Testing Compact + log1p transform ---")
evaluate_loso(make_compact, use_log1p=True, lr=0.0005, weight_decay=0.05, loss_mode="cls_only")

print("--- Testing Compact + log1p + Label Smoothing 0.1 ---")
evaluate_loso(make_compact, use_log1p=True, lr=0.0005, weight_decay=0.05, label_smoothing=0.1, loss_mode="cls_only")
