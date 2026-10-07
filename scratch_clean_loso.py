import sys
import os
import json
import torch
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from train import extract_real_eeg_features, balance_training_data, TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH, WEIGHTS_DIR
from model import HybridBCINet

def compute_multiclass_roc_auc(y_true, y_probs):
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        return float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    except Exception as e:
        return 0.5

print("=" * 70)
print("CLEAN LEAVE-ONE-SUBJECT-OUT (LOSO) EVALUATION OF HYBRIDBCINET")
print("Zero test-subject leakage: Internal validation used for ALL model/epoch selection")
print("=" * 70)

# 1. Extract genuine PhysioNet EEGBCI data
X, y_reg, y_task, subject_arr, run_arr = extract_real_eeg_features(window_size_sec=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loso_results = {}
all_test_preds = []
all_test_targets = []
all_test_probs = []

for test_subj in [1, 2, 3]:
    print(f"\n" + "=" * 50)
    print(f"FOLD {test_subj}/3: Test Subject = {test_subj}")
    print("=" * 50)
    
    # Step 1: Isolate the held-out test subject completely
    te_mask = (subject_arr == test_subj)
    tr_pool_mask = (subject_arr != test_subj)
    
    X_test_raw = X[te_mask].numpy()
    y_test = y_task[te_mask].numpy()
    
    X_pool_raw = X[tr_pool_mask].numpy()
    y_reg_pool = y_reg[tr_pool_mask]
    y_pool = y_task[tr_pool_mask]
    
    print(f"Held-out Test Subject {test_subj}: N = {len(X_test_raw)} windows (Class counts: {dict(enumerate(np.bincount(y_test, minlength=5)))})")
    print(f"Training Pool (Subjects {[s for s in [1,2,3] if s != test_subj]}): N = {len(X_pool_raw)} windows")
    
    # Step 2: Internal train/validation split (80% internal train, 20% internal validation)
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
    
    print(f"  Internal Training Split: N = {len(X_in_train_raw)}")
    print(f"  Internal Validation Split: N = {len(X_in_val_raw)}")
    
    # Step 3 & 4: Fit StandardScaler STRICTLY on internal training data only
    scaler = StandardScaler()
    X_in_train_scaled = torch.tensor(scaler.fit_transform(X_in_train_raw), dtype=torch.float32)
    X_in_val_scaled = torch.tensor(scaler.transform(X_in_val_raw), dtype=torch.float32).to(device)
    X_test_scaled = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(device)
    
    # Step 5: Balance ONLY the internal training portion
    X_in_train, y_reg_in_train, y_in_train = balance_training_data(
        X_in_train_scaled, y_reg_in_train_raw, y_in_train_raw
    )
    print(f"  Internal Training Split Balanced: N = {len(X_in_train)}")
    
    # Step 6: Initialize HybridBCINet
    model = HybridBCINet(
        input_features=INPUT_FEATURES,
        num_channels=N_CHANNELS,
        cnn_filters=128,
        lstm_units=64,
        num_heads=8,
        dropout_p=0.4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(X_in_train, y_reg_in_train, y_in_train),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_in_val_scaled, y_reg_in_val, torch.tensor(y_in_val, dtype=torch.long)),
        batch_size=32,
        shuffle=False
    )
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    # Step 7: Train and select best epoch STRICTLY on Internal Validation
    for epoch in range(1, 36):
        model.train()
        train_loss = 0.0
        for bx, by_reg, by in train_loader:
            bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
            by = by.to(device)
            by_reg = by_reg.to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss_cls = criterion_cls(out["emotion_logits"], by)
            loss_reg = criterion_reg(out["focus"], by_reg[:, 0:1]) + criterion_reg(out["stress"], by_reg[:, 2:3])
            loss = loss_cls + 0.10 * loss_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Evaluate on INTERNAL VALIDATION ONLY
        model.eval()
        val_loss = 0.0
        val_preds, val_targets_list = [], []
        with torch.no_grad():
            for bx, by_reg, by in val_loader:
                bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
                by = by.to(device)
                out = model(bx)
                v_loss = criterion_cls(out["emotion_logits"], by)
                val_loss += v_loss.item()
                val_preds.extend(torch.argmax(out["emotion_logits"], dim=1).cpu().numpy())
                val_targets_list.extend(by.cpu().numpy())
                
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        val_acc = accuracy_score(val_targets_list, val_preds) * 100
        
        # Save best model based ONLY on internal validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"  Best Epoch Selected by Internal Validation: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
    
    # Step 8: Load best internal model state and evaluate ONCE on held-out test subject
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    
    with torch.no_grad():
        out_test = model(X_test_scaled.view(-1, N_CHANNELS, FEATURES_PER_CH))
        test_logits = out_test["emotion_logits"]
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        
    # Step 9: Compute all metrics for this held-out subject
    test_acc = accuracy_score(y_test, test_preds) * 100
    test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0) * 100
    test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0) * 100
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0) * 100
    test_roc_auc = compute_multiclass_roc_auc(y_test, test_probs)
    test_kappa = float(cohen_kappa_score(y_test, test_preds))
    test_mcc = float(matthews_corrcoef(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2, 3, 4])
    pred_dist = np.bincount(test_preds, minlength=5).tolist()
    
    # Per-class metrics
    pc_prec = precision_score(y_test, test_preds, average=None, zero_division=0) * 100
    pc_rec = recall_score(y_test, test_preds, average=None, zero_division=0) * 100
    pc_f1 = f1_score(y_test, test_preds, average=None, zero_division=0) * 100
    
    loso_results[f"Subject_{test_subj}"] = {
        "selected_epoch": best_epoch,
        "internal_val_acc": round(best_val_acc, 2),
        "internal_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(test_acc, 2),
        "test_macro_precision": round(test_prec, 2),
        "test_macro_recall": round(test_rec, 2),
        "test_macro_f1": round(test_f1, 2),
        "test_roc_auc": round(test_roc_auc, 4),
        "test_cohens_kappa": round(test_kappa, 4),
        "test_mcc": round(test_mcc, 4),
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": pred_dist,
        "per_class_precision": [round(float(x), 2) for x in pc_prec],
        "per_class_recall": [round(float(x), 2) for x in pc_rec],
        "per_class_f1": [round(float(x), 2) for x in pc_f1],
    }
    
    all_test_preds.extend(test_preds)
    all_test_targets.extend(y_test)
    all_test_probs.extend(test_probs)
    
    print(f"  Held-out Test Subject {test_subj} Evaluation Results:")
    print(f"    Accuracy: {test_acc:.2f}% (Chance: 20.00%)")
    print(f"    Macro Precision: {test_prec:.2f}% | Macro Recall: {test_rec:.2f}% | Macro F1: {test_f1:.2f}%")
    print(f"    ROC-AUC: {test_roc_auc:.4f} | Cohen's Kappa: {test_kappa:.4f} | MCC: {test_mcc:.4f}")
    print(f"    Prediction Distribution: {pred_dist}")
    print(f"    Confusion Matrix:\n{cm}")

# 3. Overall Aggregate Summary across all 3 subjects
acc_list = [loso_results[f"Subject_{s}"]["test_accuracy"] for s in [1, 2, 3]]
prec_list = [loso_results[f"Subject_{s}"]["test_macro_precision"] for s in [1, 2, 3]]
rec_list = [loso_results[f"Subject_{s}"]["test_macro_recall"] for s in [1, 2, 3]]
f1_list = [loso_results[f"Subject_{s}"]["test_macro_f1"] for s in [1, 2, 3]]
roc_list = [loso_results[f"Subject_{s}"]["test_roc_auc"] for s in [1, 2, 3]]
kappa_list = [loso_results[f"Subject_{s}"]["test_cohens_kappa"] for s in [1, 2, 3]]
mcc_list = [loso_results[f"Subject_{s}"]["test_mcc"] for s in [1, 2, 3]]

overall_cm = confusion_matrix(all_test_targets, all_test_preds, labels=[0, 1, 2, 3, 4])
overall_pred_dist = np.bincount(all_test_preds, minlength=5).tolist()

aggregate_summary = {
    "mean_accuracy": round(float(np.mean(acc_list)), 2),
    "std_accuracy": round(float(np.std(acc_list)), 2),
    "mean_macro_precision": round(float(np.mean(prec_list)), 2),
    "mean_macro_recall": round(float(np.mean(rec_list)), 2),
    "mean_macro_f1": round(float(np.mean(f1_list)), 2),
    "mean_roc_auc": round(float(np.mean(roc_list)), 4),
    "mean_cohens_kappa": round(float(np.mean(kappa_list)), 4),
    "mean_mcc": round(float(np.mean(mcc_list)), 4),
    "overall_confusion_matrix": overall_cm.tolist(),
    "overall_prediction_distribution": overall_pred_dist,
    "chance_level": 20.00,
}

final_output = {
    "per_subject_results": loso_results,
    "aggregate_summary": aggregate_summary,
}

clean_results_path = os.path.join(WEIGHTS_DIR, "loso_clean_results.json")
with open(clean_results_path, "w") as f:
    json.dump(final_output, f, indent=2)

print("\n" + "=" * 70)
print("FINAL AGGREGATE CLEAN LOSO RESULTS (HYBRIDBCINET)")
print("=" * 70)
print(f"Mean LOSO Accuracy: {aggregate_summary['mean_accuracy']:.2f}% ± {aggregate_summary['std_accuracy']:.2f}% (Chance: 20.00%)")
print(f"Mean Macro Precision: {aggregate_summary['mean_macro_precision']:.2f}%")
print(f"Mean Macro Recall: {aggregate_summary['mean_macro_recall']:.2f}%")
print(f"Mean Macro F1-Score: {aggregate_summary['mean_macro_f1']:.2f}%")
print(f"Mean ROC-AUC: {aggregate_summary['mean_roc_auc']:.4f}")
print(f"Mean Cohen's Kappa: {aggregate_summary['mean_cohens_kappa']:.4f}")
print(f"Mean MCC: {aggregate_summary['mean_mcc']:.4f}")
print(f"Overall Prediction Distribution (N=735): {overall_pred_dist}")
print(f"Overall Confusion Matrix:\n{overall_cm}")
print(f"\nResults saved to: {clean_results_path}")
