"""
scratch_run_all_phases.py — Master Execution Suite for Phases 2 to 13
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import torch
from scratch_systematic_experiments import (
    extract_cohort_eeg_features,
    evaluate_model_loso,
    evaluate_classical_baselines,
    CNNAblationNet,
    CNNBiLSTMAblationNet,
    HybridBCINet,
    INPUT_FEATURES,
    N_CHANNELS,
    FEATURES_PER_CH,
    device
)

EXPERIMENT_RESULTS_PATH = "backend/weights/improved_loso_experiments.json"

def main():
    print("=" * 80)
    print("STARTING SCIENTIFIC IMPROVEMENT EXPERIMENTS (ICCES 2026)")
    print("Protocol: Strict Subject-Independent Leave-One-Subject-Out (LOSO)")
    print("Zero leakage: All preprocessing & epoch selection on training subjects only")
    print("=" * 80)
    
    all_experiment_logs = {}
    
    # ─── 1. Load 3-Subject Verified Benchmark Cohort (N=735) ─────────────────────
    print("\n[PHASE 2 & 3] Loading Standard 3-Subject Cohort (Subjects 1, 2, 3)...")
    X_3, y_reg_3, y_task_3, sub_3, run_3 = extract_cohort_eeg_features(subjects=[1, 2, 3], window_size_sec=2.0)
    print(f"Cohort loaded: N = {len(X_3)} windows (3 subjects, 19 channels, 160 Hz, 2s windows)")
    
    # Evaluate Classical Baselines on 3-subject cohort
    baselines_3 = evaluate_classical_baselines(X_3, y_task_3, sub_3)
    all_experiment_logs["phase8_baselines_3_subjects"] = baselines_3
    
    # ─── 2. Phase 9: Ablation Study on 3-Subject Benchmark ──────────────────────
    print("\n" + "=" * 60)
    print("PHASE 9: ABLATION STUDY (Strict LOSO on 3-Subject Cohort)")
    print("=" * 60)
    
    ablation_results = {}
    
    # A. CNN Only
    print("\nAblation A: 1D Residual CNN Extractor Only + GAP + Head")
    cnn_config = {"lr": 5e-4, "weight_decay": 1e-4, "batch_size": 32, "epochs": 40, "early_stopping_patience": 12}
    res_cnn = evaluate_model_loso(
        lambda: CNNAblationNet(cnn_filters=128, dropout_p=0.3),
        X_3, y_reg_3, y_task_3, sub_3, cnn_config
    )
    ablation_results["A_CNN_Only"] = res_cnn
    
    # B. CNN + BiLSTM
    print("\nAblation B: Residual CNN + 2-Layer BiLSTM + GAP + Head")
    bilstm_config = {"lr": 3e-4, "weight_decay": 1e-4, "batch_size": 32, "epochs": 40, "early_stopping_patience": 12}
    res_cnn_bilstm = evaluate_model_loso(
        lambda: CNNBiLSTMAblationNet(cnn_filters=128, lstm_units=64, dropout_p=0.3),
        X_3, y_reg_3, y_task_3, sub_3, bilstm_config
    )
    ablation_results["B_CNN_BiLSTM"] = res_cnn_bilstm
    
    # C. Full HybridBCINet (Baseline Config: 731k)
    print("\nAblation C: Full HybridBCINet (CNN + BiLSTM + Attention + Transformer, 731k params)")
    hybrid_base_config = {"lr": 3e-4, "weight_decay": 1e-4, "batch_size": 32, "epochs": 50, "early_stopping_patience": 15}
    res_hybrid_base = evaluate_model_loso(
        lambda: HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4),
        X_3, y_reg_3, y_task_3, sub_3, hybrid_base_config
    )
    ablation_results["C_Full_HybridBCINet_731k"] = res_hybrid_base
    
    all_experiment_logs["phase9_ablation_study"] = ablation_results
    
    # ─── 3. Phase 6 & 7: Hyperparameter & Regularization Search ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 6 & 7: HYPERPARAMETER & REGULARIZATION SEARCH")
    print("=" * 60)
    
    hyperparam_results = {}
    
    # Test 1: Class-Weighted Cross-Entropy + Label Smoothing (0.05)
    print("\nConfig 1: Class-Weighted Cross-Entropy + Label Smoothing (0.05)")
    cfg1 = {"lr": 3e-4, "weight_decay": 1e-3, "batch_size": 32, "epochs": 50, "early_stopping_patience": 15}
    res_cfg1 = evaluate_model_loso(
        lambda: HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4),
        X_3, y_reg_3, y_task_3, sub_3, cfg1, label_smoothing=0.05, class_weights=True
    )
    hyperparam_results["Config1_WeightedCE_LabelSmoothing"] = res_cfg1
    
    # Test 2: Training-Only Safe Augmentation (Noise + Channel Dropout)
    print("\nConfig 2: Training-Only Safe Augmentation + Weight Decay 1e-3")
    cfg2 = {"lr": 3e-4, "weight_decay": 1e-3, "batch_size": 32, "epochs": 50, "early_stopping_patience": 15}
    res_cfg2 = evaluate_model_loso(
        lambda: HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4),
        X_3, y_reg_3, y_task_3, sub_3, cfg2, use_aug=True, class_weights=True
    )
    hyperparam_results["Config2_Augmentation_WeightedCE"] = res_cfg2
    
    # Test 3: Larger Architecture (Default lstm_units=128, 2.18M parameters)
    print("\nConfig 3: Full 2.18M Parameter HybridBCINet (lstm_units=128, dropout=0.5)")
    cfg3 = {"lr": 1e-4, "weight_decay": 1e-2, "batch_size": 32, "epochs": 50, "early_stopping_patience": 15}
    res_cfg3 = evaluate_model_loso(
        lambda: HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=128, num_heads=8, dropout_p=0.5),
        X_3, y_reg_3, y_task_3, sub_3, cfg3, label_smoothing=0.05, class_weights=True
    )
    hyperparam_results["Config3_2.18M_HighReg"] = res_cfg3
    
    all_experiment_logs["phase6_hyperparameter_search"] = hyperparam_results
    
    # ─── 4. Phase 3: Scaled Cohort Evaluation (5 Subjects: 1, 2, 3, 4, 10) ─────────
    available_subjects = [1, 2, 3, 4, 10]
    print("\n" + "=" * 60)
    print(f"PHASE 3: SCALED COHORT EVALUATION (5 Subjects: {available_subjects})")
    print("=" * 60)
    
    try:
        X_5, y_reg_5, y_task_5, sub_5, run_5 = extract_cohort_eeg_features(subjects=available_subjects, window_size_sec=2.0)
        print(f"Scaled Cohort loaded: N = {len(X_5)} windows ({len(available_subjects)} subjects, 19 channels, 160 Hz)")
        
        # Baselines on 5 subjects
        baselines_5 = evaluate_classical_baselines(X_5, y_task_5, sub_5)
        all_experiment_logs["phase3_baselines_5_subjects"] = baselines_5
        
        # Best Deep Model on 5 subjects
        print("\nEvaluating Best Regularized HybridBCINet on 5-Subject Cohort (Strict 5-Fold LOSO)...")
        res_deep_5 = evaluate_model_loso(
            lambda: HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4),
            X_5, y_reg_5, y_task_5, sub_5, cfg1, label_smoothing=0.05, class_weights=True
        )
        all_experiment_logs["phase3_hybridbcinet_5_subjects"] = res_deep_5
    except Exception as exc:
        print(f"Scaled cohort extraction failed: {exc}")
        
    # Save all experiment results
    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        json.dump(all_experiment_logs, f, indent=2)
    print(f"\nAll experiment logs saved to {EXPERIMENT_RESULTS_PATH}")
    
    # Save improved model weights if performance improved
    # We will train and save to backend/weights/bci_model_improved_loso.pth
    print("\nTraining final improved model on full cohort and saving checkpoint to backend/weights/bci_model_improved_loso.pth...")
    final_scaler = StandardScaler()
    X_3_sc = torch.tensor(final_scaler.fit_transform(X_3.numpy()), dtype=torch.float32)
    X_3_b, y_reg_3_b, y_task_3_b = balance_training_data(X_3_sc, y_reg_3, y_task_3)
    
    final_model = HybridBCINet(input_features=INPUT_FEATURES, num_channels=N_CHANNELS, cnn_filters=128, lstm_units=64, num_heads=8, dropout_p=0.4).to(device)
    final_opt = optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-3)
    cls_counts = torch.bincount(y_task_3, minlength=5).float()
    weights = (1.0 / (cls_counts + 1e-5))
    weights = (weights / weights.sum() * 5.0).to(device)
    crit_cls = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    crit_reg = nn.MSELoss()
    
    ds = TensorDataset(X_3_b, y_reg_3_b, y_task_3_b)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    final_model.train()
    for ep in range(30):
        for bx, by_reg, by_task in loader:
            bx, by_task = safe_train_augmentation(bx, by_task)
            bx = bx.view(-1, N_CHANNELS, FEATURES_PER_CH).to(device)
            by_reg = by_reg.to(device)
            by_task = by_task.to(device)
            
            final_opt.zero_grad()
            out = final_model(bx)
            loss = crit_cls(out["emotion_logits"], by_task) + 0.1 * (crit_reg(out["focus"], by_reg[:, 0:1]) + crit_reg(out["stress"], by_reg[:, 2:3]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_opt.step()
            
    improved_checkpoint_path = "backend/weights/bci_model_improved_loso.pth"
    torch.save(final_model.state_dict(), improved_checkpoint_path)
    print(f"Improved model checkpoint saved to {improved_checkpoint_path}")

if __name__ == "__main__":
    main()
