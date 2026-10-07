import sys
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from train import extract_real_eeg_features, balance_training_data, TASK_CLASSES, INPUT_FEATURES, N_CHANNELS, FEATURES_PER_CH
from model import HybridBCINet

# 1. Load data
X, y_reg, y_task, subject_arr, run_arr = extract_real_eeg_features(window_size_sec=2)

print(f"Data shape: X={X.shape}, y_task={y_task.shape}")
print(f"Subject distribution: {np.bincount(subject_arr)}")
print(f"Task class distribution: {np.bincount(y_task)}")

# Let's inspect feature stats per subject
for s in [1, 2, 3]:
    s_mask = (subject_arr == s)
    Xs = X[s_mask].numpy()
    print(f"\nSubject {s} (N={len(Xs)}):")
    print(f"  Mean of features: {np.mean(Xs):.4f}, Std: {np.std(Xs):.4f}, Min: {np.min(Xs):.4f}, Max: {np.max(Xs):.4f}")
    # Check first 5 features (Delta, Theta, Alpha, Beta, Gamma) on Ch 0
    print(f"  Ch0 Band powers mean: {np.mean(Xs[:, :5], axis=0)}")
    print(f"  Ch0 Hjorth mean: {np.mean(Xs[:, 5:8], axis=0)}")

# Let's test what happens if we use log-transform: log1p on non-negative energy/variance features
# Or relative band powers
