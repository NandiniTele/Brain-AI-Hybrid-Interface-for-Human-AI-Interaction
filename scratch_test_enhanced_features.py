import sys
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.signal as signal
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('backend'))
from dataset_loader import BCIDatasetLoader
from model import HybridBCINet

_STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]
N_CH = 19

def extract_enhanced_features_from_window(window: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts subject-invariant EEG features:
    - Relative band powers (delta, theta, alpha, beta, gamma)
    - Log band powers
    - Spectral power ratios (Theta/Beta, Alpha/Beta, Theta/Alpha)
    - Normalized Hjorth parameters (log activity, mobility, complexity)
    - Shannon entropy
    - Log variance, normalized mean, std
    - Wavelet sub-band relative energies (db4 level 4)
    - Channel-level spatial contrasts
    """
    ch_features = []
    
    # Store band powers per channel for spatial asymmetry computation
    ch_alpha = []
    ch_beta = []
    ch_theta = []
    
    for ch_data in window:
        # 1. Band Powers via Welch
        nperseg = min(256, len(ch_data))
        freqs, psd = signal.welch(ch_data, sfreq, nperseg=nperseg)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        abs_powers = []
        for fmin, fmax in bands.values():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            abs_powers.append(np.sum(psd[idx]))
        abs_powers = np.array(abs_powers)
        total_power = np.sum(abs_powers) + 1e-8
        rel_powers = abs_powers / total_power
        log_powers = np.log1p(abs_powers)
        
        # Power ratios
        theta_beta = abs_powers[1] / (abs_powers[3] + 1e-8)
        alpha_beta = abs_powers[2] / (abs_powers[3] + 1e-8)
        theta_alpha = abs_powers[1] / (abs_powers[2] + 1e-8)
        
        ch_alpha.append(abs_powers[2])
        ch_beta.append(abs_powers[3])
        ch_theta.append(abs_powers[1])
        
        # 2. Hjorth Parameters
        diff1 = np.diff(ch_data)
        diff2 = np.diff(diff1)
        var_y = np.var(ch_data)
        var_d1 = np.var(diff1)
        var_d2 = np.var(diff2)
        
        log_activity = np.log1p(var_y)
        mobility = np.sqrt(var_d1 / var_y) if var_y > 0 else 0
        complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0
        
        # 3. Shannon Entropy
        p, _ = np.histogram(ch_data, bins=20, density=True)
        p = p[p > 0]
        shannon = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
        
        # 4. Statistical Features
        stat_mean = np.mean(ch_data)
        stat_std = np.std(ch_data)
        
        # 5. Wavelet Features
        coeffs = pywt.wavedec(ch_data, 'db4', level=4)
        wave_feats = []
        tot_wave_energy = sum(np.sum(np.square(c)) for c in coeffs) + 1e-8
        for c in coeffs:
            c_energy = np.sum(np.square(c))
            wave_feats.extend([np.mean(c), np.std(c), c_energy / tot_wave_energy, np.log1p(c_energy)])
            
        feat_vector = (
            list(rel_powers) +           # 5
            list(log_powers) +           # 5
            [theta_beta, alpha_beta, theta_alpha] + # 3
            [log_activity, mobility, complexity] +  # 3
            [shannon, stat_mean, stat_std] +        # 3
            wave_feats                              # 20 (4 * 5)
        )
        # Total per channel: 5 + 5 + 3 + 3 + 3 + 20 = 39 features per channel
        ch_features.append(feat_vector)
        
    ch_features = np.array(ch_features) # (19, 39)
    
    # Let's add channel asymmetry pairs (F3-F4, C3-C4, P3-P4, O1-O2, T3-T4)
    # Channel indices in _STANDARD_CHANNELS:
    # F3: 3, F4: 5
    # C3: 8, C4: 10
    # P3: 13, P4: 15
    # O1: 17, O2: 18
    # T3: 7, T4: 11
    asym_pairs = [(3, 5), (8, 10), (13, 15), (17, 18), (7, 11)]
    asym_feats = []
    for l_idx, r_idx in asym_pairs:
        # Alpha asymmetry
        a_l, a_r = ch_alpha[l_idx], ch_alpha[r_idx]
        asym_feats.append((a_l - a_r) / (a_l + a_r + 1e-8))
        # Beta asymmetry
        b_l, b_r = ch_beta[l_idx], ch_beta[r_idx]
        asym_feats.append((b_l - b_r) / (b_l + b_r + 1e-8))
        
    return ch_features.flatten() # 19 * 39 = 741 features

# Let's extract dataset with enhanced features
loader = BCIDatasetLoader()
physionet_mapping = {1: 0, 4: 1, 6: 2, 2: 3, 3: 4}
X_list, y_list, subj_list = [], [], []

for subject in [1, 2, 3]:
    for run, class_id in physionet_mapping.items():
        data, sfreq = loader.load_physionet(subject=subject, run=run)
        chunk_size = int(sfreq * 2)
        data = data[:N_CH]
        n_chunks = data.shape[1] // chunk_size
        for i in range(n_chunks):
            w = data[:, i * chunk_size : (i + 1) * chunk_size]
            feats = extract_enhanced_features_from_window(w, sfreq=sfreq)
            X_list.append(feats)
            y_list.append(class_id)
            subj_list.append(subject)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
subjects = np.array(subj_list, dtype=np.int32)

print(f"Extracted enhanced feature dataset: X={X.shape}, y={y.shape}")

# Evaluate RBF SVM with enhanced features on strict LOSO
print("\n" + "=" * 50)
print("EVALUATING RBF SVM WITH ENHANCED PHYSIOLOGICAL FEATURES")
print("=" * 50)
svm_accs = []
svm_f1s = []
for test_subj in [1, 2, 3]:
    te_mask = (subjects == test_subj)
    tr_mask = (subjects != test_subj)
    
    X_tr_raw = X[tr_mask]
    y_tr = y[tr_mask]
    X_te_raw = X[te_mask]
    y_te = y[te_mask]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    
    clf = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    
    acc = accuracy_score(y_te, preds) * 100
    f1 = f1_score(y_te, preds, average="macro", zero_division=0) * 100
    svm_accs.append(acc)
    svm_f1s.append(f1)
    print(f"  Subj {test_subj}: Acc = {acc:.2f}%, F1 = {f1:.2f}%, Pred Dist = {np.bincount(preds, minlength=5).tolist()}")

print(f"==> RBF SVM MEAN LOSO: Acc = {np.mean(svm_accs):.2f}% ± {np.std(svm_accs):.2f}%, Macro F1 = {np.mean(svm_f1s):.2f}%")

with open("scratch_enhanced_results.txt", "w") as f:
    f.write(f"RBF SVM Enhanced: Acc={np.mean(svm_accs):.2f}%, F1={np.mean(svm_f1s):.2f}%\n")
