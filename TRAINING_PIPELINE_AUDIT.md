# COMPREHENSIVE TRAINING PIPELINE AUDIT

**Project:** BRAIN-AI-HYBRID-INTERFACE-SIMULATOR (ICCES 2026 Revision)  
**Date of Audit:** October 2026  
**Status:** Audit Complete — Zero Code Modified During Audit  

---

## 1. EXECUTIVE SUMMARY & AUDIT FINDINGS

This audit provides a component-by-component inspection of the active training, preprocessing, feature extraction, architectural, and evaluation pipelines in the repository prior to launching performance optimization experiments.

### Key Audit Findings:
1. **Model Parameterization:** The `HybridBCINet` architecture in `backend/model.py` supports scalable hidden dimensions. The baseline experiment used `lstm_units=64` ($731,340$ parameters), while the default configuration `lstm_units=128` yields $2,180,300$ parameters ($2.18\text{M}$).
2. **Current Dataset Scale:** Currently, `backend/train.py` loads 3 subjects (Subjects 1, 2, 3) across 5 runs (Runs 1, 4, 6, 2, 3), yielding 735 non-overlapping 2.0s windows ($320$ samples per window at $160\text{ Hz}$).
3. **Data Preprocessing & Normalization:** Signal preconditioning applies zero-phase FIR band-pass ($0.5\text{--}40.0\text{ Hz}$) and $50\text{ Hz}$ notch filtering. Per-channel z-score standardization is applied during dataset extraction, and feature-level `StandardScaler` is applied out-of-sample.
4. **Subject Independence & Leakage Safeguards:** In the clean LOSO baseline, test subjects are strictly isolated out-of-sample; `StandardScaler` and class balancing are strictly confined to the training subjects.
5. **Class Imbalance:** In the 735-window dataset, Class 0 ($90$ windows, $12.2\%$) and Class 3 ($90$ windows, $12.2\%$) are smaller than Classes 1, 2, and 4 ($185$ windows each, $25.2\%$). Class-weighted cross-entropy and fold-isolated balancing are critical for preventing class collapse.
6. **Feature Dimensionality:** 27 features are extracted per channel ($19 \times 27 = 513$ input features), spanning PSD band powers, Hjorth parameters, Shannon spectral entropy, time-domain moments, and 4-level Daubechies wavelet sub-band coefficients.

---

## 2. DETAILED PIPELINE COMPONENT AUDIT

### A. Dataset Loading & Channel Topography
- **File:** `backend/dataset_loader.py` & `backend/train.py`
- **Data Source:** PhysioNet EEGBCI (Motor Movement/Imagery Dataset).
- **Montage:** 19 standardized channels matching the International 10–20 System:
  `['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']`.
- **Sampling Rate:** $160.0\text{ Hz}$ (Native EDF rate from PhysioNet).
- **Run Mapping to 5 Task Conditions:**
  - `Run 1` $\to$ Class 0: Baseline Eyes-Open Rest (Proxy Alias: *Calm*, 60s, 30 windows/subj)
  - `Run 4` $\to$ Class 1: Motor Imagery — Left/Right Fist (Proxy Alias: *Focused*, 120s, ~61–62 windows/subj)
  - `Run 6` $\to$ Class 2: Motor Imagery — Bilateral Fists/Feet (Proxy Alias: *Stressed*, 120s, ~61–62 windows/subj)
  - `Run 2` $\to$ Class 3: Baseline Eyes-Closed Rest (Proxy Alias: *Fatigued*, 60s, 30 windows/subj)
  - `Run 3` $\to$ Class 4: Motor Execution — Left/Right Fist (Proxy Alias: *Excited*, 120s, ~61–62 windows/subj)
- **Duplicate Removal:** Hash-based verification (`hash(features.tobytes())`) guarantees zero duplicate feature vectors in the dataset.

### B. Signal Preprocessing & Temporal Windowing
- **Band-Pass Filtering:** Zero-phase FIR filter ($0.5\text{--}40.0\text{ Hz}$, `fir_design='firwin'`).
- **Notch Filtering:** $50.0\text{ Hz}$ powerline attenuation.
- **Window Length:** $2.0\text{ seconds}$ ($320\text{ samples}$ per window).
- **Window Slicing Scheme:** Non-overlapping consecutive temporal windows.

### C. Feature Extraction Pipeline
- **File:** `backend/feature_extraction.py`
- **Total Features:** $19\text{ channels} \times 27\text{ features/channel} = 513\text{ features}$.
- **Feature Breakdown per Channel:**
  1. *PSD Band Powers (5)*: Delta ($0.5\text{--}4\text{ Hz}$), Theta ($4\text{--}8\text{ Hz}$), Alpha ($8\text{--}13\text{ Hz}$), Beta ($13\text{--}30\text{ Hz}$), Gamma ($30\text{--}45\text{ Hz}$) via Welch's method ($N_{\text{perseg}} = 256$).
  2. *Hjorth Parameters (3)*: Activity ($\sigma_x^2$), Mobility ($\sigma_{x'} / \sigma_x$), Complexity ($\text{Mobility}(x') / \text{Mobility}(x)$).
  3. *Shannon Spectral Entropy (1)*: 20-bin probability density entropy.
  4. *Statistical Moments (3)*: Mean, Variance, Standard Deviation.
  5. *Wavelet Transform Features (15)*: 4-level decomposition with Daubechies-4 (`db4`), yielding 5 sub-bands ($A_4, D_4, D_3, D_2, D_1$) with Mean, Std, and Energy ($\sum c_i^2$) per sub-band.

### D. Neural Architecture (`HybridBCINet`)
- **File:** `backend/model.py`
- **Component 1 (Residual 1D CNN):**
  - 3 sequential blocks ($19 \to 64 \to 128 \to 128$).
  - Kernel size $k=3$, padding $p=1$, Batch Normalization, GELU non-linearities, 1D Dropout ($p=0.30\text{--}0.40$).
  - $1 \times 1$ Conv shortcut projections on channel change; identity shortcut on Block 3.
- **Component 2 (Bidirectional LSTM):**
  - 2 stacked BiLSTM layers with hidden dimension $h=64$ (output dim 128) or $h=128$ (output dim 256), batch first, dropout $0.30$.
- **Component 3 (Multi-Head Self-Attention):**
  - 8 attention heads over embedding dimension $256$, dropout $0.30$.
- **Component 4 (Transformer Encoder):**
  - 2 Transformer Encoder layers ($d_{\text{model}}=256, d_{\text{ff}}=512$, dropout $0.30$).
- **Component 5 (Pooling & Normalization):**
  - Temporal Global Average Pooling (`mean(dim=1)`), followed by Layer Normalization (`LayerNorm(256)`).
- **Component 6 (Prediction Heads):**
  - 5-class Task-Condition Classification Head (`Linear(256, 5)` + Softmax).
  - 7 Continuous Proxy Regression Heads (`Linear(256, 1)` + Sigmoid).

### E. Training & Optimization Settings
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $\lambda = 10^{-4}$).
- **Learning Rate:** Initial $\eta = 3 \times 10^{-4}\text{--}5 \times 10^{-4}$ with `ReduceLROnPlateau` scheduler (decay factor $\gamma = 0.5$, patience $= 5$ epochs).
- **Loss Function:** Multi-task composite loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.3 \times \mathcal{L}_{\text{MSE}}$.
- **Gradient Clipping:** Max norm threshold of 1.0.
- **Batch Size:** 32.

---

## 3. AUDIT CONCLUSION & EXPERIMENTAL ACTION PLAN

The pipeline structure is scientifically sound, modular, and free of hidden leakage in its clean LOSO formulation. The primary bottlenecks restricting cross-subject accuracy are:
1. **Cohort Sample Size:** 3 subjects ($735$ windows) provides insufficient cross-subject variance for deep sequence models.
2. **Subject Distribution Shift:** Direct cross-subject generalization requires broader cohort training, feature scaling, and regularization.

**Planned Systematic Experiments (Phases 2–14):**
- **Phase 2:** Strict LOSO evaluation harness with automated leakage assertions.
- **Phase 3:** Scaling PhysioNet cohort to 5, 8, and 10 subjects.
- **Phase 4–5:** Preprocessing and window length explorations (2s vs. 4s vs. 6s).
- **Phase 6–7:** Hyperparameter optimization (BiLSTM capacity, LR, weight decay, dropout, label smoothing, training-only augmentation).
- **Phase 8–9:** Comprehensive baseline benchmarking (RBF SVM, Random Forest, Logistic Regression, Gradient Boosting) and ablation study (CNN, CNN+BiLSTM, Full HybridBCINet).
- **Phase 10–14:** Final model selection, leakage verification, and reporting.
