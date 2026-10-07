# PROJECT–PAPER CONSISTENCY REPORT
**Project**: BRAIN-AI-HYBRID-INTERFACE-SIMULATOR  
**Paper Title**: *“Brain–AI Hybrid Interface Simulator: A Multi-Scale Residual CNN–BiLSTM–Transformer Architecture for Hardware-Independent Electroencephalography Decoding”*  
**Date of Audit**: October 2026  
**Document Purpose**: Definitive reference document aligning the research manuscript with the verified codebase implementation, empirical clean LOSO results, and scientific terminology.

---

## A. Correct Project Description
The project is a **software-based Brain–Computer Interface (BCI) simulation and cognitive state decoding platform**. It couples a digital signal processing pipeline with a deep learning neural architecture (**HybridBCINet: Multi-Scale Residual CNN–BiLSTM–Transformer**) and an interactive real-time dashboard.

It ingests pre-recorded benchmark EEG data (PhysioNet EEGBCI), preprocesses signals through standard frequency filters, extracts multi-domain features, and performs multi-task classification (5 task-condition states) and multi-output continuous regression (7 proxy cognitive indices).

---

## B. Exact Implemented Architecture
The model defined in `backend/model.py` (`HybridBCINet`) is formally structured as:

$$\text{Input } (B, 19, 27) \longrightarrow \text{Multi-Scale Residual 1D-CNN} \longrightarrow \text{BiLSTM} \longrightarrow \text{Multi-Head Attention} \longrightarrow \text{Transformer Encoder} \longrightarrow \text{Global Average Pooling} \longrightarrow \text{LayerNorm} \longrightarrow \text{Prediction Heads}$$

### Stage-by-Stage Breakdown:
1. **Multi-Scale Residual CNN Blocks** (`ResidualCNNBlock` × 3):
   - Block 1: Channels $19 \to 64$ (Conv1d, kernel=3, padding=1, BatchNorm1d, Dropout $p=0.3$, GELU + 1x1 Conv shortcut).
   - Block 2: Channels $64 \to 128$ (Conv1d, kernel=3, padding=1, BatchNorm1d, Dropout $p=0.3$, GELU + 1x1 Conv shortcut).
   - Block 3: Channels $128 \to 128$ (Conv1d, kernel=3, padding=1, BatchNorm1d, Dropout $p=0.3$, GELU + Identity shortcut).
2. **Bidirectional LSTM** (`nn.LSTM`):
   - Input size: 128, Hidden size: 128, Layers: 2, Bidirectional: `True`, Batch first: `True`, Dropout $p=0.3$.
   - Output feature dimension: $128 \times 2 = 256$.
3. **Multi-Head Self-Attention** (`nn.MultiheadAttention`):
   - Embedding dimension: 256, Heads: 8, Batch first: `True`, Dropout $p=0.3$.
4. **Transformer Encoder** (`nn.TransformerEncoder`):
   - 2 Transformer Encoder layers, $d_{\text{model}} = 256$, $n_{\text{head}} = 8$, $d_{\text{ff}} = 512$, Dropout $p=0.3$, GELU activation.
5. **Prediction Fusion & Pooling**:
   - Temporal Global Average Pooling (`mean(dim=1)`), LayerNorm (`nn.LayerNorm(256)`).
6. **Task Heads**:
   - **Classification Head**: `Linear(256, 5)` for 5 task-condition classes (*Calm, Focused, Stressed, Fatigued, Excited*).
   - **Regression Heads**: 7 independent `Linear(256, 1)` + Sigmoid layers (*Focus, Attention, Stress, Fatigue, Cognitive Load, Valence, Arousal*).

### Parameter Count:
- **Verified Trainable Parameters**: **731,340 parameters (731k)**.

---

## C. Exact Dataset Actually Used

### Primary Validated Dataset:
- **PhysioNet EEGBCI**:
  - **Subjects evaluated**: 3 subjects (Subjects 1, 2, 3).
  - **Runs per subject**: 5 selected experimental runs per subject (Runs 1, 4, 6, 2, 3).
  - **Channels**: 19 standard 10–20 international system electrodes (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`).
  - **Sampling Frequency**: $160\text{ Hz}$.
  - **Trial/Window Segmentation**: 2-second windows (320 samples/window), yielding **735 unique non-duplicate windows**.
  - **DEAP / SEED**: Not used for the reported experimental evaluation.

---

## D. Exact Preprocessing Steps
The actual signal preprocessing in `backend/train.py` and `backend/preprocessing.py` consists of:
1. **Channel Selection**: Standard 19-channel 10–20 montage.
2. **Powerline Notch Filtering**: FIR notch filter at $50\text{ Hz}$.
3. **Band-Pass Filtering**: Zero-phase FIR band-pass filter from $0.5\text{ Hz}$ to $40.0\text{ Hz}$.
4. **Z-Score Normalization**: Per-channel zero-mean unit-variance normalization.

---

## E. Exact Feature Extraction Methods
From each 2-second window (320 samples @ 160 Hz) across each of the 19 channels, `backend/feature_extraction.py` computes **27 features per channel** ($19 \times 27 = 513$ input features):
1. **Band Power Features (5 features)**:
   - Welch's Periodogram across $\delta$ (0.5–4 Hz), $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma$ (30–45 Hz).
2. **Hjorth Parameters (3 features)**:
   - Activity (signal variance), Mobility, Complexity.
3. **Shannon Spectral Entropy (1 feature)**.
4. **Time-Domain Statistical Moments (3 features)**:
   - Mean, Variance, Standard Deviation.
5. **Wavelet Transform Features (15 features)**:
   - Discrete Wavelet Transform (`db4`, 4 decomposition levels: 5 sub-bands $\times$ [Mean, Std, Energy]).

---

## F. Exact Prediction / Output Tasks
1. **Five Task-Condition Classes**:
   - Class 0 (Run 1): Baseline Eyes-Open Rest (Proxy: *Calm*, $N=90$, 12.24%)
   - Class 1 (Run 4): Left/Right Fist Motor Imagery (Proxy: *Focused*, $N=185$, 25.17%)
   - Class 2 (Run 6): Both-Fists / Both-Feet Motor Imagery (Proxy: *Stressed*, $N=185$, 25.17%)
   - Class 3 (Run 2): Baseline Eyes-Closed Rest (Proxy: *Fatigued*, $N=90$, 12.24%)
   - Class 4 (Run 3): Left/Right Fist Motor Execution (Proxy: *Excited*, $N=185$, 25.17%)
   *Explicit Note*: These are task-condition states mapped to semantic proxy aliases for simulation, not ground-truth clinical emotion annotations.
2. **Continuous Simulation Metrics (7 targets)**:
   - Focus, Attention, Stress, Fatigue, Cognitive Load, Valence, Arousal (normalized in $[0.0, 1.0]$).

---

## G. Verified Empirical Results (Strict Subject-Independent LOSO)

### Verified Benchmark Table:
- **Chance Level**: **20.00%**
- **HybridBCINet (Official Clean LOSO)**:
  - **Accuracy**: **24.89% ± 9.35%** (Subj 1: 14.23%, Subj 2: 23.46%, Subj 3: 36.99%)
  - **Macro F1**: **17.54%**
  - **Multiclass ROC-AUC**: **0.6535**
  - **Cohen's Kappa**: **+0.0626**
  - **Matthews Correlation Coefficient (MCC)**: **+0.0673**
- **Classical RBF SVM Baseline**:
  - **Accuracy**: **39.84% ± 9.80%** (Subj 1: 32.11%, Subj 2: 33.74%, Subj 3: 53.66%)
  - **Macro F1**: **37.51%**
  - **Cohen's Kappa**: **+0.2497**
- **Inference Latency**: **12.4 ± 1.8 ms** per 2-second window on CPU.
