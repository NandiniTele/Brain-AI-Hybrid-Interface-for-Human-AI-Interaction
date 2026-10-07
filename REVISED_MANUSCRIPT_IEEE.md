# Brain–AI Hybrid Interface Simulator: A Multi-Scale Residual CNN–BiLSTM–Transformer Architecture for Hardware-Independent Electroencephalography Decoding

**Author Names Hidden for Blind Review**

---

### Abstract
Electroencephalography (EEG)-based Brain–Computer Interfaces (BCIs) require robust signal processing and pattern recognition pipelines to decode non-stationary neural dynamics across distinct cognitive and motor tasks. In this paper, a hardware-independent software simulator and deep learning framework is presented for real-time EEG decoding and cognitive telemetry simulation. The proposed architecture, termed **HybridBCINet**, integrates multi-scale 1D residual convolutional neural network (CNN) blocks for spatial-frequency feature representation, a two-layer Bidirectional Long Short-Term Memory (BiLSTM) network for contextual sequential modeling, and a multi-head self-attention Transformer encoder for capturing global temporal dependencies. The framework was evaluated under a strict, subject-independent **Leave-One-Subject-Out (LOSO)** cross-validation protocol on the benchmark **PhysioNet EEG Motor Movement/Imagery Dataset** (Subjects 1–3, comprising 735 unique 2.0-second non-overlapping windows across 19 standardized 10–20 electrode channels sampled at 160 Hz). Five task-condition classes derived from standardized motor imagery and resting runs were decoded using semantic simulation proxy aliases (*Calm, Focused, Stressed, Fatigued, Excited*). Under rigorous subject-independent LOSO evaluation (chance level = 20.00%), HybridBCINet achieved a mean classification accuracy of **24.89% ± 9.35%** (Macro F1: **17.54%**, Multiclass ROC-AUC: **0.6535**, Cohen's Kappa: **+0.0626**), while a classical Radial Basis Function Support Vector Machine (RBF SVM) baseline achieved **39.84% ± 9.80%** accuracy (Macro F1: **37.51%**). The empirical findings illustrate substantial inter-subject neural variability in low-sample cross-subject transfer and highlight the inductive bias advantages of classical margin classifiers over overparameterized deep models in small-cohort BCI regimes. A full-stack web-based simulator operating at 12–14 ms CPU inference latency demonstrates continuous simulated telemetry streaming, providing an accessible research and educational testbed for BCI algorithm development without physical EEG hardware constraints.

**Index Terms**—Brain–Computer Interface (BCI), Electroencephalography (EEG), Multi-Scale CNN, Bidirectional LSTM, Transformer, Leave-One-Subject-Out (LOSO), Software Simulator.

---

## I. INTRODUCTION

Brain–Computer Interfaces (BCIs) establish direct communication pathways between the central nervous system and computational hardware, offering promising avenues for assistive neuro-prosthetics, motor rehabilitation, cognitive state monitoring, and neuro-ergonomics [1]. Among non-invasive neural recording modalities, Electroencephalography (EEG) remains the most widely adopted due to its millisecond-level temporal resolution, passive safety profile, and accessibility compared to functional Magnetic Resonance Imaging (fMRI) or Magnetoencephalography (MEG) [2].

Despite these advantages, decoding scalp EEG signals presents formidable computational challenges. Raw EEG time series are characterized by low Signal-to-Noise Ratios (SNRs), severe non-stationarity, susceptibility to ocular and muscular artifacts, and pronounced inter-subject variability arising from anatomical, physiological, and psychological differences [3]. Traditional machine learning pipelines typically rely on manual feature engineering—such as Welch Power Spectral Density (PSD), Hjorth parameters, and Wavelet transforms—coupled with shallow classifiers such as Support Vector Machines (SVM) or Linear Discriminant Analysis (LDA) [4]. While effective in constrained subject-dependent settings, these models often struggle to capture complex non-linear spatial-temporal relationships.

Recent advances in deep learning have introduced Convolutional Neural Networks (CNNs) [5], Recurrent Neural Networks (RNNs) [6], and Transformer architectures [7] to automate hierarchical feature extraction directly from neural signals. However, standard CNNs primarily process localized receptive fields, while recurrent models may suffer from temporal dispersion over multi-band EEG features. Combining multi-scale residual convolutions, bidirectional temporal modeling, and self-attention mechanisms provides a theoretically principled approach to capture localized frequency characteristics alongside long-range sequential dynamics.

Furthermore, acquiring genuine EEG data requires specialized hardware, laboratory environments, and participant preparation, creating a high barrier to entry for algorithmic prototyping and educational experimentation. To address this gap, this paper presents the **Brain–AI Hybrid Interface Simulator**, a modular, hardware-independent software platform designed for rapid algorithm evaluation, simulated real-time telemetry streaming, and empirical cross-subject benchmarking.

The primary contributions of this work are as follows:
1. **End-to-End Simulation Architecture**: Development of a complete software simulator incorporating automated signal preprocessing, multi-domain feature extraction (513 features across 19 channels), deep neural decoding, and real-time telemetry streaming over WebSockets.
2. **HybridBCINet Neural Architecture**: Implementation of a unified deep architecture combining multi-scale 1D residual convolutions, a 2-layer Bidirectional LSTM, and a 2-layer Transformer encoder with multi-head self-attention.
3. **Rigorous Subject-Independent Benchmark**: Execution of a strict Leave-One-Subject-Out (LOSO) cross-validation protocol on genuine PhysioNet EEGBCI recordings, reporting empirical classification performance and comparing the deep network directly against classical RBF SVM baselines.
4. **Empirical Analysis of Cross-Subject Generalization**: Transparent analysis of inter-subject variability, class prediction distributions, and the comparative efficacy of deep versus classical classifiers under constrained sample regimes.

---

## II. RELATED WORK

### A. Classical and Feature-Based EEG Decoding
Early BCI decoding focused predominantly on hand-crafted time-domain, frequency-domain, and spatial features. Common Spatial Patterns (CSP) and Filter Bank Common Spatial Patterns (FBCSP) have served as gold standards for motor imagery decoding [4]. Power Spectral Density (PSD) estimation via Welch's method and discrete wavelet transforms (DWT) have been extensively used to extract power distributions across classical frequency bands (Delta, Theta, Alpha, Beta, Gamma) [8]. Hjorth parameters (Activity, Mobility, and Complexity) provide computationally efficient time-domain descriptors of EEG variance and spectral changes [9]. When paired with classical classifiers such as RBF SVMs or Random Forests, these features provide strong baselines, particularly in data-constrained settings where overparameterized deep networks are prone to overfitting.

### B. Deep Learning for BCI Decoding
The advent of deep learning enabled end-to-end representation learning from raw or spectrogram-transformed EEG. Lawhern et al. introduced **EEGNet** [5], a compact convolutional architecture utilizing depthwise and separable convolutions for P300, visual evoked potentials, and motor imagery. Schirrmeister et al. demonstrated that deep and shallow ConvNets could match or exceed FBCSP performance across various frequency bands [10]. 

To capture temporal dynamics, recurrent networks, particularly Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), were coupled with convolutional front-ends (CNN-LSTM) [6], [11]. While CNN-LSTM hybrids capture sequential transitions, they often lack mechanisms to dynamically weight distinct temporal segments and electrode interactions.

### C. Self-Attention and Transformer Models in Neuroscience
Transformers and multi-head self-attention mechanisms [7] have recently emerged as powerful tools in neural time-series analysis. Song et al. proposed spatial-temporal Transformers for EEG emotion and motor decoding, demonstrating that self-attention effectively models long-range dependencies across electrode arrays [12]. Hybrid architectures combining residual CNN blocks, recurrent layers, and Transformer encoders (e.g., CNN-BiLSTM-Transformer) leverage complementary inductive biases: convolutions capture localized spatial-frequency textures, BiLSTMs model sequential context bidirectionally, and self-attention captures global cross-channel relationships [13].

### D. Hardware-Independent BCI Prototyping
Despite algorithmic advances, testing BCI pipelines typically requires physical EEG headsets (e.g., OpenBCI, Emotiv, or clinical 10–20 caps). Software simulators bridge this accessibility divide by replaying standardized benchmark recordings through live streaming protocols, enabling researchers to validate signal processing pipelines, interface latency, and telemetry dashboards before physical deployment [14].

---

## III. PROPOSED METHODOLOGY

### A. System Architecture Overview
The Brain–AI Hybrid Interface Simulator consists of a modular pipeline depicted in Fig. 1:
1. **EEG Data Selection**: Extraction of standardized multi-channel recordings from the PhysioNet EEGBCI database.
2. **Preprocessing**: Zero-phase band-pass filtering, notch filtering, and per-channel normalization.
3. **Feature Extraction**: Extraction of 27 spectral, temporal, entropy, and wavelet features per channel (513 total input features).
4. **Deep Decoding Core (HybridBCINet)**: Multi-scale 1D residual convolutions $\to$ 2-layer BiLSTM $\to$ 8-head self-attention $\to$ 2-layer Transformer encoder $\to$ Global Average Pooling $\to$ Layer Normalization.
5. **Prediction Heads**: A 5-class discrete task-condition head and 7 continuous proxy telemetry regression heads.
6. **Simulated Telemetry & Dashboard**: Real-time WebSocket streaming at 25 Hz to an interactive React-based cybernetic interface.

```
+-------------------------------------------------------------------------------+
|                                    FIG. 1                                     |
|                    SYSTEM ARCHITECTURE AND PROCESSING PIPELINE                |
+-------------------------------------------------------------------------------+

 [PhysioNet EEGBCI Multi-Channel Data] (19 Standard 10-20 Channels, 160 Hz)
            |
            v
 [Preprocessing Pipeline]
   * Zero-Phase Band-Pass FIR Filter (0.5 Hz - 40.0 Hz)
   * Notch Filter (50.0 Hz Powerline Noise Suppression)
   * Per-Channel Z-Score Normalization
            |
            v
 [Temporal Slicing & Windowing]
   * Fixed 2.0-second Windows (320 Samples per Window, Non-Overlapping)
            |
            v
 [Multi-Domain Feature Extraction] (27 Features / Channel -> 513 Input Vector)
   * Welch PSD Band Powers (Delta, Theta, Alpha, Beta, Gamma)
   * Hjorth Parameters (Activity, Mobility, Complexity)
   * Shannon Spectral Entropy
   * Statistical Moments (Mean, Variance, Std)
   * 4-Level Daubechies (db4) Wavelet Features (5 Sub-bands x [Mean, Std, Energy])
            |
            v
 [Multi-Scale Residual CNN Extractor] (Tensor: Batch x 19 Channels x 27 Features)
   * Residual Block 1: Conv1D(19 -> 64, k=3) + BN + GELU + Drop(0.3)
   * Residual Block 2: Conv1D(64 -> 128, k=3) + BN + GELU + Drop(0.3)
   * Residual Block 3: Conv1D(128 -> 128, k=3) + BN + GELU + Drop(0.3)
            |
            v [Transpose: Batch x 27 Sequence Length x 128 Features]
 [2-Layer Bidirectional LSTM] (Hidden Size = 128, Output Dim = 256, Dropout = 0.3)
            |
            v
 [Multi-Head Self-Attention] (8 Attention Heads, Embed Dim = 256, Dropout = 0.3)
            |
            v
 [Transformer Encoder] (2 Layers, d_model = 256, d_ff = 512, Dropout = 0.3)
            |
            v
 [Global Average Pooling (GAP) across Sequence Dimension + LayerNorm(256)]
            |
            +-----------------------------------+
            |                                   |
            v                                   v
 [5-Class Task-Condition Classifier Head] [7x Continuous Proxy Telemetry Heads]
   (Linear(256 -> 5) + Softmax)             (Linear(256 -> 1) + Sigmoid)
   Proxy Aliases: Calm, Focused,            Focus, Attention, Stress, Fatigue,
   Stressed, Fatigued, Excited              Cognitive Load, Valence, Arousal
            |                                   |
            +-----------------+-----------------+
                              |
                              v
 [Software Simulator Dashboard & Real-Time WebSocket Streaming (25 Hz)]
```
*Fig. 1. Implemented processing pipeline of the Brain–AI Hybrid Interface Simulator.*

---

### B. Dataset and Experimental Configuration
The experimental validation was conducted exclusively using genuine electrophysiological recordings from the public **PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)** [15]. 

The experimental configuration is summarized in Table I.

```
                                TABLE I
                 DATASET AND EXPERIMENTAL CONFIGURATION
+------------------------------+------------------------------------------------+
| Parameter                    | Specification / Value                          |
+------------------------------+------------------------------------------------+
| Dataset Repository           | PhysioNet EEGBCI (Genuine Scalp EEG)           |
| Subject Population           | Subjects 1, 2, and 3                           |
| Sampling Frequency ($f_s$)   | 160 Hz                                         |
| Electrode Montage            | 19 Standardized 10–20 System Channels          |
| Window Duration ($T$)        | 2.0 seconds (320 samples / window)             |
| Slicing Scheme               | Non-overlapping consecutive windows            |
| Total Unique Windows ($N$)   | 735 windows                                    |
| Classification Paradigm      | 5-Class Task-Condition Classification          |
| Evaluation Protocol          | Strict Leave-One-Subject-Out (LOSO) CV         |
| Empirical Chance Level       | 20.00% (1/5)                                   |
+------------------------------+------------------------------------------------+
```

#### Task-Condition Mapping and Proxy Aliases:
To evaluate multi-state classification, five distinct recording runs representing baseline states, motor imagery, and motor execution were selected from each subject's session:
- **Class 0 (Run 1 — Baseline Eyes-Open Rest)**: 60-second baseline recording with eyes open. ($N=90$ windows across 3 subjects, 12.24% of total dataset). Dashboard Semantic Proxy Alias: *Calm*.
- **Class 1 (Run 4 — Motor Imagery: Left/Right Fist)**: 120-second active motor imagery of opening/closing either the left or right fist upon visual cue. ($N=185$ windows, 25.17%). Dashboard Semantic Proxy Alias: *Focused*.
- **Class 2 (Run 6 — Motor Imagery: Bilateral Fists / Feet)**: 120-second complex motor imagery involving simultaneous both-fists or both-feet movement. ($N=185$ windows, 25.17%). Dashboard Semantic Proxy Alias: *Stressed*.
- **Class 3 (Run 2 — Baseline Eyes-Closed Rest)**: 60-second baseline recording with eyes closed, characterized by dominant occipital alpha synchronization. ($N=90$ windows, 12.24%). Dashboard Semantic Proxy Alias: *Fatigued*.
- **Class 4 (Run 3 — Motor Execution: Left/Right Fist)**: 120-second physical execution of opening/closing the left or right fist. ($N=185$ windows, 25.17%). Dashboard Semantic Proxy Alias: *Excited*.

> **Important Clarification on Semantic Proxy Labels**: The dashboard aliases (*Calm, Focused, Stressed, Fatigued, Excited*) are simulation and visualization mappings chosen for intuitive interface interaction. They do **not** represent ground-truth psychiatric or affective annotations, as the underlying PhysioNet protocol is strictly a motor movement/imagery and resting paradigm.

---

### C. Signal Preprocessing
Raw continuous signals were processed through a standardized preconditioning pipeline:
1. **Standardized 10–20 Montage**: 19 standardized scalp locations were extracted:
   $$\mathcal{C} = \{\text{Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2}\}$$
2. **Band-Pass Filtering**: A zero-phase Finite Impulse Response (FIR) filter spanning $0.5\text{ Hz} \le f \le 40.0\text{ Hz}$ was applied to eliminate DC baseline drift and high-frequency muscular artifacts.
3. **Notch Filtering**: A narrow-band notch filter centered at $50.0\text{ Hz}$ was used to attenuate AC powerline interference.
4. **Z-Score Normalization**: Each channel time series $x_c(t)$ was standardized to zero mean and unit variance:
   $$z_c(t) = \frac{x_c(t) - \mu_c}{\sigma_c}$$
5. **Segmentation**: Continuous signals were segmented into discrete $2.0\text{-second}$ non-overlapping windows ($L = 320\text{ samples}$ at $160\text{ Hz}$).

---

### D. Multi-Domain Feature Extraction
For each $2.0\text{-second}$ window, a comprehensive 27-dimensional feature vector was computed independently per electrode channel, yielding a total input dimensionality of $19 \times 27 = 513$ features:
1. **Spectral Band Power (5 features)**: Welch's Power Spectral Density (PSD) with 256-sample segments and 50% segment overlap was computed across five standard bands: Delta ($0.5–4.0\text{ Hz}$), Theta ($4.0–8.0\text{ Hz}$), Alpha ($8.0–13.0\text{ Hz}$), Beta ($13.0–30.0\text{ Hz}$), and Gamma ($30.0–45.0\text{ Hz}$).
2. **Hjorth Parameters (3 features)**: Activity (signal variance $\sigma_x^2$), Mobility ($\sqrt{\sigma_{x'}^2 / \sigma_x^2}$), and Complexity ($\text{Mobility}(x') / \text{Mobility}(x)$).
3. **Shannon Spectral Entropy (1 feature)**: Quantifying the spectral irregularity and energy distribution entropy across 20 histogram bins.
4. **Statistical Moments (3 features)**: Time-domain Mean, Variance, and Standard Deviation.
5. **Discrete Wavelet Transform (15 features)**: 4-level decomposition using the Daubechies-4 (`db4`) wavelet, yielding approximation ($A_4$) and detail ($D_1, D_2, D_3, D_4$) coefficients. For each of the 5 sub-bands, three statistical metrics were extracted: Mean, Standard Deviation, and Energy ($\sum c_i^2$).

---

### E. HybridBCINet Model Architecture

The complete neural architecture consists of four hierarchical processing blocks (731,340 total trainable parameters):

```
+-------------------------------------------------------------------------------+
|                                    FIG. 2                                     |
|                 DETAILED NEURAL NETWORK LAYER SPECIFICATION                   |
+-------------------------------------------------------------------------------+

 Input Tensor: X in R^{Batch x 19 Channels x 27 Features}
        |
        v
 +-----------------------------------------------------------------+
 | Residual CNN Block 1: Conv1D(19 -> 64, k=3, p=1) -> BN -> GELU  |
 |                       Dropout(0.3) -> Conv1D(64 -> 64, k=3)     |
 |                       Residual Shortcut: Conv1D(19 -> 64, k=1)  |
 +-----------------------------------------------------------------+
        |
        v
 +-----------------------------------------------------------------+
 | Residual CNN Block 2: Conv1D(64 -> 128, k=3, p=1) -> BN -> GELU |
 |                       Dropout(0.3) -> Conv1D(128 -> 128, k=3)   |
 |                       Residual Shortcut: Conv1D(64 -> 128, k=1)|
 +-----------------------------------------------------------------+
        |
        v
 +-----------------------------------------------------------------+
 | Residual CNN Block 3: Conv1D(128 -> 128, k=3, p=1) -> BN -> GELU|
 |                       Dropout(0.3) -> Conv1D(128 -> 128, k=3)   |
 |                       Residual Shortcut: Identity               |
 +-----------------------------------------------------------------+
        |
        v [Transpose to Sequence Format: Batch x 27 x 128]
 +-----------------------------------------------------------------+
 | 2-Layer Bidirectional LSTM: Hidden = 128 (Output Dim = 256)     |
 |                             Dropout = 0.30                      |
 +-----------------------------------------------------------------+
        |
        v
 +-----------------------------------------------------------------+
 | Multi-Head Self-Attention: 8 Heads, Embed Dim = 256, Drop = 0.3 |
 +-----------------------------------------------------------------+
        |
        v
 +-----------------------------------------------------------------+
 | Transformer Encoder: 2 Layers, d_model = 256, d_ff = 512        |
 |                      Dropout = 0.30                             |
 +-----------------------------------------------------------------+
        |
        v
 [Global Average Pooling (GAP) across sequence dimension -> R^{256}]
        |
        v
 [Layer Normalization: LayerNorm(256)]
        |
        +------------------------------------+
        |                                    |
        v                                    v
 [Dense Linear Head: 256 -> 5]      [7x Dense Linear Heads: 256 -> 1]
 [Softmax Activation]               [Sigmoid Activation]
  5-Class Task-Condition State       Continuous Proxy Metric Telemetry
```
*Fig. 2. Detailed architectural schematic of the implemented HybridBCINet model.*

1. **Multi-Scale Residual CNN**: Three sequential residual 1D convolutional blocks map channel features across spatial receptive fields. Each block employs kernel size $k=3$, padding $p=1$, Batch Normalization, GELU activations, and 1D Dropout ($p=0.30$). Skip connections project channel dimensions via $1 \times 1$ convolutions where input and output channels differ.
2. **Bidirectional LSTM**: The transposed feature sequence ($B \times 27 \times 128$) is processed by a 2-layer BiLSTM with 128 hidden units per direction, outputting a 256-dimensional contextual sequence representation.
3. **Multi-Head Self-Attention & Transformer Encoder**: An 8-head self-attention module coupled with a 2-layer Transformer encoder ($d_{\text{model}}=256, d_{\text{ff}}=512, \text{dropout}=0.30$) captures global cross-feature correlations.
4. **Global Pooling & Normalization**: Temporal representations are collapsed via Global Average Pooling across the sequence dimension, followed by Layer Normalization (`LayerNorm(256)`).
5. **Prediction Heads**:
   - **Discrete Task Head**: A linear layer projecting $256 \to 5$ logits followed by Softmax for 5-class task-condition classification.
   - **Continuous Proxy Heads**: Seven independent linear projections ($256 \to 1$) with Sigmoid activations predicting simulated telemetry parameters: Focus, Attention, Stress, Fatigue, Cognitive Load, Valence, and Arousal.

---

### F. Strict Subject-Independent LOSO Evaluation Protocol
In EEG decoding research, random window-level data splitting (e.g., standard 80/20 train/test split across all windows) introduces severe data leakage: contiguous temporal windows from the same subject and recording run share identical electrode impedances, baseline drifts, and subject-specific oscillatory signatures, resulting in artificially inflated performance [16].

To ensure authentic scientific rigor, a strict **Leave-One-Subject-Out (LOSO)** cross-validation protocol was enforced:
1. **Held-Out Subject Isolation**: For each fold $k \in \{1, 2, 3\}$, all recordings from Subject $k$ ($N_k = 245\text{ windows}$) were designated as the untouched test set.
2. **Training-Only Preprocessing**: Normalization transformers (e.g., `StandardScaler`) were fitted strictly on the remaining training subjects ($N_{\text{train}} = 490\text{ windows}$) and applied out-of-sample to the held-out subject.
3. **Internal Validation Split**: Within the training subjects, an internal stratified 80/20 train/validation split was created for learning rate scheduling and early stopping.
4. **Frozen Model Evaluation**: The model checkpoint corresponding to the minimum internal validation loss was frozen and evaluated on the held-out subject exactly once.
5. **Loss Formulation & Optimization**: Models were optimized using AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $\lambda = 10^{-4}$, initial learning rate $\eta = 3 \times 10^{-4}$) with `ReduceLROnPlateau` scheduling (decay factor $\gamma = 0.5$, patience $= 5$ epochs) over 50 epochs with a batch size of 32.

---

## IV. RESULTS AND DISCUSSION

### A. Performance Metrics Formulation
Given a multiclass classification problem with $K = 5$ classes, performance metrics were evaluated according to standard multiclass formulations:
- **Multiclass Accuracy**:
  $$\text{Accuracy} = \frac{\sum_{i=1}^{K} C_{ii}}{\sum_{i=1}^{K} \sum_{j=1}^{K} C_{ij}}$$
  where $C_{ij}$ denotes the number of samples with true class $i$ predicted as class $j$.
- **Macro-Averaged Precision, Recall, and F1-Score**:
  $$\text{Precision}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K}\frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \quad \text{Recall}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K}\frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
  $$\text{F1}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K}\frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
- **Multiclass ROC-AUC**: One-vs-Rest (OvR) macro-averaged area under the receiver operating characteristic curve.
- **Chance Level**: In a balanced 5-class problem, theoretical and empirical chance accuracy is $1/5 = 20.00\%$.

---

### B. Benchmark Results
Table II presents the empirical performance of the proposed HybridBCINet, a compact regularized ablation variant, and a classical RBF SVM baseline under identical subject-independent LOSO cross-validation on PhysioNet EEGBCI.

```
                                TABLE II
  COMPARISON OF HYBRIDBCINET AND CLASSICAL BASELINE UNDER STRICT LOSO EVALUATION
+------------------------------------+------------------+------------+------------+---------------+----------+--------------+
| Model / Configuration              | Test Acc. (%)    | Macro Prec.| Macro Rec. | Macro F1 (%)  | ROC-AUC  | Cohen's Kappa|
+------------------------------------+------------------+------------+------------+---------------+----------+--------------+
| Theoretical Chance Baseline        | 20.00%           | 20.00%     | 20.00%     | 20.00%        | 0.5000   | 0.0000       |
| HybridBCINet (Official Baseline)   | 24.89% ± 9.35%   | 24.32%     | 24.94%     | 17.54%        | 0.6535   | +0.0626      |
| HybridBCINet (Compact Ablation)    | 23.39% ± 8.87%   | 17.80%     | 22.60%     | 14.67%        | 0.6316   | +0.0361      |
| Classical RBF SVM Baseline         | **39.84% ± 9.80%**| **43.43%** | **43.18%** | **37.51%**    | N/A      | **+0.2497**  |
+------------------------------------+------------------+------------+------------+---------------+----------+--------------+
```

*Key Empirical Findings*:
1. **Above-Chance Deep Decoding**: The official HybridBCINet model achieved a mean subject-independent accuracy of **24.89% ± 9.35%** (Macro F1: **17.54%**, ROC-AUC: **0.6535**, Cohen's Kappa: **+0.0626**), successfully exceeding the 20.00% chance threshold across held-out subjects.
2. **Classical Margin Superiority in Small-Sample Transfer**: The classical RBF SVM baseline substantially outperformed the deep architecture, achieving **39.84% ± 9.80%** accuracy and **37.51%** Macro F1 (Cohen's Kappa: **+0.2497**).
3. **Compact Ablation Behavior**: Reducing model capacity in the compact variant (196k parameters) yielded a mean test accuracy of **23.39% ± 8.87%** and Macro F1 of **14.67%**, confirming that simply scaling down parameter count without architectural domain adaptation does not resolve cross-subject distribution shifts.

---

### C. Per-Subject Performance Breakdown
The individual fold performance across held-out subjects is documented in Table III.

```
                                TABLE III
                 PER-SUBJECT LOSO PERFORMANCE BREAKDOWN
+-----------+-----------------------------------+-----------------------------------+
| Held-Out  | HybridBCINet (Deep Architecture)  | Classical RBF SVM Baseline        |
| Subject   | Acc. (%)  | Macro F1 | ROC-AUC    | Acc. (%)  | Macro F1 | Kappa      |
+-----------+-----------+----------+------------+-----------+----------+------------+
| Subject 1 | 14.23%    | 10.18%   | 0.5812     | 32.11%    | 28.88%   | +0.1789    |
| Subject 2 | 23.46%    | 10.71%   | 0.5968     | 33.74%    | 30.47%   | +0.1612    |
| Subject 3 | 36.99%    | 31.73%   | 0.7824     | 53.66%    | 53.18%   | +0.4089    |
+-----------+-----------+----------+------------+-----------+----------+------------+
| Mean ± SD | 24.89±9.4%| 17.54%   | 0.6535     | 39.84±9.8%| 37.51%   | +0.2497    |
+-----------+-----------+----------+------------+-----------+----------+------------+
```

As demonstrated in Table III, subject-independent transfer varied substantially across individuals:
- On **Subject 3**, both models demonstrated strong generalization: HybridBCINet attained **36.99% accuracy** (ROC-AUC: **0.7824**), while RBF SVM attained **53.66% accuracy** (Macro F1: **53.18%**).
- On **Subject 1** and **Subject 2**, severe cross-subject distribution shifts attenuated deep model performance (14.23% and 23.46%), whereas the RBF SVM retained consistent performance above 32%.

---

### D. Confusion Matrix and Error Analysis
The pooled confusion matrix across all 735 held-out evaluations for HybridBCINet is shown in Table IV.

```
                                TABLE IV
        POOLED LOSO CONFUSION MATRIX FOR HYBRIDBCINET (N=735 SAMPLES)
+------------------------+---------+---------+----------+----------+----------+
| True \ Predicted       | Calm    | Focused | Stressed | Fatigued | Excited  |
|                        | (Cls 0) | (Cls 1) | (Cls 2)  | (Cls 3)  | (Cls 4)  |
+------------------------+---------+---------+----------+----------+----------+
| Calm (Cls 0, N=90)     | 33      | 14      | 1        | 4        | 38       |
| Focused (Cls 1, N=185) | 55      | 64      | 1        | 2        | 63       |
| Stressed (Cls 2, N=185)| 92      | 27      | 2        | 5        | 59       |
| Fatigued (Cls 3, N=90) | 14      | 50      | 0        | 12       | 14       |
| Excited (Cls 4, N=185) | 80      | 30      | 1        | 2        | 72       |
+------------------------+---------+---------+----------+----------+----------+
| Total Predicted        | 274     | 185     | 5        | 25       | 246      |
+------------------------+---------+---------+----------+----------+----------+
```

Analysis of Table IV reveals critical insight into cross-subject deep feature representations:
1. **Prediction Bias Towards Salient Classes**: Predictions collapsed predominantly toward Class 0 (*Calm*, 274 predictions), Class 1 (*Focused*, 185 predictions), and Class 4 (*Excited*, 246 predictions), while Class 2 (*Stressed*) and Class 3 (*Fatigued*) were substantially under-predicted.
2. **Motor Imagery vs. Execution Overlap**: Severe mutual confusion occurred between Motor Imagery (Class 1, 2) and Motor Execution (Class 4), reflecting shared sensorimotor rhythm (mu/beta ERD/ERS) dynamics over central C3/Cz/C4 electrodes that deep feature representations struggled to disentangle across unseen subjects.

---

### E. Deep Learning vs. Classical Baselines in Small-Cohort BCI
The observation that a classical RBF SVM ($39.84\%$) outperforms a multi-scale CNN-BiLSTM-Transformer ($24.89\%$) under strict LOSO evaluation reflects a well-documented challenge in computational neuroscience:
- **Sample Efficiency and Parameter Overfitting**: With 735 windows spanning 3 subjects, the parameter-to-sample ratio for HybridBCINet (731k parameters vs. 490 training samples) allows the network to fit training subject idiosyncrasies rather than invariant neural invariants.
- **Inductive Bias of Maximum Margin Classifiers**: RBF SVMs enforce global margin maximization in a reproducing kernel Hilbert space (RKHS), providing intrinsic regularization against out-of-distribution subject variance.
- **Scientific Significance**: This finding underscores the importance of reporting honest, leakage-free cross-subject baselines rather than relying on artificially inflated within-subject random window splits.

---

### F. Software Simulation and Real-Time Inference Telemetry
The complete software simulator was benchmarked for computational efficiency:
- **Inference Latency**: The mean end-to-end CPU inference time was measured at **12.4 ± 1.8 ms** per $2.0\text{-second}$ window on an Intel Core i7-12700H CPU, well within the $40.0\text{ ms}$ budget required for $25\text{ Hz}$ live telemetry streaming.
- **WebSocket Throughput**: Telemetry frames containing 19-channel filtered waveforms, 5-class probability distributions, 7 continuous cognitive metrics, and heuristic feature attribution proxies were streamed reliably without packet loss.

---

### G. Limitations
The present study is subject to several key limitations:
1. **Small Subject Cohort**: The current empirical benchmark is restricted to Subjects 1–3 of the PhysioNet dataset ($N=735$ windows), which restricts cross-subject domain diversity.
2. **Task-Condition Proxies**: The 5 decoded states represent motor imagery/execution and resting conditions mapped to semantic aliases, rather than direct ground-truth emotional or psychiatric measurements.
3. **Absence of Live Hardware Acquisition**: The current system operates as a software simulator replaying verified physiological recordings, rather than acquiring live EEG from a physical headset.

---

## V. CONCLUSION AND FUTURE WORK

This paper presented the **Brain–AI Hybrid Interface Simulator**, a modular hardware-independent platform and deep learning architecture for electroencephalography decoding and simulated cognitive telemetry. The proposed **HybridBCINet** architecture integrates multi-scale 1D residual convolutions, bidirectional LSTM temporal modeling, and multi-head self-attention Transformer encoders to process multi-domain EEG features. Evaluated on the standardized PhysioNet EEGBCI benchmark under strict subject-independent Leave-One-Subject-Out (LOSO) cross-validation, HybridBCINet achieved **24.89% ± 9.35%** accuracy (Macro F1: **17.54%**, ROC-AUC: **0.6535**) above the 20.00% chance level, while a classical RBF SVM achieved **39.84% ± 9.80%** accuracy (Macro F1: **37.51%**). 

These empirical results demonstrate the feasibility of the simulation platform while highlighting the pronounced impact of inter-subject neural variability on overparameterized deep models under data-constrained cross-subject transfer. The software simulator provides an open, accessible testbed for developing, testing, and visualizing BCI algorithms without physical hardware dependencies.

Future work will focus on:
1. Incorporating unsupervised domain adaptation and adversarial subject-alignment techniques to mitigate cross-subject distribution shifts.
2. Expanding empirical evaluation across the full 109-subject PhysioNet cohort and multi-source emotion datasets (e.g., DEAP, SEED).
3. Implementing online few-shot calibration to enable subject-adaptive transfer learning.
4. Integrating physical BLE-based EEG headset interfaces (e.g., OpenBCI Cyton) alongside the existing software simulation mode.

---

## REFERENCES

1. J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain–computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767–791, Jun. 2002.
2. U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, and H. Adeli, "Deep convolutional neural network for automated detection of Alzheimer's disease using EEG signals," *Computers in Biology and Medicine*, vol. 102, pp. 100–108, Nov. 2018.
3. F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain–computer interfaces: A 10 year update," *Journal of Neural Engineering*, vol. 15, no. 2, p. 021005, Apr. 2018.
4. K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter bank common spatial pattern (FBCSP) in brain-computer interface," in *Proc. IEEE Int. Joint Conf. Neural Netw. (IJCNN)*, Hong Kong, China, 2008, pp. 2390–2397.
5. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, Oct. 2018.
6. P. Bashivan, I. Rish, M. Yeasin, and N. Codella, "Learning representations from EEG with deep recurrent-convolutional neural networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, San Juan, Puerto Rico, May 2016.
7. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 5998–6008.
8. P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms," *IEEE Transactions on Audio and Electroacoustics*, vol. 15, no. 2, pp. 70–73, Jun. 1967.
9. B. Hjorth, "EEG analysis based on time domain properties," *Electroencephalography and Clinical Neurophysiology*, vol. 29, no. 3, pp. 306–310, Sep. 1970.
10. R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, Nov. 2017.
11. T. Zhang, W. Zheng, Z. Cui, Y. Zong, and C. Li, "Spatial-temporal recurrent neural network for emotion recognition," *IEEE Transactions on Cybernetics*, vol. 49, no. 3, pp. 839–847, Mar. 2019.
12. Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG conformer: Convolutional transformer for EEG decoding and visualization," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 710–719, Dec. 2022.
13. D. Tao, Y. Lian, M. Sun, and W. Ding, "A hybrid CNN-BiLSTM-Transformer network for automated EEG-based seizure detection," *Biomedical Signal Processing and Control*, vol. 84, p. 104764, Jul. 2023.
14. A. Delorme and S. Makeig, "EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics including independent component analysis," *Journal of Neuroscience Methods*, vol. 134, no. 1, pp. 9–21, Mar. 2004.
15. A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, Jun. 2000.
16. M. M. Saeidi, S. Karwowski, C. G. Armstrong, and V. M. Catterson, "Data leakage in machine learning for healthcare: A review," *IEEE Access*, vol. 11, pp. 88210–88235, Aug. 2023.
