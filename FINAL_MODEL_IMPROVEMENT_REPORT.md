# Final Model Improvement Report: Leave-One-Subject-Out (LOSO) Evaluation

**Project:** BRAIN-AI-HYBRID-INTERFACE-SIMULATOR  
**Conference:** ICCES 2026  
**Date:** October 7, 2026  
**Status:** Validated Empirical Findings (Zero Leakage)  

---

## 1. Original Clean Baseline

Prior to this controlled improvement study, the verified subject-independent baseline performance of the unweighted `HybridBCINet` and the classical RBF-SVM baseline on PhysioNet EEGBCI (3 subjects: 1, 2, 3; $N = 735$ non-overlapping 2-second windows) was:

- **Original Clean HybridBCINet (Unweighted CE, 731.3K parameters):**
  - **Accuracy:** $24.89\% \pm 9.35\%$
  - **Macro F1-Score:** $17.54\%$
  - **Macro Precision:** $24.32\%$
  - **Macro Recall:** $24.94\%$
  - **ROC-AUC:** $0.6535$
  - **Cohen's Kappa ($\kappa$):** $+0.0626$
  - **Matthews Correlation Coefficient (MCC):** $+0.0673$
  - *Failure Mode:* Severe majority-class bias on Subject 1 (predicting Rest Eyes Open for 208/246 windows).

- **Classical RBF-SVM Baseline (Handcrafted Spectral/Statistical Features, 513-D):**
  - **Accuracy:** $41.47\% \pm 12.29\%$
  - **Macro F1-Score:** $38.73\%$
  - **Linear SVM:** $38.05\% \pm 11.59\%$ (Macro F1: $34.12\%$)
  - **Random Forest:** $36.70\% \pm 6.82\%$ (Macro F1: $33.54\%$)
  - **Logistic Regression:** $37.52\% \pm 12.38\%$ (Macro F1: $32.70\%$)

---

## 2. Dataset Used

- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI).
- **Format:** European Data Format (`.edf`) raw recordings.
- **Acquisition Protocol:** 64-channel BCI2000 system downsampled to 19 standard 10-20 international channels.
- **Harmonized Task Runs:**
  - **Run 1:** Baseline Eyes Open (Rest / Class 0)
  - **Run 2:** Baseline Eyes Closed (Rest / Class 0)
  - **Run 3:** Task 1 — Motor Imagery / Execution Left Fist (Class 1) vs. Right Fist (Class 2)
  - **Run 4:** Task 2 — Motor Imagery / Execution Both Fists (Class 3) vs. Both Feet (Class 4)
  - **Run 6:** Task 4 — Motor Imagery Both Fists (Class 3) vs. Both Feet (Class 4)

---

## 3. Number of Subjects

- **Primary Cohort (Benchmark Set):** 3 subjects (Subjects 1, 2, 3).
- **Extended Evaluation Cohort (Scaling Study):** 5 subjects (Subjects 1, 2, 3, 4, 5).

---

## 4. Number of Windows

- **Primary 3-Subject Set:** $N = 735$ non-overlapping 2-second windows ($T = 2.0\text{ s}$, 320 samples per window at 160 Hz).
  - Subject 1: 246 windows
  - Subject 2: 243 windows
  - Subject 3: 246 windows
- **Extended 5-Subject Set:** $N = 1,221$ non-overlapping 2-second windows.
  - Subject 4: 243 windows
  - Subject 5: 243 windows

---

## 5. Class Distribution

The class distribution across the 5 distinct task conditions in the 3-subject dataset ($N = 735$):
- **Class 0 (Rest / Baseline):** 90 windows ($12.24\%$)
- **Class 1 (Left Fist Imagery/Execution):** 185 windows ($25.17\%$)
- **Class 2 (Right Fist Imagery/Execution):** 185 windows ($25.17\%$)
- **Class 3 (Both Fists Imagery/Execution):** 90 windows ($12.24\%$)
- **Class 4 (Both Feet Imagery/Execution):** 185 windows ($25.17\%$)

**Class Imbalance Ratio:** $1 : 2.06$ between minority (Rest / Both Fists) and majority (Left/Right Fist, Both Feet).

---

## 6. Preprocessing

1. **Montage Standardization:** 19 channels selected according to the international 10-20 system (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`).
2. **Temporal Filtering:** 4th-order zero-phase Butterworth band-pass filter ($0.5\text{--}45\text{ Hz}$).
3. **Notch Filtering:** 50 Hz IIR notch filter ($Q = 30$) for powerline artifact suppression.
4. **Fold-Isolated Standardization:** Per-channel `StandardScaler` fitted **exclusively on the training partition** of each cross-validation fold:
   $$\mu_{\text{train}} = \frac{1}{N_{\text{train}} T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T X_{i, c, t}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}} T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T (X_{i, c, t} - \mu_{\text{train}})^2}$$
   Held-out test subjects are transformed strictly using $(\mu_{\text{train}}, \sigma_{\text{train}})$.

---

## 7. Windowing

- **Window Length:** 2.0 seconds (320 samples at $f_s = 160\text{ Hz}$).
- **Window Stride:** 2.0 seconds (step = 320 samples, 0% temporal overlap).
- **Integrity Guarantee:** Zero sample overlap prevents intra-session autocorrelation across train/test splits.

---

## 8. Model Architecture

We investigated three neural architectures alongside classical ML baselines:

1. **Full HybridBCINet (731.3K parameters):**
   - Multi-scale 1D Residual Convolutional Blocks (kernel sizes 3, 5, 7)
   - Bidirectional LSTM (hidden size 64 per direction = 128 concatenated)
   - Multi-Head Scaled Dot-Product Attention (4 heads, $d_{\text{model}} = 128$)
   - 2-Layer Transformer Encoder ($d_{\text{model}} = 128$, feed-forward dim 256)
   - Multi-task Prediction Head (Classification logits + auxiliary regression projection)

2. **Residual CNN + BiLSTM (312.4K parameters):**
   - 1D Multi-scale Residual CNN extractor
   - Bidirectional LSTM (128 units total)
   - Temporal Mean Pooling + Linear Classification Head

3. **1D Residual CNN Extractor (86.2K parameters):**
   - 3-Stage 1D Residual Convolutional Blocks with spatial batch normalization, GELU activations, and Squeeze-and-Excitation (SE) channel attention
   - Global Average Pooling (GAP) across time
   - Dense linear classification head with Dropout ($p = 0.3$)

---

## 9. Training Configuration

- **Loss Function:** Class-Weighted Cross-Entropy Loss:
  $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C w_c \cdot y_c \log(\hat{y}_c), \quad w_c = \frac{N_{\text{train}}}{C \cdot N_{c, \text{train}}}$$
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 1 \times 10^{-3}$).
- **Learning Rate:** $\eta = 3 \times 10^{-4}$ with Cosine Annealing learning rate schedule ($\eta_{\text{min}} = 1 \times 10^{-5}$).
- **Batch Size:** 32.
- **Epochs:** 40 epochs per fold.
- **Gradient Clipping:** Max norm $\|g\|_2 \leq 1.0$.

---

## 10. Hyperparameter Search & Inner Validation

Hyperparameter tuning was conducted exclusively on training folds using inner 2-fold cross-validation:
- **Dropout Rates Explored:** 0.2, 0.3, 0.45, 0.5. (Optimal: $0.3$).
- **Weight Decay Rates:** $0, 10^{-4}, 10^{-3}, 10^{-2}$. (Optimal: $10^{-3}$; $10^{-2}$ caused underfitting with accuracy dropping to $25.58\%$).
- **Loss Weighting:** Inverse frequency weighting strictly prevented single-class collapse.

---

## 11. Best Configuration

- **Best Deep Learning Architecture:** **1D Residual CNN Extractor (with Global Average Pooling and SE Attention)**.
- **Key Insight:** In small-cohort subject-independent BCI regimes ($N = 3\text{--}5$ subjects), deep overparameterized sequence models ($>700\text{k}$ parameters with BiLSTM and Transformers) suffer from cross-subject overfitting to idiosyncratic temporal dynamics. A compact, regularized 1D Residual CNN ($86\text{k}$ params) captures invariant spatial-spectral temporal receptive fields more reliably across subjects.

---

## 12. LOSO Protocol

Leave-One-Subject-Out (LOSO) cross-validation was strictly executed:
- **Fold 1:** Train on Subjects 2 & 3 ($N_{\text{train}} = 489$), Test on Subject 1 ($N_{\text{test}} = 246$).
- **Fold 2:** Train on Subjects 1 & 3 ($N_{\text{train}} = 492$), Test on Subject 2 ($N_{\text{test}} = 243$).
- **Fold 3:** Train on Subjects 1 & 2 ($N_{\text{train}} = 489$), Test on Subject 3 ($N_{\text{test}} = 246$).

---

## 13. Final Results Table

| Model | Evaluation Protocol | Accuracy (Mean ± SD) | Macro F1 (%) | ROC-AUC | Cohen's $\kappa$ | MCC | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RBF SVM (Classical Baseline)** | Strict 3-Fold LOSO | **$41.47\% \pm 12.29\%$** | **$38.73\%$** | **$0.7620$** | **$+0.2497$** | **$+0.2646$** | N/A (Dual) |
| **1D Residual CNN (Best Deep Model)** | Strict 3-Fold LOSO | **$34.17\% \pm 9.84\%$** | **$31.50\%$** | **$0.7313$** | **$+0.2132$** | **$+0.2320$** | $86,213$ |
| **Residual CNN + BiLSTM** | Strict 3-Fold LOSO | $32.27\% \pm 7.90\%$ | $31.05\%$ | $0.7356$ | $+0.1986$ | $+0.2265$ | $312,421$ |
| **Full HybridBCINet (Weighted CE)** | Strict 3-Fold LOSO | $29.93\% \pm 3.01\%$ | $29.00\%$ | $0.7031$ | $+0.1536$ | $+0.1675$ | $731,340$ |
| **Linear SVM** | Strict 3-Fold LOSO | $38.05\% \pm 11.59\%$ | $34.12\%$ | $0.7180$ | $+0.2110$ | $+0.2240$ | N/A (Linear) |
| **Random Forest** | Strict 3-Fold LOSO | $36.70\% \pm 6.82\%$ | $33.54\%$ | $0.6950$ | $+0.1945$ | $+0.2012$ | N/A (100 trees) |
| **Logistic Regression** | Strict 3-Fold LOSO | $37.52\% \pm 12.38\%$ | $32.70\%$ | $0.7120$ | $+0.2050$ | $+0.2180$ | $2,570$ |
| **Original Clean HybridBCINet (Unweighted)** | Strict 3-Fold LOSO | $24.89\% \pm 9.35\%$ | $17.54\%$ | $0.6535$ | $+0.0626$ | $+0.0673$ | $731,340$ |
| *Theoretical Chance Level (5 Classes)* | Theoretical | $20.00\%$ | $20.00\%$ | $0.5000$ | $0.0000$ | $0.0000$ | — |

---

## 14. Per-Subject Results

### Best Deep Model (1D Residual CNN):
- **Subject 1 (Held Out):** Accuracy = $20.33\%$, Macro F1 = $17.23\%$, ROC-AUC = $0.6327$, Kappa = $+0.0742$
- **Subject 2 (Held Out):** Accuracy = $39.92\%$, Macro F1 = $35.77\%$, ROC-AUC = $0.7583$, Kappa = $+0.2562$
- **Subject 3 (Held Out):** Accuracy = $42.28\%$, Macro F1 = $41.52\%$, ROC-AUC = $0.8028$, Kappa = $+0.3093$

### Full HybridBCINet (with Weighted Cross-Entropy):
- **Subject 1 (Held Out):** Accuracy = $26.02\%$, Macro F1 = $23.68\%$, ROC-AUC = $0.6485$, Kappa = $+0.1302$
- **Subject 2 (Held Out):** Accuracy = $30.45\%$, Macro F1 = $28.54\%$, ROC-AUC = $0.6986$, Kappa = $+0.1219$
- **Subject 3 (Held Out):** Accuracy = $33.33\%$, Macro F1 = $34.77\%$, ROC-AUC = $0.7624$, Kappa = $+0.2088$

---

## 15. Confusion Matrices (Aggregated Over All 3 LOSO Folds, N = 735)

### 1D Residual CNN (Best Deep Model):
$$\begin{pmatrix}
\text{Rest} & 43 & 5 & 20 & 15 & 7 \\
\text{Left Fist} & 34 & 47 & 9 & 61 & 34 \\
\text{Right Fist} & 85 & 1 & 40 & 54 & 5 \\
\text{Both Fists} & 5 & 7 & 4 & 74 & 0 \\
\text{Both Feet} & 72 & 5 & 13 & 48 & 47
\end{pmatrix}$$

### Full HybridBCINet (731k, Weighted CE):
$$\begin{pmatrix}
\text{Rest} & 54 & 2 & 14 & 8 & 12 \\
\text{Left Fist} & 33 & 42 & 5 & 45 & 60 \\
\text{Right Fist} & 97 & 22 & 27 & 18 & 21 \\
\text{Both Fists} & 5 & 10 & 4 & 65 & 6 \\
\text{Both Feet} & 92 & 13 & 10 & 38 & 32
\end{pmatrix}$$

---

## 16. RBF-SVM vs. Deep Learning Comparison

1. **Margin Maximization in RKHS:** The classical RBF-SVM achieves $41.47\%$ accuracy by operating on 513 engineered features (bandpowers in delta, theta, alpha, beta, gamma across 19 channels, plus Hjorth complexity/mobility and differential entropy). The RBF kernel computes high-dimensional maximum margin boundaries that are less vulnerable to sample scarcity than deep networks.
2. **Deep Learning Generalization:** Deep models trained from raw time-series directly must learn both spectral decomposition and spatial filtering from only 489 training windows per fold. Despite this challenge, the compact 1D Residual CNN achieved $34.17\%$ accuracy ($+9.28\%$ over unweighted baseline), successfully demonstrating genuine deep feature learning.

---

## 17. Ablation Study Results

The ablation study systematically isolates the contribution of each architectural stage:
- **CNN Only ($86\text{k}$ params):** $34.17\% \pm 9.84\%$ Acc, $31.50\%$ F1. Highest generalizability.
- **CNN + BiLSTM ($312\text{k}$ params):** $32.27\% \pm 7.90\%$ Acc, $31.05\%$ F1. Moderate stability.
- **Full HybridBCINet ($731\text{k}$ params):** $29.93\% \pm 3.01\%$ Acc, $29.00\%$ F1. Lower variance across subjects ($\pm 3.01\%$) and best Subject 1 performance ($26.02\%$), but lower peak accuracy due to parameter count.

---

## 18. Cohort Scaling Analysis (5 Subjects, N = 1,221)

To evaluate scaling behavior, we evaluated 5 PhysioNet subjects (Subjects 1–5, $N = 1,221$ windows):
- **RBF SVM (5-Fold LOSO):** $28.95\% \pm 10.04\%$ Acc, $27.50\%$ Macro F1.
- **HybridBCINet (5-Fold LOSO):** $21.43\% \pm 6.89\%$ Acc, $21.54\%$ Macro F1.
- *Finding:* Subject 5 exhibited significant anatomical/signal variance ($9.05\%$ accuracy), demonstrating the well-documented "BCI illiteracy" phenomenon where specific subjects exhibit non-canonical sensorimotor rhythm dynamics.

---

## 19. Scientific Leakage Checks

All checks in `FINAL_LEAKAGE_CHECK.md` passed:
- Preprocessing fit strictly on training partitions.
- Class weights derived exclusively from training partitions.
- Zero subject overlap, zero window overlap, zero test-set feedback.

---

## 20. Limitations

1. **Small Subject Pool:** Evaluating 3 to 5 subjects under LOSO is subject to high inter-subject anatomical variance.
2. **Non-stationary EEG:** Cross-subject sensorimotor rhythm shifts reduce deep model transferability without domain adaptation (e.g., CORAL, DANN, or Riemannian alignment).
3. **Hardware Constraints:** Training was conducted on standard CPU compute without large-scale pretraining.

---

## 21. Recommended Metrics for Manuscript

When updating the manuscript after this experiment:
- **Report RBF-SVM Baseline:** Accuracy = $41.47\% \pm 12.29\%$, Macro F1 = $38.73\%$.
- **Report Improved 1D Residual CNN:** Accuracy = $34.17\% \pm 9.84\%$, Macro F1 = $31.50\%$, ROC-AUC = $0.7313$.
- **Report Full HybridBCINet (Weighted CE):** Accuracy = $29.93\% \pm 3.01\%$, Macro F1 = $29.00\%$, ROC-AUC = $0.7031$.
- **Report Unweighted Clean Baseline:** Accuracy = $24.89\% \pm 9.35\%$, Macro F1 = $17.54\%$.
- **Clearly Highlight:** Zero data leakage, strict LOSO cross-validation, and an honest account of deep learning challenges in subject-independent BCI.
