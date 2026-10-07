# RESPONSE TO REVIEWERS' COMMENTS

**Manuscript Title:** Brain–AI Hybrid Interface Simulator: A Multi-Scale Residual CNN–BiLSTM–Transformer Architecture for Hardware-Independent Electroencephalography Decoding  
**Target Venue:** IEEE Transactions / Conference on Neural Systems & Rehabilitation Engineering  
**Date of Revision:** October 2026

The authors sincerely thank the Associate Editor and Reviewers for their insightful, constructive critiques. Below is the point-by-point response detailing all revisions made to the manuscript to align the paper with the verified, leakage-free empirical implementation.

---

## 📋 Summary of Major Revisions

| Issue / Topic | Original Manuscript State | Revised Verified Manuscript | Rationale & Revised Location |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Inconsistent descriptions (CNN-LSTM vs. Transformer). | **HybridBCINet**: Multi-Scale Residual CNN $\to$ BiLSTM $\to$ Self-Attention $\to$ Transformer Encoder $\to$ GAP $\to$ LayerNorm $\to$ Prediction Heads. | Fully harmonized across Title, Abstract, Section I, Section III, Fig. 1, Fig. 2, and Section V. |
| **Dataset Description** | Claimed DEAP, SEED, and 109 PhysioNet subjects evaluated. | **PhysioNet EEGBCI exclusively** (Subjects 1–3, $N=735$ unique 2.0s windows, 19 channels @ 160 Hz). | Corrected in Abstract, Section I, Table I, Section III-B. DEAP/SEED removed from experimental claims. |
| **Evaluation Protocol & Data Leakage** | Random window-level splitting (yielded unverified 98% metrics due to autocorrelation leakage). | **Strict Subject-Independent Leave-One-Subject-Out (LOSO)** cross-validation with internal validation on training subjects. | Section III-F, Table II. Completely eliminates subject and contiguous window leakage. |
| **Task Labels vs. Emotion Claims** | Described 5 classes as ground-truth emotions. | **5-Class Task-Condition Classification** with simulation proxy aliases (*Calm, Focused, Stressed, Fatigued, Excited*). | Clarified in Abstract, Section III-B, Table I, and Section IV-D. |
| **Empirical Results** | Fabricated / unverified 98.20% test accuracy and baseline claims. | **HybridBCINet: 24.89% ± 9.35%** (Macro F1: 17.54%, ROC-AUC: 0.6535) vs. **Classical RBF SVM: 39.84% ± 9.80%** (Macro F1: 37.51%) vs. Chance (20.00%). | Section IV, Table II, Table III, Table IV. All values verified directly against `loso_clean_results.json`. |
| **Hardware Scope** | Implied physical headset acquisition. | **Software Simulator & Research Testbed** for hardware-independent algorithm prototyping. | Abstract, Section I, Section III-A, Section IV-F, Section V. |
| **Stylistic & Formatting** | Personal pronouns ("we", "our") and non-standard structure. | **Standard IEEE passive voice and formal layout**: I. Introduction $\to$ II. Related Work $\to$ III. Methodology $\to$ IV. Results $\to$ V. Conclusion $\to$ References. | Throughout entire manuscript. |

---

## 🔬 Point-by-Point Rebuttal & Detailed Responses

### Reviewer 1 Comments

**Point 1 (Architecture Harmonization & System Unification):**  
*Reviewer Comment:* Abstract describes CNN-LSTM on DEAP, SEED and PhysioNet, but results use a multi-scale CNN-BiLSTM-Transformer mainly on PhysioNet. Make the abstract, methodology, results and conclusion describe one system. Decide final model and dataset roles.  
*Author Response:* The manuscript has been completely harmonized around the **HybridBCINet** architecture across the Abstract, Introduction, Methodology, Results, and Conclusion. The architecture is formally specified as:
$$\text{Residual CNN Blocks } (19 \to 64 \to 128 \to 128) \longrightarrow \text{2-Layer BiLSTM } (256\text{-dim}) \longrightarrow \text{Multi-Head Self-Attention } (8\text{ heads}) \longrightarrow \text{Transformer Encoder } (2\text{ layers}) \longrightarrow \text{GAP} \longrightarrow \text{LayerNorm} \longrightarrow \text{Prediction Heads}$$
PhysioNet EEGBCI (Subjects 1–3, 735 windows) is established as the sole experimental benchmark dataset. All claims of experimental evaluation on DEAP or SEED have been eliminated from the reported results.

**Point 2 (Removal of Old 98% Claims & Strict LOSO Protocol):**  
*Reviewer Comment:* Clarify 98.65% vs 98.20% accuracy numbers. Report validation and test accuracy separately. Address potential leakage from window-level splitting.  
*Author Response:* An exhaustive methodological audit revealed that the earlier 98% accuracy claims resulted from random window-level splitting where correlated windows from the same subject and recording run were shared between training and test sets. All unverified 98% claims have been completely removed. The manuscript now reports the **strict, leakage-free Leave-One-Subject-Out (LOSO)** cross-validation results:
- **HybridBCINet**: Mean Test Accuracy of **24.89% ± 9.35%**, Macro F1 of **17.54%**, Multiclass ROC-AUC of **0.6535**, Cohen's Kappa of **+0.0626**, and Matthews Correlation Coefficient of **+0.0673** (against a 20.00% chance level).
- **Classical RBF SVM Baseline**: Mean Test Accuracy of **39.84% ± 9.80%**, Macro F1 of **37.51%**, and Cohen's Kappa of **+0.2497**.

**Point 3 (Dataset Harmonization & Preprocessing Details):**  
*Reviewer Comment:* Add a subsection called "Data Harmonization" covering sampling rate, channels, filtering, windowing, normalization, and label mapping table. Explicitly state if datasets were not merged.  
*Author Response:* Section III-B, Section III-C, and Table I document the complete preprocessing and harmonization specifications: 19 standardized 10–20 electrode channels (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`), zero-phase FIR band-pass filtering (0.5–40.0 Hz), 50.0 Hz powerline notch filtering, per-channel z-score standardization, 2.0-second non-overlapping temporal windowing (320 samples per window at 160 Hz), and explicit run-to-condition mapping.

**Point 4 (Sample Counts, Subjects, Class Distributions, and Splits):**  
*Reviewer Comment:* Add a dataset table with subjects, samples, classes, class distributions, and train/val/test splits. Correct emotion claims if labels are task-derived.  
*Author Response:* Table I (**DATASET AND EXPERIMENTAL CONFIGURATION**) provides exact sample counts and experimental parameters:
- **Subjects**: 3 (Subjects 1, 2, and 3)
- **Sampling Rate**: 160 Hz
- **Channels**: 19 standardized 10–20 channels
- **Window Length**: 2.0 seconds (320 samples)
- **Total Unique Windows**: 735 windows (Subj 1: 245, Subj 2: 245, Subj 3: 245)
- **Classes**: 5 task conditions (Run 1: Eyes-Open Rest [$N=90$, 12.24%], Run 4: Left/Right Fist MI [$N=185$, 25.17%], Run 6: Bilateral MI [$N=185$, 25.17%], Run 2: Eyes-Closed Rest [$N=90$, 12.24%], Run 3: Left/Right Fist ME [$N=185$, 25.17%])
- **Evaluation**: Leave-One-Subject-Out (LOSO) cross-validation
The manuscript explicitly clarifies that UI aliases (*Calm, Focused, Stressed, Fatigued, Excited*) are semantic simulation proxy labels and not ground-truth clinical emotion annotations.

**Point 5 (Verified Model Architecture & Actual Hyperparameters):**  
*Reviewer Comment:* Rewrite methodology architecture section and redraw diagram to match the results exactly with layer sizes, kernel sizes, heads, optimizer, lr, epochs, and batch size.  
*Author Response:* Section III-E, Fig. 1, and Fig. 2 provide the exact implemented specifications without invented parameters:
- **Residual 1D CNN**: 3 blocks ($19 \to 64 \to 128 \to 128$), kernel $k=3$, padding $p=1$, BatchNorm, GELU, 1D Dropout ($p=0.30$), $1 \times 1$ conv shortcuts.
- **BiLSTM Core**: 2 layers, hidden size $h=128$ per direction (256 concatenated output), dropout $p=0.30$.
- **Transformer Encoder**: 2 layers, 8 self-attention heads, $d_{\text{model}}=256, d_{\text{ff}}=512$, dropout $p=0.30$.
- **Total Trainable Parameters**: 731,340 parameters.
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $\lambda = 10^{-4}$, initial $\eta = 3 \times 10^{-4}$) with `ReduceLROnPlateau` scheduler (factor $= 0.5$, patience $= 5$ epochs), batch size 32, maximum 50 epochs with early stopping patience of 15 epochs.

**Point 6 (Formal Academic Language & Passive Voice):**  
*Reviewer Comment:* Replace "we", "our", and "I" with passive voice or neutral subjects.  
*Author Response:* The manuscript has been completely revised to eliminate all first-person pronouns ("we", "our", "I"). All descriptions use formal passive constructions and third-person neutral phrasing in compliance with IEEE guidelines.

---

### Reviewer 2 Comments

**Point 1 (Abbreviations Defined at First Occurrence):**  
*Reviewer Comment:* Define all abbreviations at first use.  
*Author Response:* All acronyms and abbreviations (EEG, BCI, CNN, LSTM, BiLSTM, PSD, FIR, SNR, EOG, EMG, ROC-AUC, LOSO, SVM, DWT, GAP, RKHS, ERD, ERS) are explicitly defined at their initial occurrence in both the Abstract and Section I.

**Point 2 (Replacement of Informal Figures with Publication-Grade Schematics):**  
*Reviewer Comment:* Replace informal / handwritten figure with publication-quality schematic.  
*Author Response:* Fig. 1 (System Pipeline), Fig. 2 (Detailed Neural Architecture), and `architecture_diagram.svg` have been redrawn with publication-ready vector-aligned typography, explicit tensor dimensionality annotations, and directional data-flow connections.

**Point 3 (IEEE Formatting Consistency):**  
*Reviewer Comment:* Standardize IEEE formatting, headings, captions, and alignments.  
*Author Response:* The entire document adheres to IEEE transactions conventions: Roman numeral section headings (I. Introduction $\to$ II. Related Work $\to$ III. Proposed Methodology $\to$ IV. Results and Discussion $\to$ V. Conclusion and Future Work $\to$ References), table titles centered strictly above tables (*TABLE I*, *TABLE II*, *TABLE III*, *TABLE IV*), and figure captions positioned strictly below figures (*Fig. 1.*, *Fig. 2.*).

**Point 4 (Removal of Informal Phrasing):**  
*Reviewer Comment:* Replace colloquial phrasing with formal academic English.  
*Author Response:* All informal wording has been replaced with precise technical vocabulary (e.g., *"removes the hardware step"* $\to$ *"eliminates the requirement for dedicated physical recording hardware"*; *"gets"* $\to$ *"achieves / obtains"*; *"a lot of"* $\to$ *"substantial"*).

**Point 5 (Grammar and Technical Proofreading):**  
*Reviewer Comment:* Proofread for grammar, syntax, and consistent mathematical notation.  
*Author Response:* Full copy-editing and mathematical verification was conducted. All metric formulas (multiclass accuracy, macro-averaged precision, recall, F1, and confusion matrices) are mathematically rigorous and consistent with the codebase.
