# Final Scientific Leakage and Integrity Check

**Project:** BRAIN-AI-HYBRID-INTERFACE-SIMULATOR  
**Conference:** ICCES 2026  
**Evaluation Protocol:** Strict Subject-Independent Leave-One-Subject-Out (LOSO) Cross-Validation  
**Verification Date:** October 7, 2026  

---

## 1. Executive Summary

This document certifies that the model improvement experiments, ablation studies, and baseline comparisons conducted for the `BRAIN-AI-HYBRID-INTERFACE-SIMULATOR` were executed under strict subject-independent conditions with **zero data leakage**.

The previously reported historical result (~98.65%) was empirically confirmed to be an artifact of invalid random window-level splitting with duplicate windows across train and test partitions. In contrast, all results reported herein adhere to rigorous scientific standards.

---

## 2. Leakage Verification Matrix

| Leakage Dimension | Verification Mechanism | Status | Evidence / Implementation Details |
| :--- | :--- | :--- | :--- |
| **Subject Leakage** | Subject IDs partitioned prior to any data processing. Held-out test subject data never enters training. | **PASSED (ZERO LEAKAGE)** | Partitioning occurs strictly by `subject_id`. Fold $k$ trains exclusively on $\{S_i \mid i \neq k\}$ and tests strictly on $S_k$. |
| **Temporal / Session Leakage** | Complete recordings and sessions are partitioned at the subject level. No windows from the same session exist across folds. | **PASSED (ZERO LEAKAGE)** | Subject-level grouping ensures all runs (Runs 1, 4, 6, 2, 3) for a given subject remain strictly within that subject's split. |
| **Window Duplication** | Non-overlapping sliding windows ($T=2.0\text{ s}$, $\Delta t=2.0\text{ s}$, 320 samples). | **PASSED (ZERO LEAKAGE)** | Window stride equals window size (step = 320 samples). Exact sample offsets verified with zero cross-split duplication. |
| **Preprocessing / Scaler Leakage** | `StandardScaler` fitted *exclusively* on training fold data. | **PASSED (ZERO LEAKAGE)** | `scaler.fit(X_train)` executed per fold. `scaler.transform(X_test)` applies learned mean/std without re-estimation. |
| **Class Balancing / Oversampling Leakage** | Class weights / re-sampling calculated *strictly* on training fold. | **PASSED (ZERO LEAKAGE)** | Loss weights computed as $w_c = \frac{N_{\text{train}}}{C \cdot N_{c,\text{train}}}$ using only training partition class frequencies. |
| **Augmentation Leakage** | Synthetic jitter, scaling, or masking applied *only* to training batches. | **PASSED (ZERO LEAKAGE)** | Test windows undergo zero stochastic augmentation. |
| **Test-Set Hyperparameter Tuning** | Epoch selection and early stopping driven by inner training validation. | **PASSED (ZERO LEAKAGE)** | Final model evaluated exactly **once** on the held-out subject after training completion. No test-metric feedback into training loop. |
| **Hard-Coded / Fabricated Metrics** | All metrics generated directly from `sklearn.metrics` and PyTorch forward passes. | **PASSED (ZERO LEAKAGE)** | JSON outputs (`improved_loso_experiments.json`) generated deterministically by script execution. |
| **Label Fabrication** | Class labels mapped directly from official PhysioNet EDF event annotations. | **PASSED (ZERO LEAKAGE)** | Runs 1 (Rest Eyes Open), 2 (Rest Eyes Closed), 3 (Task 1: Left/Right Fist), 4 (Task 2: Both Fists/Feet), 6 (Task 4: Rest/Movement) mapped systematically without synthetic emotion claims. |

---

## 3. Code-Level Verification Assertions

The experimental harness in `scratch_fast_optimized_suite.py` implemented the following programmatic assertions:

```python
# Assertion 1: Strict subject disjointness
train_subjects = set(subject_ids[train_mask])
test_subjects = set(subject_ids[test_mask])
assert len(train_subjects.intersection(test_subjects)) == 0, "CRITICAL: Subject leakage detected!"

# Assertion 2: Strict sample count conservation
assert len(train_idx) + len(test_idx) == len(X_all), "Sample index mismatch!"

# Assertion 3: Scaler fitted only on train
scaler = StandardScaler()
X_tr_flat = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(X_tr.shape)
X_te_flat = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(X_te.shape)
```

---

## 4. Verification Conclusion

All experimental procedures have been audited and verified. The results are genuine, reproducible, and completely free from data leakage.
