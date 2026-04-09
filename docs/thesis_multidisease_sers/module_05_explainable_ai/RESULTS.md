# Module 05: Explainable AI – SVM Model Interpretation

## Executive Summary

Module 05 applies **SHAP (SHapley Additive exPlanations)** to interpret the winning SVM model from Module 04's baseline modeling. SHAP values quantify how much each spectral feature (Raman shift) contributes to the model's predictions for each disease class.

**Critical Data Caveat:** Test set has **0 cancer samples** (support=0). Cancer-class interpretations cannot be validated on holdout test data. All cancer feature rankings below are exploratory only.

---

## Methodology

### SHAP Framework
- **Explainer Type**: KernelExplainer (model-agnostic, works with any black-box model)
- **Background Sample Set**: 20 training samples (subset of 88 with_aug or 22 no_aug) for computational efficiency
- **Test Set**: 7 samples (5 stone, 2 polyp, 0 cancer)

### Feature Index Mapping
- 2048 spectral features sampled evenly across 0–3200 cm⁻¹ Raman shift range
- Feature index → Wavenumber: `wavenumber ≈ (idx / 2048) × 3200` cm⁻¹
- Typical regions:
  - **< 300 cm⁻¹**: Lattice / substrate modes
  - **300–800 cm⁻¹**: Fingerprint (S-S stretching, ring breathing)
  - **800–1200 cm⁻¹**: C-C stretching (protein, lipid)
  - **1200–1700 cm⁻¹**: Amide I/II, C=C stretching
  - **> 1700 cm⁻¹**: C-H / O-H stretching

---

## Results: with_aug Configuration (Winning Model)

### Model Details
- **Algorithm**: SVM with RBF kernel
- **Hyperparameters**: C=1.0, gamma=scale
- **Training Set**: 88 samples (22 original + 66 augmented)
- **Validation Set**: 7 samples
- **Test Set**: 7 samples (5 stone, 2 polyp)

### Top 5 Most Discriminative Features by Class

#### Stone (Class 0)
| Rank | Feature Index | Wavenumber Region | Mean |SHAP| |
|------|---------------|-------------------|---------|
| 1    | 25            | ~39 cm⁻¹ (Low freq)| 0.0138  |
| 2    | 26            | ~41 cm⁻¹ (Low freq)| 0.0136  |
| 3    | 28            | ~44 cm⁻¹ (Low freq)| 0.0128  |
| 4    | 48            | ~75 cm⁻¹ (Low freq)| 0.0127  |
| 5    | 27            | ~42 cm⁻¹ (Low freq)| 0.0123  |

**Interpretation**: Stone class is discriminated by **low-frequency lattice / substrate modes** (~39–75 cm⁻¹). These likely reflect crystalline structure differences in cholesterol stones vs. soft tissue (polyp).

#### Polyp (Class 1)
| Rank | Feature Index | Wavenumber Region | Mean |SHAP| |
|------|---------------|-------------------|---------|
| 1    | 1360          | ~2100 cm⁻¹ (High) | 0.0148  |
| 2    | 210           | ~329 cm⁻¹ (Finger)| 0.0118  |
| 3    | 749           | ~1171 cm⁻¹ (C-C)  | 0.0104  |
| 4    | 724           | ~1132 cm⁻¹ (C-C)  | 0.0103  |
| 5    | 25            | ~39 cm⁻¹ (Low)    | 0.0100  |

**Interpretation**: Polyp class is strongly discriminated by **Feature 1360 (~2100 cm⁻¹)**, a high-wavenumber region often attributed to aromatic C-H or C≡ stretching. Secondary features include mid-frequency (~330, 1130–1170 cm⁻¹) likely representing protein / polysaccharide components of adenomatous tissue.

#### Cancer (Class 2)
| Rank | Feature Index | Wavenumber Region | Mean |SHAP| |
|------|---------------|-------------------|---------|
| 1    | 5             | ~8 cm⁻¹ (Low)     | [validation blocked] |
| 2    | … | … | … |

**⚠️ CRITICAL**: Test split contains 0 cancer samples. Feature rankings for cancer class are **not validated** on holdout data and must be treated as exploratory hypotheses only.

---

## Results: no_aug Configuration (Baseline)

### Model Details
- **Algorithm**: SVM with RBF kernel
- **Hyperparameters**: C=1.0, gamma=0.001
- **Training Set**: 22 samples (original only, no augmentation)
- **Test Set**: 7 samples (5 stone, 2 polyp, 0 cancer)

### Top 5 Most Discriminative Features (no_aug)

#### Stone (Class 0)
| Rank | Feature Index | Wavenumber Region | Mean |SHAP| |
|------|---------------|-------------------|---------|
| 1    | 25            | ~39 cm⁻¹ (Low freq)| 0.0142  |
| 2    | 26            | ~41 cm⁻¹ (Low freq)| 0.0140  |
| 3    | 28            | ~44 cm⁻¹ (Low freq)| 0.0124  |
| 4    | 48            | ~75 cm⁻¹ (Low freq)| 0.0119  |
| 5    | 27            | ~42 cm⁻¹ (Low freq)| 0.0107  |

**Consistency Check**: Feature ranking **identical** to with_aug. Robustness ✓

#### Polyp (Class 1)
| Rank | Feature Index | Wavenumber Region | Mean |SHAP| |
|------|---------------|-------------------|---------|
| 1    | 1360          | ~2100 cm⁻¹ (High) | 0.0156  |
| 2    | 210           | ~329 cm⁻¹ (Finger)| 0.0121  |
| 3    | 749           | ~1171 cm⁻¹ (C-C)  | 0.0105  |
| 4    | 724           | ~1132 cm⁻¹ (C-C)  | 0.0102  |
| 5    | 25            | ~39 cm⁻¹ (Low)    | 0.0098  |

**Consistency Check**: Feature ranking **identical** to with_aug. Robustness significantly confirmed ✓

---

## Comparison: with_aug vs. no_aug

| Dimension | with_aug | no_aug | Delta |
|-----------|----------|--------|-------|
| Train samples | 88 | 22 | +66 augmented |
| Top stone feature | Idx 25 (~39 cm⁻¹) | Idx 25 (~39 cm⁻¹) | **Same** |
| Top polyp feature | Idx 1360 (~2100 cm⁻¹) | Idx 1360 (~2100 cm⁻¹) | **Same** |
| Stone SHAP range | 0.0123–0.0138 | 0.0107–0.0142 | Similar |
| Polyp SHAP range | 0.0100–0.0148 | 0.0098–0.0156 | Similar |

**Key Finding**: Feature-level rankings are **highly robust** across both augmentation conditions. Augmentation improves model calibration (Module 04) without substantially shifting which spectral regions are diagnostic.

---

## Clinical Interpretation & Recommendations

### Stone Differentiation
- **Mechanism**: Low-frequency lattice modes (39–75 cm⁻¹) reflect crystalline structure.
- **Clinical Relevance**: Cholesterol stones have distinct crystal morphology vs. soft tissue.
- **Recommendation**: Feature 25–28 could become targets for handheld SERS device design optimized for low-frequency sensitivity.

### Polyp Detection
- **Mechanism**: High-wavenumber feature (∼2100 cm⁻¹) + mid-frequency proteomic signature.
- **Clinical Relevance**: Adenomatous polyp tissue shows distinct aromatic / protein fingerprint vs. stone.
- **Recommendation**: Feature 1360 warrants spectroscopic validation using reference tissue library; investigate atomic/molecular assignment.

### Cancer Detection
- **Mechanism**: Cannot be determined; test set lacks cancer samples.
- **Clinical Recommendation**:
  1. **Collect additional test set** with ≥5 cancer samples (ideally 10+) before clinical deployment.
  2. **Re-run Module 05** after balanced data collection.
  3. Until validated, cancer predictions should be treated as **model-learned patterns** without clinical basis.

---

## Visualization Outputs

### Summary Plots (per class)
- **shap_summary_bar_*.png**: Top 20 features by average impact (bar chart)
- **shap_summary_beeswarm_*.png**: Full SHAP value distribution (beeswarm + density)

### Per-Sample Force Plots
- **shap_force_sample*.png**: Feature contributions for each test sample
  - Red bars: features pushing prediction toward that class
  - Blue bars: features pushing away
  - Base value: average prediction across training set

All plots saved to:
- `outputs/reports/module_05/` (with_aug)
- `outputs/reports/module_05_no_aug/` (no_aug)

---

## Limitations & Future Work

1. **Test Set Data Imbalance**: Cancer class has 0 samples. Highly recommend stratified oversampling or focused biobanking to collect cancer-positive samples.
2. **Feature Assignment Uncertainty**: ~2100 cm⁻¹ region assignment (C≡? aromatic C-H?) requires orthogonal spectroscopic validation.
3. **Confounding by Augmentation**: While top features remain stable, subtle SHAP magnitudes differ. Multi-seed stability analysis recommended.
4. **Clinical Validation**: SHAP values are model-intrinsic. External validation on independent cohort essential before clinical deployment.

---

## Files Generated

- [outputs/reports/module_05/explain_report.md](../../../../outputs/reports/module_05/explain_report.md) – Main SHAP analysis report (with_aug)
- [outputs/reports/module_05/feature_importance.csv](../../../../outputs/reports/module_05/feature_importance.csv) – Top 5 features per class
- [outputs/reports/module_05/feature_interpretation_table.md](../../../../outputs/reports/module_05/feature_interpretation_table.md) – Wavelength-mapped interpretation
- outputs/reports/module_05/shap_summary_bar_*.png – Feature importance (bar)
- outputs/reports/module_05/shap_summary_beeswarm_*.png – SHAP distributions
- outputs/reports/module_05/shap_force_sample*.png – Per-sample force plots (21 plots: 7 samples × 3 classes)
- [outputs/reports/module_05_no_aug/explain_report.md](../../../../outputs/reports/module_05_no_aug/explain_report.md) – SHAP analysis (no_aug baseline)
- outputs/reports/module_05_no_aug/* – Corresponding plots for no_aug configuration

---

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4765–4774).
- Molnar, C. (2020). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.).
- SHAP Documentation: https://shap.readthedocs.io/ 

---

## Module Status

✅ **COMPLETE**

- [x] Load winning SVM model from Module 04 (with_aug)
- [x] Compute SHAP features values for test set
- [x] Generate summary plots per class
- [x] Generate per-sample force plots
- [x] Document data caveat (cancer support = 0)
- [x] Run baseline comparison (no_aug)
- [x] Validate feature robustness across configurations

**Next Module**: Module 06 – Validation & Clinical Platform
