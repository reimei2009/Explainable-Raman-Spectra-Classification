# Module 06: Validation & Clinical Reliability

## Executive Summary

Module 06 compiles the clinical validation evidence from Modules 02, 04, and 05. The result is intentionally conservative: the validation split is small, the test split is tiny, and the test set contains **0 cancer samples**, so cancer-specific clinical validation cannot be claimed yet.

## Key Findings

- QC preprocessing raised a validation warning in both baseline and augmented preprocessing.
- The winning Module 04 configuration remains **with_aug / svm_rbf_C1.0_gscale**.
- Test performance is acceptable for stone/polyp discrimination, but the confidence intervals are wide because $n=7$.
- Module 05 showed stable SHAP feature rankings across with_aug and no_aug, which supports feature-level robustness.

## Quantitative Validation Summary

| Metric | Point estimate | 95% CI | Notes |
|---|---:|---:|---|
| Accuracy | 0.8571 | [0.4869, 0.9743] | Wilson interval from 6/7 correct |
| Balanced accuracy | 0.9000 | [0.3590, 0.9819] | Present classes only; cancer support is 0 |
| Stone recall | 0.8000 | [0.3755, 0.9638] | 4/5 correct |
| Polyp recall | 1.0000 | [0.3424, 1.0000] | 2/2 correct |
| Cancer recall | N/A | N/A | No test samples available |

## Subgroup Representation in Test Set

| Subgroup | Distribution |
|---|---|
| Gender | M=4, F=3 |
| Sample type | S=5, P=2 |

## Clinical Interpretation

- The system is currently more defensible as a **research prototype** than a clinical decision tool.
- Stone and polyp discrimination is supported by both performance metrics and SHAP interpretability.
- Cancer detection remains **unvalidated** because the test split has no cancer samples.

## Robustness Notes

- Module 05 top SHAP features were identical across with_aug and no_aug for stone and polyp.
- Module 02 QC shows weak validation separation, so downstream conclusions should be treated cautiously.
- Further validation should include repeated resampling, noise stress tests, and a revised split that guarantees cancer examples in holdout evaluation.

## Recommendation

Before any clinical deployment claim, the pipeline should be extended with:

1. A revised split or independent cohort that includes cancer samples in test.
2. Repeated resampling or grouped CV to reduce variance from tiny sample sizes.
3. External validation on a separate cohort.

## Linked Evidence

- [Module 02 QC decision report](../../../../outputs/reports/module_02_qc_decision_report.md)
- [Module 04 final decision](../../../../outputs/reports/module_04/final_decision.md)
- [Module 05 explainability](../../../../outputs/reports/module_05/explain_report.md)
- [Module 05 no_aug explainability](../../../../outputs/reports/module_05_no_aug/explain_report.md)

## Status

✅ Complete
