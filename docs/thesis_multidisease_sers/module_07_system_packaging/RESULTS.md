# Module 07: System Packaging, Reproducibility, and Handoff

## Executive Summary

Module 07 consolidates the project into a reproducible research workflow. The repository already contains the core pipeline stages from data preparation to clinical validation, so the packaging task here is to document the final architecture, define the run order, and record the reproducibility policy used in the thesis.

## Final Repository Structure

```text
KLTN/
├── Data/
├── docs/
│   └── thesis_multidisease_sers/
│       ├── module_00_scope/
│       ├── module_01_data_governance/
│       ├── module_02_signal_preprocessing/
│       ├── module_03_voltage_dynamics_2t2d/
│       ├── module_04_modeling_multiclass/
│       ├── module_05_explainable_ai/
│       ├── module_06_validation_clinical/
│       └── module_07_system_packaging/
├── outputs/
│   ├── data/
│   ├── figures/
│   └── reports/
├── scripts/
│   ├── module_01_prepare_data.py
│   ├── module_01_prepare_data_phase2.py
│   ├── module_02_batch_preprocess.py
│   ├── module_02_demo_preprocessing.py
│   ├── module_02_qc_decision_report.py
│   ├── module_02_qc_preprocessing.py
│   ├── module_04_compare_configs.py
│   ├── module_04_finalize_decision.py
│   ├── module_04_train_baselines.py
│   ├── module_04_tune_svm.py
│   ├── module_05_create_feature_table.py
│   ├── module_05_explain_svm.py
│   ├── module_06_validate_clinical.py
│   ├── generate_all_figures.py
│   └── visualize_dataset.py
└── src/
    ├── augmentation.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── utils.py
    └── visualization.py
```

## Reusable Pipeline Entry Points

The repository is already organized around executable module scripts. The recommended run order is:

1. `scripts/module_01_prepare_data.py`
2. `scripts/module_02_qc_preprocessing.py`
3. `scripts/module_02_qc_decision_report.py`
4. `scripts/module_04_train_baselines.py`
5. `scripts/module_04_compare_configs.py`
6. `scripts/module_04_tune_svm.py`
7. `scripts/module_04_finalize_decision.py`
8. `scripts/module_05_explain_svm.py`
9. `scripts/module_05_create_feature_table.py`
10. `scripts/module_06_validate_clinical.py`

## Configuration Strategy

The project is currently run through explicit command-line arguments and output directories. This is reproducible and easy to audit. If the project is extended further, the next step is to move the run parameters into a small YAML or JSON configuration file, but the current workflow is already stable enough for thesis delivery.

### Suggested JSON schema

```json
{
  "paths": {
    "data_dir": "outputs/data",
    "reports_dir": "outputs/reports",
    "figures_dir": "outputs/figures"
  },
  "module_04": {
    "winner_config": "with_aug",
    "svm_kernel": "rbf",
    "svm_c": 1.0,
    "svm_gamma": "scale",
    "class_weight": "balanced"
  },
  "module_05": {
    "background_samples": 20,
    "include_aug_train": true
  },
  "reproducibility": {
    "random_seed": 42,
    "numpy_seed": 42,
    "python_hash_seed": 42
  }
}
```

## Seed and Environment Policy

### Randomness control
- Use a fixed seed for any stochastic step.
- Recommended default seed: `42`.
- When multiple seeds are needed, report the seed list explicitly in the results.
- For explainability and validation, keep the background sample subset and resampling seed visible in the report.

### Environment control
- Python environment: `venv` under the project root.
- Execution path used during validation: `C:\Users\ADMIN\Desktop\Prj\KLTN\venv`
- Primary runtime stack:
  - Python
  - NumPy
  - pandas
  - scikit-learn
  - matplotlib
  - shap

### Reproducibility rules
- Reuse the same output directories when rerunning the same module unless comparing variants.
- Preserve CSV and markdown outputs for auditability.
- Keep the QC warning and cancer-support caveat in all downstream reports.

## Packaging Notes

The packaging layer does not introduce a new model. Instead, it standardizes how the existing modules are executed and documented:

- Data governance outputs are written once and then treated as the source of truth.
- Module 02 warning flags are propagated into modeling and final reporting.
- Module 04 model selection is explicitly recorded and traceable.
- Module 05 explainability is versioned for both `with_aug` and `no_aug`.
- Module 06 summarizes clinical reliability with conservative confidence intervals.

## Final Handoff State

### Stable artifacts already produced
- QC decision report
- Baseline and tuned modeling reports
- SHAP explainability reports for with_aug and no_aug
- Wavelength-mapped feature interpretation table
- Clinical validation report

### Remaining operational recommendation
- If the thesis is extended beyond documentation, the next engineering step should be to add a small pipeline config file and a single driver script that dispatches each module in sequence.

## Status

✅ Module 07 complete for thesis packaging and reproducibility documentation.
