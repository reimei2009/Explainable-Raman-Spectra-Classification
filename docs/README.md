# KLTN – Khóa Luận Tốt Nghiệp
## Raman Spectra Analysis / Phân tích Phổ Raman

> Electrochemical SERS dataset – analysis tools for Raman spectra recorded at
> different applied potentials from human tissue samples.

---

## Project Structure

```
KLTN/
├── Data/
│   └── different potential/      Raw data (24 sample folders)
│       ├── M78S/                 e.g. Male, ID 78, Solid
│       │   ├── -400_1.spc
│       │   ├── -400_2.spc
│       │   └── ...               ~7 replicates × 9 potentials
│       └── ...
│
├── src/                          Python source package
│   ├── __init__.py
│   ├── data_loader.py            Load .spc / .mat files
│   ├── preprocessing.py          Baseline, smoothing, normalisation
│   ├── utils.py                  Logging, paths, helpers
│   └── visualization.py          All plots (matplotlib / seaborn)
│
├── scripts/
│   └── sample_display.py         Runnable demo – prints summary + saves plots
│
├── docs/
│   ├── 01_data_description.md    Dataset description (EN + VI)
│   ├── 02_code_documentation.md  API reference with examples
│   ├── 03_workflow.md            Step-by-step analysis workflow
│   └── 04_troubleshooting.md     Common errors and fixes
│
├── outputs/
│   ├── figures/                  Generated PNG plots
│   ├── processed/                Preprocessed spectra (NPZ / CSV)
│   └── reports/                  Summary tables / reports
│
├── logs/                         Processing logs
├── requirements.txt              Python dependencies
├── setup.bat                     Windows one-click setup
└── README.md                     This file
```

---

## Quick Start

### 1. Setup (Windows Command Prompt)

```batch
cd C:\Users\ADMIN\Desktop\Prj\KLTN
setup.bat
```

> If `setup.bat` fails because `pwsh.exe` is not available, see
> `docs/04_troubleshooting.md § 6`.

### 2. Run the demo script

```batch
venv\Scripts\activate.bat
python scripts\sample_display.py
```

This prints a summary table and saves two figures to `outputs/figures/`.

### 3. Jupyter Notebook

```batch
venv\Scripts\activate.bat
jupyter notebook
```

---

## Data Overview

| Field | Value |
|---|---|
| Number of samples | 24 |
| Female / Male | 14 F / 10 M |
| Sample types | S (Solid), P (Pellet), C (Cell), GA |
| Potentials | -400, -300, -200, -100, 0, +100, +200, +300, +400 mV |
| Replicates per potential | 1–7 |
| File format | `.spc` (Thermo Galactic), `.mat` (MATLAB) |

---

## Module Quick Reference

| Module | Key Class / Functions | One-liner |
|---|---|---|
| `src/data_loader.py` | `RamanDataLoader` | Load `.spc` and `.mat` files |
| `src/preprocessing.py` | `RamanPreprocessor` | Baseline, smooth, normalise |
| `src/visualization.py` | `RamanVisualizer` | Plot spectra, heatmaps |
| `src/utils.py` | various functions | Logging, paths, dir helpers |

---

## Dependencies

Install via `pip install -r requirements.txt` (done automatically by `setup.bat`):

- `numpy`, `scipy`, `pandas` – numerical core
- `matplotlib`, `seaborn` – visualisation
- `spc-spectra` – parse `.spc` binary files
- `scikit-learn` – machine-learning utilities
- `h5py` – MATLAB v7.3 HDF5 file support
- `jupyter`, `notebook`, `ipykernel` – interactive notebooks

---

## Documentation

| File | Contents |
|---|---|
| `docs/01_data_description.md` | Dataset structure, naming conventions, file formats |
| `docs/02_code_documentation.md` | Full API reference with code examples |
| `docs/03_workflow.md` | End-to-end analysis pipeline walkthrough |
| `docs/04_troubleshooting.md` | Common errors and solutions |

---

## Thesis Roadmap (New)

This project now includes a dedicated thesis roadmap for the selected large-scale topic:

- Multi-disease GB diagnosis from voltage-applied SERS + extended 2T2D + explainable ML

Entry point:

- `docs/thesis_multidisease_sers/README.md`

Module documents:

- `docs/thesis_multidisease_sers/module_00_scope/README.md`
- `docs/thesis_multidisease_sers/module_01_data_governance/README.md`
- `docs/thesis_multidisease_sers/module_02_signal_preprocessing/README.md`
- `docs/thesis_multidisease_sers/module_03_voltage_dynamics_2t2d/README.md`
- `docs/thesis_multidisease_sers/module_04_modeling_multiclass/README.md`
- `docs/thesis_multidisease_sers/module_05_explainable_ai/README.md`
- `docs/thesis_multidisease_sers/module_06_validation_clinical/README.md`
- `docs/thesis_multidisease_sers/module_07_system_packaging/README.md`
- `docs/thesis_multidisease_sers/module_08_execution_log/README.md`

---


