# Visualization Scripts

## Available Scripts

### 1. `generate_all_figures.py` (Simple)
Generate waterfall plots for all 38 samples in deduplicated dataset.

```bash
python scripts/generate_all_figures.py
```

**Output:**
- `outputs/figures/per_sample/<sample>_waterfall.png` (38 files)
- Skips existing figures, only generates missing ones

**Features:**
- ✓ Processes 38 deduplicated samples
- ✓ Shows all potentials (-400 to +400 mV) for each sample
- ✓ Creates one waterfall plot per sample
- ✓ Progress tracking with counter

---

### 2. `visualize_dataset.py` (Comprehensive)
Complete dataset visualization package.

```bash
python scripts/visualize_dataset.py
```

**Output:**
- `outputs/figures/disease_distribution.png` – Pie chart of disease breakdown
- `outputs/figures/disease_age_scatter.png` – Scatter: Disease vs Age vs Spectra count
- `outputs/figures/gender_age_distribution.png` – Gender pie + Age histogram
- `outputs/figures/per_sample/<sample>_waterfall.png` (38 files)
- Console summary table

**Features:**
- ✓ Disease distribution pie chart
- ✓ Gender & age analysis
- ✓ Disease vs age scatter plot
- ✓ Per-sample waterfall plots
- ✓ Summary statistics table
- ✓ Progress logging

---

### 3. `sample_display.py` (Original Demo)
Quick 3×3 grid overview of first 3 samples.

```bash
python scripts/sample_display.py
```

**Output:**
- `outputs/figures/sample_overview.png` – 3×3 grid
- `outputs/figures/potential_series.png` – Waterfall of 1st sample

---

## Quick Start

**Option A: Generate only missing waterfall plots (fast)**
```bash
python scripts/generate_all_figures.py
```

**Option B: Complete analysis (comprehensive)**
```bash
python scripts/visualize_dataset.py
```

**Option C: Original demo**
```bash
python scripts/sample_display.py
```

---

## Output Structure

```
outputs/figures/
├── sample_overview.png                     (original)
├── potential_series.png                    (original)
├── disease_distribution.png                (new)
├── disease_age_scatter.png                 (new)
├── gender_age_distribution.png             (new)
└── per_sample/
    ├── F35S_waterfall.png
    ├── f41p_waterfall.png
    ├── F42S_waterfall.png
    ├── f45c_waterfall.png
    ├── F45S_waterfall.png
    ├── F49GA_waterfall.png
    ├── F50P_waterfall.png
    ├── F51S_waterfall.png
    ├── F59C_waterfall.png
    ├── F60S_waterfall.png
    ├── F64SP_waterfall.png
    ├── F64SS_waterfall.png
    ├── f65p_waterfall.png
    ├── f70p_waterfall.png
    ├── m27s_waterfall.png
    ├── M29P_waterfall.png
    ├── m39s_waterfall.png
    ├── m40p_waterfall.png
    ├── M41P_waterfall.png
    ├── M48S_waterfall.png
    ├── M56S_waterfall.png
    ├── m66c_waterfall.png
    ├── M68P_waterfall.png
    ├── M78S_waterfall.png
    ├── F43S_waterfall.png        ← from Stone
    ├── F43SK_waterfall.png       ← from Stone
    ├── F46S_waterfall.png        ← from Stone
    ├── F61S_waterfall.png        ← from Stone
    ├── F63S_waterfall.png        ← from Stone
    ├── F80SK_waterfall.png       ← from Stone
    ├── F84S_waterfall.png        ← from Stone
    ├── M47S_waterfall.png        ← from Stone
    ├── M52S_waterfall.png        ← from Stone
    ├── M55S_waterfall.png        ← from Stone
    ├── M58S_waterfall.png        ← from Stone
    ├── M62S_waterfall.png        ← from Stone
    ├── M68S_waterfall.png        ← from Stone
    ├── M75S_waterfall.png        ← from Stone
    └── ... (38 total)
```

---

## Notes

- Scripts automatically skip existing figures to avoid duplication
- All plots use consistent styling (matplotlib + seaborn)
- Progress is shown in console with [current/total] counter
- No dependencies beyond project requirements (matplotlib, pandas, numpy)
