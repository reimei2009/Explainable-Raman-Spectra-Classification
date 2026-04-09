# 03. Analysis Workflow

## Step 1 – Environment Setup

```batch
# Windows Command Prompt (NOT PowerShell if pwsh.exe is unavailable)
cd C:\Users\ADMIN\Desktop\Prj\KLTN
setup.bat
```

This will:
1. Create `venv\` virtual environment
2. Install all packages from `requirements.txt`
3. Create `outputs\figures\`, `outputs\processed\`, `outputs\reports\`, `logs\`

To activate the environment in future sessions:
```batch
venv\Scripts\activate.bat
```

---

## Step 2 – Load Data

```python
from src.data_loader import RamanDataLoader

loader = RamanDataLoader(verbose=True)

# Load a single file
sp = loader.load_spc("Data/different potential/M78S/-200_1.spc")
wn  = sp["wavenumber"]   # Raman shift (cm⁻¹)
it  = sp["intensity"]    # intensity counts

# Load all replicates for one sample
sample_data = loader.load_sample_folder("Data/different potential/M78S")
# → dict keyed by e.g. "-400_1", "-200_3", "0_1", …

# Load the entire dataset
all_data = loader.load_all_samples("Data/different potential")
# → dict keyed by sample name (e.g. "M78S"), values are sample_data dicts

# Get a summary DataFrame
df = loader.get_sample_summary("Data/different potential")
print(df)
```

---

## Step 3 – Preprocess

```python
from src.preprocessing import RamanPreprocessor

pp = RamanPreprocessor()

raw = sample_data["0_1"]["intensity"]

# Recommended pipeline:
clean    = pp.remove_cosmic_rays(raw, threshold=3.0)
corrected = pp.baseline_correction_als(clean, lam=1e5, p=0.01)
smoothed  = pp.smooth_savgol(corrected, window=11, polyorder=3)
normed    = pp.normalize_minmax(smoothed)

# Average replicates at a given potential
reps = [
    sample_data[k]["intensity"]
    for k in sample_data if k.startswith("0_")
]
avg = pp.average_replicates(reps)
```

---

## Step 4 – Visualise

```python
from src.visualization import RamanVisualizer

viz = RamanVisualizer(output_dir="outputs/figures")

# Single spectrum
fig = viz.plot_spectrum(wn, normed, title="M78S @ 0 mV (preprocessed)")
viz.save_figure(fig, "m78s_0mv.png")

# Multiple overlaid spectra
spectra = {
    f"{pot} mV": sample_data[f"{pot}_1"]
    for pot in [-200, 0, 200]
    if f"{pot}_1" in sample_data
}
fig = viz.plot_multiple_spectra(spectra, title="M78S – potential comparison")
viz.save_figure(fig, "m78s_comparison.png")

# Waterfall plot (all potentials, with vertical offset)
fig = viz.plot_potential_series(sample_data, sample_name="M78S")
viz.save_figure(fig, "m78s_waterfall.png")

# 2-D heatmap
fig = viz.plot_heatmap(sample_data, sample_name="M78S")
viz.save_figure(fig, "m78s_heatmap.png")
```

---

## Step 5 – Quick Demo Script

```batch
# Activate environment first
venv\Scripts\activate.bat

python scripts\sample_display.py
```

This script:
1. Finds the data root automatically
2. Loads 3 sample folders
3. Prints a summary table in the terminal
4. Saves `outputs/figures/sample_overview.png` (3×3 grid)
5. Saves `outputs/figures/potential_series.png` (waterfall)

---

## Step 6 – Jupyter Notebook (Interactive)

```batch
venv\Scripts\activate.bat
jupyter notebook
```

Open `http://localhost:8888` in your browser and create a new notebook.

Suggested notebook cells:

```python
# Cell 1 – Setup
import sys
sys.path.insert(0, "..")   # if notebook is in a subdirectory
from src.utils import setup_logging
setup_logging()
```

```python
# Cell 2 – Load
from src.data_loader import RamanDataLoader
loader = RamanDataLoader(verbose=True)
data = loader.load_all_samples("Data/different potential")
```

```python
# Cell 3 – Preprocess
from src.preprocessing import RamanPreprocessor
pp = RamanPreprocessor()
sample = data["M78S"]
wn = sample["0_1"]["wavenumber"]
avg = pp.average_replicates([
    sample[k]["intensity"] for k in sample if k.startswith("0_")
])
corrected = pp.baseline_correction_als(avg)
normed = pp.normalize_minmax(corrected)
```

```python
# Cell 4 – Plot
import matplotlib.pyplot as plt
%matplotlib inline
from src.visualization import RamanVisualizer
viz = RamanVisualizer()
fig = viz.plot_spectrum(wn, normed, title="M78S @ 0 mV – ALS baseline + min-max")
plt.show()
```

---

## Full Analysis Pipeline (Summary)

```
Raw .spc files
     │
     ▼
RamanDataLoader.load_sample_folder()
     │
     ▼
RamanPreprocessor.remove_cosmic_rays()
     │
RamanPreprocessor.baseline_correction_als()
     │
RamanPreprocessor.smooth_savgol()
     │
RamanPreprocessor.average_replicates()
     │
RamanPreprocessor.normalize_minmax()
     │
     ▼
RamanVisualizer.plot_potential_series()
RamanVisualizer.plot_heatmap()
RamanVisualizer.save_figure()
     │
     ▼
outputs/figures/*.png
```
