# 02. Code Documentation

## Module Overview

| Module | Class / Functions | Purpose |
|---|---|---|
| `src/utils.py` | Functions | Logging, paths, directory helpers |
| `src/data_loader.py` | `RamanDataLoader` | Load `.spc` and `.mat` files |
| `src/preprocessing.py` | `RamanPreprocessor` | Baseline, smoothing, normalisation |
| `src/visualization.py` | `RamanVisualizer` | All plotting functions |
| `scripts/sample_display.py` | `main()` | End-to-end demo script |

---

## `src/utils.py`

### `get_project_root() -> Path`
Returns the absolute `Path` to the project root (two levels above `src/`).

```python
from src.utils import get_project_root
root = get_project_root()
print(root)  # …/KLTN
```

### `ensure_dir(path) -> Path`
Creates a directory (and all parents) if it does not exist.  Returns the `Path`.

```python
from src.utils import ensure_dir
ensure_dir("outputs/figures")
```

### `setup_logging(log_file="outputs/processing.log") -> logging.Logger`
Configures the root logger with a file handler (DEBUG) and a stream handler (INFO).

```python
from src.utils import setup_logging
logger = setup_logging()
```

### `list_samples(data_root) -> list[str]`
Returns a sorted list of sample folder names matching `[FfMm]<digits><suffix>`.

```python
from src.utils import list_samples
samples = list_samples("Data/different potential")
# → ['F35S', 'f41p', 'F42S', …]
```

### `get_potentials(sample_folder) -> list[int]`
Scans `.spc` filenames and returns sorted unique potential values in mV.

```python
from src.utils import get_potentials
pots = get_potentials("Data/different potential/M78S")
# → [-400, -300, -200, -100, 0, 100, 200, 300, 400]
```

---

## `src/data_loader.py`

### Class `RamanDataLoader`

```python
from src.data_loader import RamanDataLoader
loader = RamanDataLoader(verbose=True)
```

#### `load_spc(filepath) -> dict`
Loads a single `.spc` file.

**Returns** `dict` with keys:
- `"wavenumber"` – `np.ndarray` of Raman shift (cm⁻¹)
- `"intensity"`  – `np.ndarray` of intensity
- `"filepath"`, `"filename"` – metadata strings

```python
sp = loader.load_spc("Data/different potential/M78S/0_1.spc")
print(sp["wavenumber"].shape)   # e.g. (1024,)
print(sp["intensity"].min())
```

#### `load_mat(filepath) -> dict`
Loads a MATLAB `.mat` file (v5 via scipy, v7.3 via h5py).

```python
data = loader.load_mat("Data/different potential/M78S/M78S.mat")
print(list(data.keys()))
```

#### `load_sample_folder(folder_path) -> dict`
Loads all `.spc` files in a folder.  Keys are `"{potential}_{replicate}"`.

```python
sample = loader.load_sample_folder("Data/different potential/M78S")
print(list(sample.keys())[:5])
# → ['-400_1', '-400_2', '-300_1', '-300_2', '-200_1']
```

#### `load_all_samples(data_root) -> dict`
Loads every sample folder under `data_root`.

```python
all_data = loader.load_all_samples("Data/different potential")
for sample_name, sample_data in all_data.items():
    print(f"{sample_name}: {len(sample_data)} spectra")
```

#### `parse_filename(filename) -> (int|None, int|None)`
Parses `"<potential>_<replicate>.spc"` → `(potential_mV, replicate_num)`.

```python
RamanDataLoader.parse_filename("-200_3.spc")  # → (-200, 3)
RamanDataLoader.parse_filename("bad.spc")     # → (None, None)
```

#### `get_sample_summary(data_root) -> pd.DataFrame`
Returns a summary DataFrame with columns: `sample`, `sex`, `subject_id`,
`suffix`, `n_spc_files`, `potentials`, `n_potentials`, `wn_min`, `wn_max`,
`n_points`.

```python
df = loader.get_sample_summary("Data/different potential")
print(df.to_string())
```

---

## `src/preprocessing.py`

### Class `RamanPreprocessor`

All methods are **static** – no instantiation state is needed, but you can
create an instance for convenience:

```python
from src.preprocessing import RamanPreprocessor
pp = RamanPreprocessor()
```

#### `normalize_minmax(spectrum) -> np.ndarray`
Scales intensity to [0, 1].

```python
normed = pp.normalize_minmax(raw_intensity)
```

#### `normalize_area(spectrum) -> np.ndarray`
Divides by the trapezoidal integral.

```python
normed = pp.normalize_area(raw_intensity)
```

#### `baseline_correction_als(spectrum, lam=1e5, p=0.01, niter=10) -> np.ndarray`
Asymmetric Least Squares baseline correction.

```python
corrected = pp.baseline_correction_als(raw_intensity, lam=1e6, p=0.005)
```

#### `smooth_savgol(spectrum, window=11, polyorder=3) -> np.ndarray`
Savitzky-Golay smoothing.

```python
smoothed = pp.smooth_savgol(corrected, window=15, polyorder=4)
```

#### `average_replicates(spectra_list) -> np.ndarray`
Element-wise mean of a list of spectra.

```python
reps = [sample[f"0_{i}"]["intensity"] for i in range(1, 4) if f"0_{i}" in sample]
avg = pp.average_replicates(reps)
```

#### `remove_cosmic_rays(spectrum, threshold=3.0) -> np.ndarray`
Replaces Z-score outliers with linearly interpolated values.

```python
clean = pp.remove_cosmic_rays(raw_intensity, threshold=3.5)
```

---

## `src/visualization.py`

### Class `RamanVisualizer`

```python
from src.visualization import RamanVisualizer
viz = RamanVisualizer(output_dir="outputs/figures")
```

#### `plot_spectrum(wavenumber, intensity, title, label, color, ax) -> Figure`
Single-spectrum line plot.

```python
fig = viz.plot_spectrum(sp["wavenumber"], sp["intensity"],
                        title="M78S @ 0 mV", label="replicate 1")
```

#### `plot_multiple_spectra(spectra_dict, title, figsize) -> Figure`
Overlays several spectra.  `spectra_dict` keys become legend labels.

```python
fig = viz.plot_multiple_spectra(
    {"0 mV": sp0, "-200 mV": sp_neg200},
    title="M78S – potential comparison"
)
```

#### `plot_potential_series(sample_data, sample_name, offset_factor, figsize) -> Figure`
Waterfall (stacked) plot across all potentials.

```python
fig = viz.plot_potential_series(sample_data, sample_name="M78S")
```

#### `plot_heatmap(sample_data, sample_name, figsize) -> Figure`
2-D colour map: potential (y) vs wavenumber (x), intensity as colour.

```python
fig = viz.plot_heatmap(sample_data, sample_name="M78S")
```

#### `save_figure(fig, filename, output_dir, dpi) -> Path`
Saves the figure and returns the saved path.

```python
path = viz.save_figure(fig, "m78s_heatmap.png")
print(path)
```

---

## Dependency Notes

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Numerical arrays |
| `scipy` | ≥1.10 | Signal processing, sparse solvers, `.mat` loader |
| `matplotlib` | ≥3.7 | All plotting |
| `seaborn` | ≥0.12 | Plot styling |
| `pandas` | ≥2.0 | DataFrames for summaries |
| `spc-spectra` | ≥0.4 | Parse `.spc` binary files |
| `scikit-learn` | ≥1.3 | Machine-learning utilities (future use) |
| `h5py` | ≥3.9 | MATLAB v7.3 HDF5 `.mat` files |
| `jupyter` | ≥1.0 | Interactive notebook environment |
