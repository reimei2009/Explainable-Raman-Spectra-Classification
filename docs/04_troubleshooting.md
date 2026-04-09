# 04. Troubleshooting Guide

## 1. `spc-spectra` Import Error

**Error:**
```
ModuleNotFoundError: No module named 'spc'
```

**Cause:** The `spc-spectra` package is not installed (or the wrong venv is active).

**Fix:**
```batch
venv\Scripts\activate.bat
pip install spc-spectra
```
> **Note:** The package is installed as `spc-spectra` but imported as `import spc`.

---

## 2. MATLAB `.mat` File Loading Issues (v7.3 / HDF5)

**Error:**
```
NotImplementedError: Please use HDF reader for matlab v7.3 files
```

**Cause:** The `.mat` file was saved with MATLAB v7.3+ which uses HDF5 format.
`scipy.io.loadmat` cannot read these.

**Fix:** The `RamanDataLoader.load_mat()` method automatically falls back to `h5py`:
```python
loader = RamanDataLoader()
data = loader.load_mat("file.mat")  # uses h5py automatically
```

If you are using `scipy.io.loadmat` directly, switch to:
```python
import h5py
with h5py.File("file.mat", "r") as f:
    keys = list(f.keys())
    data = {k: f[k][()] for k in keys}
```

---

## 3. Unicode / Encoding Issues with File Paths

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte …
```
or files not found due to accented characters in paths.

**Fix:**
* Always use `pathlib.Path` objects (not raw strings) for file paths.
* Open text files with `encoding="utf-8"`:
  ```python
  with open(path, "r", encoding="utf-8") as f:
      content = f.read()
  ```
* If running on Windows, set the console code page: `chcp 65001` before running scripts.

---

## 4. Memory Errors with Large Datasets

**Error:**
```
MemoryError
```

**Cause:** Loading all samples at once into a nested dict can use several GB of RAM
if spectra are large or there are many replicates.

**Fixes:**

**Option A – Load one sample at a time:**
```python
loader = RamanDataLoader()
for sample_name in list_samples("Data/different potential"):
    data = loader.load_sample_folder(f"Data/different potential/{sample_name}")
    # process and discard
    del data
```

**Option B – Limit replicates loaded:**
```python
import re
folder = Path("Data/different potential/M78S")
for fp in folder.glob("*_1.spc"):   # only replicate 1
    sp = loader.load_spc(fp)
```

**Option C – Save preprocessed spectra to disk:**
```python
import numpy as np
np.savez_compressed(
    "outputs/processed/M78S_0mV.npz",
    wavenumber=wn, intensity=normed
)
```

---

## 5. Matplotlib Display Issues (Headless / No Display)

**Error:**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```
or plots not appearing in Jupyter.

**Fix A – Use non-interactive backend:**
```python
import matplotlib
matplotlib.use("Agg")   # must be called BEFORE importing pyplot
import matplotlib.pyplot as plt
```

**Fix B – In Jupyter, ensure inline magic is set:**
```python
%matplotlib inline
```

**Fix C – `RamanVisualizer.save_figure()` already works headlessly** because it
calls `fig.savefig()` and does not call `plt.show()`.  Use `plt.close(fig)` after
saving to free memory.

---

## 6. `pwsh.exe` / PowerShell Not Found

**Error:**
```
'pwsh.exe' is not recognized as an internal or external command
```

**Cause:** PowerShell 7+ (Core) is not installed.  Some tools require it.

**Fixes:**

**Option A – Use `setup.bat` with classic Command Prompt:**
```
Win + R → cmd → Enter
cd C:\Users\ADMIN\Desktop\Prj\KLTN
setup.bat
```

**Option B – Install PowerShell 7:**
Download from https://aka.ms/powershell

**Option C – Activate venv manually in cmd:**
```batch
venv\Scripts\activate.bat
python scripts\sample_display.py
```

---

## 7. `spc` File Reads Wrong (Flat or Zero Intensity)

**Symptom:** All intensity values are 0 or the wavenumber array has only 2 points.

**Cause:** Some SPC files store the X axis only in the file header (not per sub-file).

**Fix:** The `load_spc` method already handles this by checking `sub.x` and falling
back to `f.x`.  If you still see issues, inspect the raw SPC object:
```python
import spc
f = spc.File("file.spc")
print("Num sub-files:", len(f.sub))
print("f.x:", f.x[:5] if hasattr(f, "x") else "none")
print("sub[0].x:", f.sub[0].x[:5] if f.sub[0].x is not None else "none")
print("sub[0].y:", f.sub[0].y[:5])
```

---

## 8. `pip install` Fails Behind a Proxy

**Error:**
```
Could not fetch URL … ProxyError
```

**Fix:**
```batch
pip install --proxy http://user:pass@proxy:port -r requirements.txt
```
or set environment variables:
```batch
set HTTPS_PROXY=http://user:pass@proxy:port
pip install -r requirements.txt
```
