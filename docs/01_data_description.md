# 01. Mô tả Dữ liệu / Data Description

## 1. Phổ Raman là gì? / What is Raman Spectroscopy?

**Tiếng Việt:**
Phổ Raman là kỹ thuật quang phổ không phá hủy dựa trên hiện tượng tán xạ không đàn hồi
(tán xạ Raman) của ánh sáng. Khi ánh sáng laser chiếu vào mẫu, một phần nhỏ photon bị
tán xạ với năng lượng khác với photon tới. Sự dịch chuyển năng lượng (Raman shift, đơn
vị cm⁻¹) phản ánh các dao động phân tử đặc trưng của vật chất, cho phép xác định thành
phần hóa học và cấu trúc phân tử.

**English:**
Raman spectroscopy is a non-destructive optical technique based on inelastic (Raman)
scattering of light. When a laser beam illuminates a sample, a small fraction of photons
are scattered at shifted energies. The energy shift (Raman shift, cm⁻¹) reflects
characteristic molecular vibrations, enabling chemical composition and molecular
structure identification.

### SERS – Surface-Enhanced Raman Spectroscopy
Trong dự án này, phổ được đo bằng **SERS** (phổ Raman tăng cường bề mặt), trong đó
tín hiệu Raman được khuếch đại nhiều bậc nhờ các hạt nano kim loại (vàng / bạc). Điện
thế điện hóa được áp dụng lên điện cực để kiểm soát quá trình hấp phụ của phân tử
lên bề mặt nano.

In this project spectra are recorded using **SERS** (Surface-Enhanced Raman Spectroscopy),
in which the Raman signal is amplified by metallic nanoparticles (gold/silver).
An electrochemical potential is applied to the electrode to control molecular adsorption
on the nano-surface.

---

## 2. Cấu trúc thư mục / Directory Structure

```
KLTN/
├── Data/
│   └── different potential/
│       ├── F35S/
│       │   ├── -400_1.spc
│       │   ├── -400_2.spc
│       │   ├── ...
│       │   └── F35S.mat        (optional MATLAB combined file)
│       ├── f41p/
│       ├── F42S/
│       └── ...  (24 sample folders total)
├── src/                        Python source modules
├── scripts/                    Runnable analysis scripts
├── docs/                       Documentation
├── outputs/
│   ├── figures/                Generated plots
│   ├── processed/              Pre-processed spectra (CSV / NPZ)
│   └── reports/                Summary reports
└── requirements.txt
```

---

## 3. Quy ước đặt tên mẫu / Sample Naming Convention

| Ký tự / Character | Ý nghĩa / Meaning |
|---|---|
| **F** / **f** | Female (giới tính nữ) |
| **M** / **m** | Male (giới tính nam) |
| **<số>** | Mã số đối tượng (subject ID) |
| **S** | Solid (mẫu rắn) |
| **P** | Pellet |
| **C** | Cell (tế bào) |
| **GA** | Glutaraldehyde-treated |

**Ví dụ / Examples:**

| Tên mẫu | Giới tính | ID | Loại |
|---|---|---|---|
| M78S | Nam | 78 | Solid |
| F60S | Nữ | 60 | Solid |
| f41p | Nữ | 41 | Pellet |
| m66c | Nam | 66 | Cell |
| F49GA | Nữ | 49 | Glutaraldehyde |

---

## 4. Định dạng tệp / File Formats

### 4.1 Tệp `.spc` (Thermo Galactic GRAMS)

Tệp nhị phân được đặt tên theo dạng `<potential>_<replicate>.spc`.

* **Potential**: điện thế áp đặt (mV), có thể âm, ví dụ `-200`, `0`, `400`
* **Replicate**: số thứ tự lần đo lặp (1–7)

Ví dụ: `-200_3.spc` là lần đo lặp thứ 3 tại điện thế -200 mV.

Đọc bằng Python với gói `spc-spectra`:
```python
import spc
f = spc.File("path/to/file.spc")
wavenumber = f.x          # mảng Raman shift (cm⁻¹)
intensity  = f.sub[0].y   # mảng cường độ
```

### 4.2 Tệp `.mat` (MATLAB workspace)

Chứa dữ liệu kết hợp nhiều lần đo của một mẫu. Có hai phiên bản:
* **v5 / v6** – đọc bằng `scipy.io.loadmat`
* **v7.3 (HDF5)** – đọc bằng `h5py`

```python
from scipy.io import loadmat
data = loadmat("M78S.mat")
```

---

## 5. Dải điện thế / Potential Range

| Điện thế (mV) | Ghi chú |
|---|---|
| -400 | Điện thế âm mạnh nhất |
| -300 | |
| -200 | |
| -100 | |
| 0 | Điện thế chuẩn |
| +100 | |
| +200 | |
| +300 | |
| +400 | Điện thế dương mạnh nhất |

Điện thế được thay đổi để nghiên cứu ảnh hưởng của trường điện đến sự hấp phụ và tín
hiệu SERS của phân tử trên bề mặt nano.

---

## 6. Danh sách mẫu / Sample List

| STT | Tên mẫu | Giới tính | ID | Loại |
|---|---|---|---|---|
| 1 | F35S | Nữ | 35 | Solid |
| 2 | f41p | Nữ | 41 | Pellet |
| 3 | F42S | Nữ | 42 | Solid |
| 4 | f45c | Nữ | 45 | Cell |
| 5 | F45S | Nữ | 45 | Solid |
| 6 | F49GA | Nữ | 49 | GA |
| 7 | F50P | Nữ | 50 | Pellet |
| 8 | F51S | Nữ | 51 | Solid |
| 9 | F59C | Nữ | 59 | Cell |
| 10 | F60S | Nữ | 60 | Solid |
| 11 | F64SP | Nữ | 64 | Solid+Pellet |
| 12 | F64SS | Nữ | 64 | Solid (2nd) |
| 13 | f65p | Nữ | 65 | Pellet |
| 14 | f70p | Nữ | 70 | Pellet |
| 15 | m27s | Nam | 27 | Solid |
| 16 | M29P | Nam | 29 | Pellet |
| 17 | m39s | Nam | 39 | Solid |
| 18 | m40p | Nam | 40 | Pellet |
| 19 | M41P | Nam | 41 | Pellet |
| 20 | M48S | Nam | 48 | Solid |
| 21 | M56S | Nam | 56 | Solid |
| 22 | m66c | Nam | 66 | Cell |
| 23 | M68P | Nam | 68 | Pellet |
| 24 | M78S | Nam | 78 | Solid |

---

## 7. Ghi chú chất lượng dữ liệu / Data Quality Notes

* Một số mẫu có thể thiếu một vài điện thế hoặc số lần lặp ít hơn 7.
* Tia vũ trụ (cosmic rays) có thể tạo ra các đỉnh giả trong phổ – hãy kiểm tra
  bằng `RamanPreprocessor.remove_cosmic_rays()`.
* Phổ từ các loại mẫu khác nhau (S, P, C, GA) không nên so sánh trực tiếp mà không
  chuẩn hóa trước.
* Một số tệp `.mat` có thể ở định dạng HDF5 (v7.3); nếu `scipy.io.loadmat` báo lỗi,
  hãy dùng `h5py`.
