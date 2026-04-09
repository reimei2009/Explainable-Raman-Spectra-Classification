# Module 01: Kế Hoạch Triển Khai Chi Tiết

## Mục Tiêu
Xây dựng pipeline dữ liệu hoàn chỉnh: metadata + nhãn + phân chia train/val/test không rò rỉ mẫu.

---

## 1. Schema Dữ Liệu và Nhãn

### 1.1 Bảng Metadata Mẫu (Metadata Table)

Sẽ tạo file `data/manifest.csv` với các cột:

| Cột | Kiểu | Mô Tả | Ví Dụ |
|-----|------|-------|-------|
| sample_id | string | Tên mẫu | M78S |
| gender | {F, M} | Giới tính | M |
| subject_id | int | ID bệnh nhân | 78 |
| sample_type | {S, P, C, GA} | Loại mẫu | S |
| disease_label | {stone, polyp, cancer} | Nhãn bệnh | stone |
| disease_code | {0, 1, 2} | Mã số để lập mô hình | 0 → stone, 1 → polyp, 2 → cancer |
| data_source | {original, stone} | Nguồn dữ liệu | original |
| available_potentials | string | Danh sách điện thế sẵn | -400,-200,0,200,400 |
| replicate_counts | dict | Số lần lặp mỗi điện thế | {"-400": 3, "-200": 2, ...} |
| notes | string | Ghi chú chất lượng | missing_300_400_mV |

### 1.2 Mapping Bệnh → Nhãn

```
stone (Sỏi) → 0
polyp (U nhú) → 1
cancer (Ung thư) → 2
```

**Nguồn gốc nhãn:**
- Từ tên folder sample hoặc tài liệu bệnh lý gốc (sẽ tra cứu).
- Nếu không chắc, ghi chú "uncertain" + exclude từ training nếu cần.

---

## 2. Quy Trình Gán Nhãn (Labeling Protocol)

### 2.1 Quy Tắc
- Mỗi **sample** (folder, VD: M78S) có **1 nhãn duy nhất**.
- Không có sample vừa stone vừa polyp.
- Nếu một người có nhiều loại mẫu (VD: F64SP = Solid+Pellet), tính là 2 samples riêng.

### 2.2 Quy Trình Khác
1. Kiểm tra dữ liệu gốc (file .mat hoặc comment).
2. Nếu không rõ → liên hệ hoặc loại trừ.
3. Ghi chú lại quyết định + tỉ lệ tin cậy (high/medium/low).

---

## 3. Chiến Lược Chia Tập (Split Strategy)

### 3.1 Mục Tiêu
Tránh rò rỉ mẫu: tất cả lần lặp + điện thế của **cùng 1 bệnh nhân** phải cùng 1 tập.

### 3.2 Quy Trình
1. **Nhóm theo bệnh nhân (subject_id)**, không phải theo sample_id.
2. Chia tập ở mức **subject**, rồi mới gán tất cả samples của subject đó.
3. Tỉ lệ dự kiến:
   - Train: 60% (bệnh nhân)
   - Validation: 20% (bệnh nhân)
   - Test: 20% (bệnh nhân)

### 3.3 Cân Bằng Lớp
**Mục tiêu:** Mỗi tập đều có đủ 3 lớp (stone, polyp, cancer).

**Thách thức:** N=24 bệnh nhân quá nhỏ.

**Giải Pháp:**
- Dùng `train_test_split(..., stratify=disease_label)` để đảm bảo tỷ lệ lớp.
- Nếu không đủ, chép một vài mẫu sang để đảm bảo có ít nhất 1 ca mỗi lớp trong val/test.
- Ghi chép lại quyết định.

---

## 4. Kiểm Tra Chất Lượng Dữ Liệu (QC)

### 4.1 Báo Cáo Chất Lượng
Sẽ tạo file `outputs/reports/data_quality_report.csv` ghi:

| sample_id | n_potentials | n_replicates | missing_potentials | cosmic_ray_suspects | status |
|-----------|-----|-----|---------|----------|--------|
| M78S | 9 | 7 | none | 0 | ✓ pass |
| F35S | 7 | 5 | -300,+300 | 2 | ⚠ warn |

**Cột Chi Tiết:**
- `n_potentials`: số điện thế có dữ liệu (tối đa 9).
- `n_replicates`: trung bình số lần lặp.
- `missing_potentials`: danh sách điện thế thiếu.
- `cosmic_ray_suspects`: số peak bất thường (dự kiến từ outlier detection).
- `status`: ✓ pass / ⚠ warn / ✗ fail (nếu thiếu quá nhiều).

### 4.2 Nguyên Tắc QC
- Lặp tối thiểu 1, tối đa 7 mỗi điện thế.
- Nếu thiếu > 3 điện thế → ghi chú "partial".
- Nếu thiếu quá nhiều → có thể exclude từ modeling.

---

## 5. Cấu Trúc Output

### 5.1 File Đầu Ra từ Module 01

```
outputs/
├── data/
│   └── manifest_v1.0.csv          # Metadata + disease label
└── reports/
    ├── data_quality_report.csv    # QC assessment
    ├── split_summary.txt          # Train/val/test breakdown
    └── labeling_notes.md          # Ghi chú gán nhãn + quyết định
```

### 5.2 Phân Chia Mẫu (Ví Dụ Dự Kiến)

```
N=24 samples (từ rougly 20 bệnh nhân sau khi merge)

Tập Train: ~12 samples (60%)
Tập Val:   ~4 samples (20%)
Tập Test:  ~8 samples (20%)

Mỗi tập có:
  - Stone: 3–4
  - Polyp: 2–3
  - Cancer: 2–3
```

---

## 6. Code Structure (Dự Kiến)

Sẽ tạo những file sau trong codebase:

```
src/data/
├── manifest.py          # Tải & parse manifest.csv
├── labeling.py          # Mapping disease → code
└── split.py            # Chia train/val/test

scripts/
└── module_01_prepare_data.py   # Pipeline chính Module 01
```

### 6.1 Script Chính
`scripts/module_01_prepare_data.py` sẽ:
1. Scan tất cả sample folders.
2. Tạo manifest.csv từ metadata.
3. Chạy QC checks.
4. Chia tập theo stratified patient-level split.
5. Lưu output vào outputs/data/ + outputs/reports/.

### 6.2 Input Cần Từ Bạn
- Bảng gán nhãn bệnh (`sample_id → disease_label`).
- Hay tìm trong tài liệu gốc.

---

## 7. Checklist Trước Khi Code

Trước khi viết `module_01_prepare_data.py`, cần xác nhận:

- [ ] Schema metadata ở trên OK không?
- [ ] Mapping bệnh → code OK không?
- [ ] Quy trình chia tập OK không?
- [ ] Bạn có thông tin gán nhãn bệnh sẵn không? (Hay mình scan từ data?)

---

## 8. Kế Tiếp

Sau khi approved:
1. Mình viết `src/data/manifest.py` + `scripts/module_01_prepare_data.py`.
2. Chạy trên dữ liệu thực.
3. Output manifest + QC report để bạn kiểm tra.
4. Nếu OK → lock manifest v1.0 → Module 02.

---

**Status:** ⏳ Chờ phê duyệt + thông tin gán nhãn bệnh
