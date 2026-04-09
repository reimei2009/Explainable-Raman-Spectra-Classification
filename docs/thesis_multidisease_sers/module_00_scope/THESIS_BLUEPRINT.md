# THESIS BLUEPRINT: Nền Tảng Chẩn Đoán Đa Bệnh Túi Mật từ Voltage-SERS

## 1. Lý Do Chọn Đề Tài

### 1.1 Bối Cảnh Lâm Sàng
- Bệnh túi mật xảy ra ở ~15-20% dân số thế giới.
- 3 bệnh chính: sỏi túi mật (GB stone), u nhú (polyp), ung thư (cancer).
- Khó phân biệt bằng siêu âm / CT ở giai đoạn sớm → cần công cụ chẩn đoán sinh hóa.

### 1.2 Khoảng Trống Khoa Học
**Paper 1 (Anal Chem 2020):** Phân biệt cancer vs polyp bằng voltage-SERS + bilirubin derivatives.
- ✓ Chứng minh voltage điều khiển được khả năng phân biệt.
- ✗ Chỉ xử lý 2 lớp; không có giải thích mô hình.

**Paper 2 (Analyst 2021):** Phân biệt stone vs polyp bằng 2T2D correlation.
- ✓ Mở rộng tới 2 lớp khác.
- ✗ Dùng thủ công 2T2D trên 2 vết phổ; không có ML tự động.

**Paper 3 (SNB 2021):** Pretreatment-free SERS trên paper-AuND + 2T2D + k-NN.
- ✓ Thiết kế nền đo thực tế.
- ✗ k-NN đơn giản; chưa có dynamic features mở rộng; không giải thích.

### 1.3 Tính Mới Lạ Của Đề Tài
1. **Phân loại đa lớp (3 bệnh):** Stone ≠ Polyp ≠ Cancer trong 1 mô hình.
2. **Trích đặc trưng động học:** Mở rộng 2T2D từ 2 vết thành ma trận/tensor đầy đủ.
3. **ML tiên tiến + hiệu chỉnh xác suất:** Vượt quá k-NN, có calibration cho lâm sàng.
4. **Explainability:** SHAP + peak-region mapping → hypothesis biomarker.
5. **Hệ thống hoàn chỉnh:** Từ dữ liệu sồng đến báo cáo lâm sàng (end-to-end).

---

## 2. Câu Hỏi Nghiên Cứu và Giả Thuyết

### RQ1: Các đặc trưng động học theo điện thế có thể phân biệt được 3 bệnh túi mật không?
**Giả thuyết H1:** Các metabolites trong nước rút từ bile (bilirubin, urobilinogen) phản ứng khác nhau với điện thế tùy theo trạng thái bệnh. Bằng cách đo SERS trên dải điện thế -400 đến +400 mV, chúng tôi có thể xây ma trận động học 2D phản ánh sự khác biệt này.

### RQ2: Một mô hình ML đa lớp có thể huấn luyện và so sánh được trên tập dữ liệu này không?
**Giả thuyết H2:** Từ N=24 mẫu, ta có thể xây dựng một mô hình ML multiclass với kiến trúc hợp lý, được tinh chỉnh và calibrated cho kết quả lâm sàng.

### RQ3: Các vùng Raman quan trọng nhất cho mô hình tương ứng với hypothesis sinh học nào?
**Giả thuyết H3:** Các vùng peak quan trọng sẽ tương ứng với các đặc trưng phân tử đã biết (e.g., bilirubin ~1555 cm⁻¹, urobilinogen signature) hoặc phát hiện các marker mới.

---

## 3. Mục Tiêu Khóa Luận

### Mục Tiêu Chính
Phát triển một nền tảng chẩn đoán thực tế, được giải thích, có độ tin cậy cao cho 3 bệnh túi mật từ SERS điện hóa trên bile.

### Mục Tiêu Từng Module
- **Module 01:** Xây dựng pipeline dữ liệu hoàn chỉnh, gán nhãn theo tiêu chuẩn y khoa.
- **Module 02:** Thiết kế preprocessing robust giữ lại thông tin bệnh.
- **Module 03:** Trích đặc trưng động học từ quỹ đạo điện thế (mở rộng 2T2D).
- **Module 04:** Phát triển multiple models (baselines + advanced), tinh chỉnh và so sánh.
- **Module 05:** Tạo bộ công cụ giải thích để truy ngược thành vùng Raman quan trọng.
- **Module 06:** Xác thực lâm sàng: tính reliability, robustness, subgroup analysis.
- **Module 07:** Đóng gói thành hệ thống runnable với documentation đầy đủ.

---

## 4. Phạm Vi và Giới Hạn

### Phạm Vi
- **Dữ liệu:** 24 bệnh nhân từ bộ dữ liệu hiện có.
- **Âm mẫu tính positive:** 1–7 lần lặp × 9 điện thế × 2000+ wavenumber points.
- **Dãi điện thế:** -400 đến +400 mV (9 mức).
- **Phương pháp:** SERS trên nền Au nanodendrite (từ papers gốc).
- **Stack công nghệ:** Python 3.10+, scikit-learn, XGBoost, SHAP.

### Giới Hạn
- **Quy mô:** N=24 không large-scale; không dùng deep learning.
- **Sinh hóa:** Không xác thực in vitro mới; trích hypothesis từ mô hình.
- **Generalizable:** Chỉ áp dụng trên 1 loại nền (AuND).
- **Thời gian:** Snapshot quá khứ; không theo dõi bệnh nhân.

---

## 5. Sản Phẩm Đầu Ra (Deliverables)

| ID | Sản Phẩm | Mô Tả |
|----|---------|-------|
| D1 | Bộ dữ liệu v1.0 | Metadata, nhãn, phân chia train/val/test |
| D2 | Ma trận đặc trưng v1.0 | 24 × (N features) file numpy standardized |
| D3 | Benchmark models | Danh sách các models + so sánh hiệu năng |
| D4 | Explainability report | SHAP + peak-region mapping |
| D5 | Validation report | Phân tích lâm sàng, robustness checks |
| D6 | Reproducible code | src/ + scripts/ + docs/ hoàn chỉnh |
| D7 | Khóa luận + slides | 80–100 trang + materials bảo vệ |

---

## 6. Ma Trận Rủi Ro

| Rủi Ro | Xác Suất | Mức Độ | Giảm Thiểu |
|--------|----------|--------|-----------|
| N=24 quá nhỏ → model overfitting | Cao | Cao | Cross-validation × regularization; simpler models as fallback |
| Features quá cao chiều | Cao | Trung | PCA + feature selection |
| Class imbalance | Trung | Trung | Class weights hoặc SMOTE; stratified split |
| Thiếu một số điện thế | Trung | Thấp | Imputation hoặc loại trừ; report missing |
| Thi hành muộn | Thấp | Trung | Weekly checkpoint; cut scope if needed |

---

## 7. Tiêu Chí Chấp Nhận

### Tiêu Chí Bắt Buộc
- ✓ Không rò rỉ mẫu (separate patient-level).
- ✓ Code reproducible trên Python 3.10+.
- ✓ Mỗi đặc trưng được giải thích được.
- ✓ Tối thiểu 3 models được so sánh.

### Tiêu Chí Nên Có
- ✓ 5+ models được benchmark.
- ✓ Explainability reports cho mô hình tốt nhất.
- ✓ Uncertainty quantification.
- ✓ Robustness vs missing potentials.

---

## 8. Lộ Trình 6 Tuần Còn Lại

**Dự kiến:** 1 module chính mỗi tuần + 1 tuần overlap final packaging.

**Tuần 1–2:** Module 01 (Data) + Module 02 (Preprocessing)  
**Tuần 3–4:** Module 03 (Features) + Module 04 (Models)  
**Tuần 5:** Module 05 (Explain) + Module 06 (Validation)  
**Tuần 6:** Module 07 (Packaging) + Thesis writing  

---

## 9. Kế Tiếp từ Module 00

Sau khi Blueprint được approved:
1. Khoá scope baseline.
2. Khởi động Module 01 (Data Governance).
3. Weekly checkpoint vs timeline.

---

**Status:** ✅ Approved. Scope locked. Proceed to Module 01.
