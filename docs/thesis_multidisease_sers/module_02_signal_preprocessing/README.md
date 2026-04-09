# Module 02: Tiền Xử Lý Tín Hiệu và Kiểm Soát Chất Lượng

## Mục Tiêu
Xây dựng pipeline tiền xử lý mạnh mẽ giữ lại thông tin Raman liên quan đến bệnh.

## Quyết định chính
- Chuẩn hóa phổ về cùng lưới 2048 điểm.
- Cắt vùng phổ là tùy chọn theo từng đợt kiểm tra dữ liệu.
- AirPLS là baseline mặc định để trừ nền.
- Augmentation chỉ áp dụng cho tập train.

## Các sản phẩm đầu ra bắt buộc
- Quy trình vận hành tiêu chuẩn tiền xử lý (SOP)
- Pipeline code tiền xử lý có thể lặp lại
- Phân tích so sánh các tùy chọn tiền xử lý
- Độ đo kiểm soát chất lượng và cờ thất bại

## Nội dung các mục đề xuất
1. Đặc điểm tín hiệu thô và lỗi
2. Chiến lược loại bỏ tia vũ trụ
3. Tùy chọn hiệu chỉnh baseline và tinh chỉnh tham số
4. Chính sách làm mịn và chuẩn hóa
5. Chiến lược tổng hợp lần lặp
6. Độ đo QC và ngưỡng từ chối
7. Tăng cường dữ liệu train-only

## Danh sách kiểm tra chấp nhận
- Pipeline tạo ra phổ ổn định trên các lần chạy.
- Các tham số được chứng minh và có thể lặp lại.
- Các cờ chất lượng được tạo tự động.
- Tiền xử lý cải thiện khả năng phân tách xuôi dòng.

## Hành động tiếp theo sau khi phê duyệt
Tạo notebook/script benchmark so sánh các candidate pipeline tiền xử lý.

## Script kiểm định nhanh chất lượng preprocessing
- File: `scripts/module_02_qc_preprocessing.py`
- Mục đích:
	- Xem nhanh phân bố dữ liệu sau preprocessing trước khi đưa vào model.
	- Tạo PCA 2D preview theo split train/val/test.
	- Tạo t-SNE preview theo lớp bệnh.
	- Tạo biểu đồ class separation (intra/inter class distance) và xuất chỉ số tách lớp.
	- Cảnh báo tự động khi val separation thấp hơn ngưỡng cấu hình.
	- Đo khoảng cách centroid giữa từng cặp lớp cho train/val/test.
	- Chạy ablation nhiều seed để kiểm tra độ ổn định t-SNE trên train.
- Đầu vào:
	- `outputs/data/X_train.npy`
	- `outputs/data/X_val.npy`
	- `outputs/data/X_test.npy`
	- `outputs/data/metadata.npz`
- Đầu ra:
	- `outputs/figures/module_02/module_02_qc_preview_baseline.png` hoặc `..._with_aug.png`
	- `outputs/reports/module_02_preprocessing_qc_baseline.csv` hoặc `..._with_aug.csv`
	- `outputs/reports/module_02_preprocessing_qc_*_pairwise_train.csv` (khoảng cách theo từng cặp lớp ở tập train)
	- `outputs/reports/module_02_preprocessing_qc_*_centroid_distances.csv`
	- `outputs/reports/module_02_preprocessing_qc_*_stability_tsne.csv`
- Ví dụ lệnh chạy:
	- `python scripts/module_02_qc_preprocessing.py`
	- `python scripts/module_02_qc_preprocessing.py --include-aug-train`
	- `python scripts/module_02_qc_preprocessing.py --include-aug-train --tag ablation_v1`
	- `python scripts/module_02_qc_preprocessing.py --val-sep-threshold 1.05 --val-sil-threshold 0.00`
	- `python scripts/module_02_qc_preprocessing.py --stability-seeds 11,22,33,44,55`

## Script tổng hợp quyết định QC (Baseline vs With-Aug)
- File: `scripts/module_02_qc_decision_report.py`
- Mục đích:
	- Tổng hợp các CSV QC thành 1 báo cáo markdown dễ đọc.
	- So sánh baseline và with-augmentation theo split.
	- Tính chênh lệch chính (`delta_sep_ratio`, `delta_silhouette`).
	- Đưa khuyến nghị tự động khi `val_warning=True`.
- Đầu ra mặc định:
	- `outputs/reports/module_02_qc_decision_report.md`
- Ví dụ lệnh chạy:
	- `python scripts/module_02_qc_decision_report.py`
	- `python scripts/module_02_qc_decision_report.py --baseline-tag baseline_r2 --with-aug-tag with_aug_r2 --stability-tag with_aug_r2_stab`
