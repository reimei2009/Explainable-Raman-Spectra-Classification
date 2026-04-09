# Module 04: Mô Hình Hóa Đa Lớp và Hiệu Chỉnh

## Mục Tiêu
Huấn luyện và so sánh mô hình đa lớp để chẩn đoán sỏi vs polyp vs ung thư.

## Các sản phẩm đầu ra bắt buộc
- Benchmark mô hình baseline
- Benchmark mô hình nâng cao
- Giao thức tìm kiếm siêu tham số
- Hiệu chỉnh xác suất và ngưỡng quyết định
- Lý do lựa chọn mô hình cuối cùng

## Nội dung các mục đề xuất
1. Mô hình baseline (k-NN, SVM, RF)
2. Mô hình nâng cao (gradient boosting, ensemble)
3. Giao thức cross-validation
4. Xử lý mất cân bằng lớp
5. Hiệu chỉnh (Platt / isotonic)
6. Phân tích lỗi và mẫu phân loại sai

## Danh sách kiểm tra chấp nhận
- Giao thức đánh giá an toàn không rò rỉ.
- Mô hình tốt nhất được chọn với tiêu chí rõ ràng.
- Hiệu chỉnh độ tin cậy được xác thực.
- Mẫu lỗi được giải thích theo lớp bệnh.

## Hành động tiếp theo sau khi phê duyệt
Triển khai pipeline huấn luyện và bảng benchmark đa lớp.

## Script baseline cho vòng huấn luyện đầu tiên
- File: `scripts/module_04_train_baselines.py`
- Mục đích:
	- Huấn luyện baseline đa lớp từ dữ liệu đã preprocess ở Module 02.
	- Dùng class-weight để giảm lệch lớp (đặc biệt lớp hiếm).
	- Xuất đầy đủ `macro-F1`, `balanced_accuracy`, `accuracy`, `weighted-F1` cho val/test.
	- Xuất per-class metrics (precision/recall/F1 theo stone/polyp/cancer).
- Mô hình baseline hiện tại:
	- `logreg_balanced`
	- `svm_rbf_balanced`
	- `rf_balanced`

## Đầu vào và đầu ra
- Đầu vào chính:
	- `outputs/data/X_train.npy`, `X_val.npy`, `X_test.npy`
	- `outputs/data/metadata.npz`
	- `outputs/reports/module_02_qc_decision_report.md` (gating QC)
- Đầu ra:
	- `outputs/reports/module_04/benchmark_summary.csv`
	- `outputs/reports/module_04/per_class_metrics.csv`
	- `outputs/reports/module_04/model_selection.txt`
	- `outputs/reports/module_04/benchmark_report.md`
	- `outputs/figures/module_04/confusion_matrix_<model>_<split>.png`

## Ví dụ lệnh chạy
- Không dùng augmentation train:
	- `python scripts/module_04_train_baselines.py`
- Có augmentation train:
	- `python scripts/module_04_train_baselines.py --include-aug-train`
- Chạy có xác nhận rõ khi QC đang warning:
	- `python scripts/module_04_train_baselines.py --include-aug-train --allow-qc-warning`

## Script so sánh cấu hình no_aug vs with_aug
- File: `scripts/module_04_compare_configs.py`
- Mục đích:
	- Tổng hợp benchmark từ 2 thư mục thí nghiệm (`no_aug`, `with_aug`).
	- Tạo bảng xếp hạng mô hình theo điểm tổng hợp (val + test).
	- Khi QC warning còn tồn tại, có thể bật rule ưu tiên `test balanced accuracy`.
	- Xuất báo cáo quyết định cấu hình baseline cho vòng tuning tiếp theo.
- Đầu ra:
	- `outputs/reports/module_04/config_comparison.csv`
	- `outputs/reports/module_04/config_decision_report.md`
- Ví dụ lệnh chạy:
	- `python scripts/module_04_compare_configs.py`
	- `python scripts/module_04_compare_configs.py --qc-warning-sensitive`
	- `python scripts/module_04_compare_configs.py --qc-warning-sensitive --min-val-macro-f1 0.30`

## Script tuning SVM (theo winner đã chốt)
- File: `scripts/module_04_tune_svm.py`
- Mục đích:
	- Grid search gọn cho SVM class-weighted (kernel/C/gamma).
	- Chọn mô hình tốt nhất theo val macro-F1 + val balanced accuracy.
	- Tie-break thêm theo recall lớp cancer trên val (có thể tắt nếu cần).
	- Xuất thêm chỉ số test để kiểm tra độ bền.
- Đầu ra:
	- `outputs/reports/module_04/svm_tuning/tuning_results.csv`
	- `outputs/reports/module_04/svm_tuning/best_model_metrics.csv`
	- `outputs/reports/module_04/svm_tuning/best_model_per_class.csv`
	- `outputs/reports/module_04/svm_tuning/tuning_report.md`
	- `outputs/figures/module_04/svm_tuning/confusion_matrix_best_val.png`
	- `outputs/figures/module_04/svm_tuning/confusion_matrix_best_test.png`
- Ví dụ lệnh chạy:
	- `python scripts/module_04_tune_svm.py`
	- `python scripts/module_04_tune_svm.py --include-aug-train`
	- `python scripts/module_04_tune_svm.py --c-grid 0.5,1,2,4 --gamma-grid scale,0.005,0.01`
	- `python scripts/module_04_tune_svm.py --disable-cancer-recall-tiebreak`

## Script chốt quyết định cuối Module 04
- File: `scripts/module_04_finalize_decision.py`
- Mục đích:
	- Gom kết quả so sánh cấu hình + tuning SVM thành báo cáo cuối module.
	- Nêu rõ caveat dữ liệu (đặc biệt khi test thiếu lớp cancer).
- Đầu ra:
	- `outputs/reports/module_04/final_decision.md`
- Ví dụ lệnh chạy:
	- `python scripts/module_04_finalize_decision.py`
