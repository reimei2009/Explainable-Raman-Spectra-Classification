# Module 06: Xác Thực Hướng Lâm Sàng và Độ Tin Cậy Thống Kê

## Mục Tiêu
Lượng hóa tiện ích lâm sàng và độ tin cậy thống kê của hệ thống được đề xuất.

## Các sản phẩm đầu ra bắt buộc
- Bộ độ đo lâm sàng đa lớp
- Khoảng tin cậy thông qua tái lấy mẫu
- Kiểm tra tính bền vững theo nhóm con và căng thẳng nhiễu
- Diễn giải hỗ trợ quyết định lâm sàng

## Nội dung các mục đề xuất
1. Các độ đo chính (macro-F1, độ chính xác cân bằng, nhạy cảm/độc ác theo lớp)
2. Khoảng tin cậy và ước lượng độ không chắc chắn
3. Phân tích nhóm con và kiểm tra công bằng
4. Tính bền vững đối với nhiễu và điện thế thiếu
5. Giải thích hỗ trợ quyết định và các tuyên bố cẩn thận

## Danh sách kiểm tra chấp nhận
- Có độ đo đầy đủ và có ý nghĩa lâm sàng.
- Khoảng tin cậy được báo cáo cho các kết quả chính.
- Hạn chế tính bền vững được ghi chép rõ ràng.
- Các yêu cầu xác thực dựa trên bằng chứng.

## Hành động tiếp theo sau khi phê duyệt
Triển khai script báo cáo cho xác thực thống kê và phân tích nhóm con.

## Đầu vào bắt buộc từ Module 02 (Gating trước khi vào Module 06)
- Báo cáo quyết định QC preprocessing:
	- `outputs/reports/module_02_qc_decision_report.md`
- Điều kiện khuyến nghị:
	- Nếu `val_warning=True` trong báo cáo QC, cần ghi chú rủi ro mất cân bằng lớp trước khi diễn giải kết quả lâm sàng.
	- Ưu tiên báo cáo thêm macro-F1 và balanced accuracy theo lớp để tránh lệch do mất cân bằng.
