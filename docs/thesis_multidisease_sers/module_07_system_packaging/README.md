# Module 07: Cấu Trúc Code, Đóng Gói, và Thực Thi Có Thể Lặp Lại

## Mục Tiêu
Xác định kiến trúc dự án rõ ràng, dễ bảo trì và luồng thực thi có thể lặp lại.

## Trạng Thái
✅ Hoàn thành. Bản bàn giao chi tiết nằm ở [RESULTS.md](RESULTS.md).

## Cấu trúc code mục tiêu
- src/data
  - loader và schema siêu dữ liệu
- src/preprocess
  - tiền xử lý tín hiệu và QC
- src/features
  - trích đặc trưng động điện thế và 2T2D
- src/models
  - huấn luyện, tinh chỉnh, hiệu chỉnh
- src/explain
  - giải thích được và ánh xạ
- src/eval
  - độ đo và xác thực lâm sàng
- scripts/pipelines
  - chạy pipeline module
- outputs/reports
  - xuất bảng và hình vẽ

## Các sản phẩm đầu ra bắt buộc
- Cây thư mục cuối cùng
- Sách chạy cho mỗi pipeline module
- Chiến lược cấu hình (yaml/json)
- Ghi chú seed có thể lặp lại và môi trường

## Chính sách an toàn thực thi
- Mọi script hoặc lệnh chạy mới phải được trình bày trước.
- Thực thi chỉ diễn ra sau khi được phê duyệt bởi người dùng.
- Log thực thi được theo dõi trong module_08_execution_log.

## Danh sách kiểm tra chấp nhận
- Cấu trúc nhất quán và dễ điều hướng.
- Pipeline có thể lặp lại trên môi trường sạch.
- Kết quả xác định dưới seed ngẫu nhiên cố định.
- Tài liệu hoàn chỉnh cho bàn giao.
