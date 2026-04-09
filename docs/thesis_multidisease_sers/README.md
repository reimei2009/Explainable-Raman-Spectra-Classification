# Khóa Luận: Chẩn Đoán Đa Bệnh Túi Mật từ Voltage-SERS

## 1. Đề Tài Khóa Luận Đã Chọn
Phát triển nền tảng chẩn đoán đa bệnh túi mật từ SERS điện hóa bằng cách sử dụng chữ ký phổ động thái theo điện thế, phân tích 2T2D mở rộng, và học máy giải thích được.

Các bệnh lâm sàng mục tiêu:
- Sỏi túi mật (GB stone)
- Polyp túi mật (GB polyp)
- Ung thư túi mật (GB cancer)

## 2. Tại sao đây là đề tài lớn và mạnh nhất
- Tích hợp cả 3 hướng từ các bài báo gốc thành một hệ thống hoàn chỉnh từ đầu đến cuối.
- Mở rộng bài toán phân loại nhị phân lên phân loại đa lớp có ý nghĩa lâm sàng.
- Kết hợp quang phổ, hóa đơn, học máy và khả năng giải thích.
- Có thể đưa ra cả kết quả kỹ thuật lẫn phương pháp học thuật có thể công bố.

## 3. Lộ trình các module
1. module_00_scope: định nghĩa vấn đề, câu hỏi nghiên cứu, sản phẩm đầu ra
2. module_01_data_governance: quản lý dữ liệu, gán nhãn, chiến lược chia tập
3. module_02_signal_preprocessing: pipeline tiền xử lý tín hiệu mạnh mẽ
4. module_03_voltage_dynamics_2t2d: trích đặc trưng động học điện thế và mở rộng 2T2D
5. module_04_modeling_multiclass: baseline đa lớp và mô hình nâng cao
6. module_05_explainable_ai: giải thích mô hình và vùng phổ quan trọng
7. module_06_validation_clinical: độ đo lâm sàng, độ không chắc chắn, diễn giải y khoa
8. module_07_system_packaging: đóng gói code, chạy lặp lại, báo cáo
9. module_08_execution_log: log từng bước tiến độ và phê duyệt

## 4. Quy Định Cộng Tác (Quan Trọng)
Chính sách brakepoint thực thi:
- Tôi sẽ đề xuất từng đoạn code/script mới trước khi chạy.
- Bạn xem xét và phê duyệt trước.
- Chỉ sau khi được phê duyệt, lệnh chạy mới được thực thi.

## 5. Cấu Trúc Code Dự Kiến
Cấu trúc triển khai được đề xuất:
- src/data: load dữ liệu, kiểm chứng schema, siêu dữ liệu mẫu
- src/preprocess: loại nhiễu, hiệu chỉnh baseline, chuẩn hóa, kiểm tra chất lượng
- src/features: trích đặc trưng chuỗi điện thế và 2T2D
- src/models: huấn luyện, hiệu chỉnh, lựa chọn mô hình
- src/explain: SHAP và giải thích vùng đỉnh
- src/eval: độ đo lâm sàng và tạo báo cáo
- scripts/pipelines: điểm vào chạy được cho mỗi module
- outputs/reports: tóm tắt thí nghiệm và bảng khóa luận

Cấu trúc chi tiết được ghi trong module_07_system_packaging.

## 6. Trạng Thái Hiện Tại
- Đề tài đã chốt.
- Khung tài liệu module đã hoàn thiện đến Module 08.
- Các kết quả từ Module 02 đến Module 07 đã được sinh và ghi nhận trong outputs/reports.
- Nhật ký thực thi đang được cập nhật để bàn giao cuối cùng.
