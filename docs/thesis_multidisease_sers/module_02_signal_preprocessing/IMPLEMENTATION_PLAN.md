# Module 02: Kế Hoạch Triển Khai Tiền Xử Lý Tín Hiệu

## 1. Mục tiêu
Xây dựng một pipeline tiền xử lý phổ Raman ổn định, có thể lặp lại và phù hợp cho dữ liệu huỳnh quang/bão hòa nền của bile SERS.

## 2. Quy tắc xử lý chính
- Cắt vùng phổ trước để loại bỏ phần không dùng tới.
- Nội suy về cùng số điểm bước sóng để đầu vào mô hình thống nhất.
- Trừ nền bằng airPLS làm phương án mặc định.
- Giữ ALS làm baseline so sánh.
- Chuẩn hóa cường độ về [0, 1].
- Không augment validation/test.

## 3. Cấu phần kỹ thuật
- `crop_spectrum()`
- `resample_spectrum()`
- `remove_cosmic_rays()`
- `baseline_correction_airpls()`
- `baseline_correction_als()`
- `smooth_savgol()`
- `normalize_minmax()`
- `preprocess_spectrum()`
- `RamanAugmenter` cho tập train

## 4. Augmentation được phép
- Gaussian noise nhỏ
- Multiplicative noise
- Polynomial baseline drift
- Fluorescence-like background
- Shift phổ nhẹ
- Stretch/compress nhẹ
- Scaling cường độ

## 5. Augmentation không được phép
- Thay đổi mạnh hình dạng peak
- Làm lệch vùng Raman quan trọng quá mức
- Áp dụng cho validation/test

## 6. Output mong muốn
- Mỗi phổ sau xử lý có cùng kích thước đầu ra
- Có thể lưu ảnh so sánh raw vs preprocessed
- Có pipeline demo để kiểm tra nhanh một sample

## 7. Kiểm tra chấp nhận
- Pipeline chạy được trên ít nhất 1 sample.
- Baseline airPLS không làm hỏng peak chính.
- Chuẩn hóa 0-1 hoạt động đúng.
- Augmentation tạo ra biến thể nhẹ, hợp lý.

## 8. Bước tiếp theo
Sau khi code xong, tạo script demo Module 02 và kiểm tra trực quan một vài sample đại diện.
