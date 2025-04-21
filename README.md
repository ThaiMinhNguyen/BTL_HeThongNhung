# Hệ Thống Phát Hiện Đuối Nước với Arduino

Hệ thống phát hiện đuối nước sử dụng mô hình YOLOv8 kết hợp với Arduino UNO R3 và còi báo động để cảnh báo khi phát hiện sự cố đuối nước.

## Thành phần cần thiết

- Arduino UNO R3
- Còi báo động (Active Buzzer)
- Dây cáp USB để kết nối Arduino
- Máy tính cài đặt Python 3.8+

## Hướng dẫn cài đặt

### 1. Cài đặt thư viện Python

```bash
# Tạo môi trường ảo (tùy chọn nhưng khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Tải lên mã Arduino

1. Mở Arduino IDE
2. Tải tệp `drowning_detection_arduino.ino`
3. Kết nối Arduino UNO R3 với máy tính qua cổng USB
4. Chọn board Arduino UNO và cổng COM trong Arduino IDE
5. Tải mã lên Arduino

### 3. Kết nối phần cứng

1. Kết nối còi báo động với Arduino:
   - Chân dương (thường dài hơn) của còi kết nối với chân số 9 trên Arduino
   - Chân âm (thường ngắn hơn) của còi kết nối với chân GND (đất) trên Arduino

## Chạy ứng dụng

```bash
python run.py
```

## Hướng dẫn sử dụng

1. **Chọn mô hình**: Chọn một trong các mô hình phát hiện đuối nước có sẵn
2. **Kết nối Arduino**: Nhấn "Scan Ports" để tìm Arduino, chọn và nhấn "Connect"
3. **Chọn video**: Nhấn "Select Video" để chọn file video phân tích
4. **Bắt đầu phát hiện**: Nhấn "Play Video" để bắt đầu phát hiện

## Cấu hình hệ thống

Bạn có thể điều chỉnh các thông số trong ứng dụng:

- **Confidence Threshold**: Ngưỡng tin cậy tối thiểu để phát hiện đối tượng (0.1-0.9)
  - Giá trị thấp: Phát hiện nhiều đối tượng nhưng có thể có nhiều cảnh báo sai
  - Giá trị cao: Ít phát hiện hơn nhưng chính xác hơn

- **Alert Confidence**: Ngưỡng tin cậy để kích hoạt cảnh báo đuối nước (0.1-0.9)
  - Chỉ khi phát hiện đuối nước có độ tin cậy cao hơn giá trị này, hệ thống mới tính thời gian để kích hoạt cảnh báo

- **Alert Time**: Thời gian phát hiện đuối nước liên tục trước khi kích hoạt cảnh báo (giây)

## Xử lý sự cố

- **Không tìm thấy Arduino**: Đảm bảo đã kết nối đúng và cài đặt driver
- **Không có âm thanh còi**: Kiểm tra kết nối dây và đảm bảo còi hoạt động
- **Lỗi cổng COM**: Đóng các ứng dụng khác có thể đang sử dụng cổng COM 