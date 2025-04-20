import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import tempfile
import torch
import sys
import yaml
from pathlib import Path

# Cài đặt trang
st.set_page_config(
    page_title="Drowning Detection System",
    page_icon="🏊‍♂️",
    layout="wide"
)

# Đặt biến môi trường để tắt weights_only
os.environ["TORCH_WEIGHTS_ONLY"] = "0"

# Kiểm tra và cài đặt thư viện
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics không được cài đặt. Đang cài đặt...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics
    from ultralytics import YOLO

# Thêm các module cần thiết vào danh sách an toàn
try:
    import torch.nn.modules.container
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential
    ])
except Exception as e:
    st.warning(f"Không thể thiết lập safe_globals: {e}")
    
class DrowningDetectionApp:
    def __init__(self):
        # Available models
        self.models = {
            "Model 1 (best.pt)": "runs/drowning_detection/weights/best.pt",
            "Model 2 (best1.pt)": "runs/drowning_detection/weights/best1.pt",
            "Model 3 (best2.pt)": "runs/drowning_detection/weights/best2.pt", #best model by far
            "Model 4 (blurry.pt)": "runs/drowning_detection/weights/blurry.pt", #blurry model
            "Model 5 (best3.pt)": "runs/drowning_detection/weights/best3.pt",
            "Model 6 (model.pt)": "runs/drowning_detection/weights/model.pt"
        }
        
        # Default values
        self.conf_threshold = 0.25
        self.alert_conf = 0.65
        self.alert_time = 5.0
        self.model = None
        self.temp_dir = "drowning_alerts"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def load_model(self, model_path):
        """Load the selected YOLO model"""
        try:
            st.write(f"Đang tải model từ: {model_path}")
            
            # Kiểm tra xem file có tồn tại
            if not os.path.exists(model_path):
                st.error(f"File model không tồn tại: {model_path}")
                # Liệt kê các file trong thư mục
                model_dir = os.path.dirname(model_path)
                if os.path.exists(model_dir):
                    st.write(f"Các file có trong thư mục {model_dir}:")
                    for file in os.listdir(model_dir):
                        st.write(f"- {file}")
                return False
            
            # Cố gắng tải model với nhiều phương pháp khác nhau
            try:
                # Phương pháp 1: Sử dụng context manager
                with torch.serialization.safe_globals([
                    ultralytics.nn.tasks.DetectionModel,
                    torch.nn.modules.container.Sequential
                ]):
                    self.model = YOLO(model_path)
                st.success("Tải model thành công (phương pháp 1)")
                return True
                
            except Exception as e1:
                st.warning(f"Phương pháp 1 thất bại: {e1}")
                
                # Phương pháp 2: Sử dụng monkey patch
                try:
                    # Lưu hàm gốc
                    original_torch_load = torch.load
                    
                    # Tạo hàm patch
                    def patched_torch_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_torch_load(*args, **kwargs)
                    
                    # Áp dụng patch
                    torch.load = patched_torch_load
                    
                    # Tải model
                    self.model = YOLO(model_path)
                    
                    # Khôi phục hàm gốc
                    torch.load = original_torch_load
                    
                    st.success("Tải model thành công (phương pháp 2)")
                    return True
                    
                except Exception as e2:
                    st.warning(f"Phương pháp 2 thất bại: {e2}")
                    
                    # Phương pháp 3: Sử dụng biến môi trường
                    try:
                        os.environ["TORCH_WEIGHTS_ONLY"] = "0"
                        self.model = YOLO(model_path)
                        st.success("Tải model thành công (phương pháp 3)")
                        return True
                    except Exception as e3:
                        st.error(f"Phương pháp 3 thất bại: {e3}")
                        return False
                        
        except Exception as e:
            st.error(f"Lỗi tải model: {str(e)}")
            return False
    
    def process_video(self, video_file, conf_threshold, alert_conf, alert_time):
        """Process video for drowning detection"""
        if self.model is None:
            st.error("Vui lòng tải model trước khi xử lý video")
            return
        
        # Tạo file tạm để lưu video tải lên
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()
        
        # Mở video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Không thể mở file video")
            return
        
        # Lấy thông số video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Biến theo dõi phát hiện đuối nước
        drowning_start_time = None
        current_time = time.time()
        alert_saved = False
        show_alert = False
        frame_count = 0
        
        # Tạo các placeholders cho Streamlit
        progress_bar = st.progress(0)
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        info_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        status_placeholder.info("Đang xử lý video...")
        
        # Tạo cột cho các chỉ số
        col1, col2, col3 = st.columns(3)
        drowning_metric = col1.empty()
        swimming_metric = col2.empty()
        tread_water_metric = col3.empty()
        
        # Hiển thị giá trị ban đầu
        drowning_metric.metric(label="Đuối nước", value=0)
        swimming_metric.metric(label="Bơi", value=0)
        tread_water_metric.metric(label="Vùng vẫy", value=0)
        
        # Xử lý video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress_value = min(float(frame_count) / total_frames, 1.0)
            progress_bar.progress(progress_value)
            
            # Chạy phát hiện
            results = self.model(frame, conf=conf_threshold, verbose=False)[0]
            
            # Xử lý kết quả phát hiện
            boxes = []
            classes = []
            confidences = []
            is_drowning_detected = False
            high_conf_drowning = False
            drowning_count = 0
            swimming_count = 0
            tread_water_count = 0
            max_drowning_conf = 0
            
            for detection in results.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Lấy tên lớp
                class_id = int(cls)
                class_name = self.model.names[class_id]
                
                # Thêm vào danh sách
                boxes.append([x1, y1, x2, y2])
                classes.append(class_name)
                confidences.append(conf.item())
                
                # Đếm theo từng lớp
                if class_name == "drowning":
                    drowning_count += 1
                    conf_value = conf.item()
                    max_drowning_conf = max(max_drowning_conf, conf_value)
                    if conf_value > alert_conf:
                        is_drowning_detected = True
                        high_conf_drowning = True
                elif class_name == "swimming":
                    swimming_count += 1
                elif class_name == "tread water":
                    tread_water_count += 1
            
            # Cập nhật số liệu
            drowning_metric.metric(label="Đuối nước", value=drowning_count)
            swimming_metric.metric(label="Bơi", value=swimming_count)
            tread_water_metric.metric(label="Vùng vẫy", value=tread_water_count)
            
            # Cập nhật bộ đếm thời gian đuối nước
            if high_conf_drowning:
                if drowning_start_time is None:
                    drowning_start_time = time.time()
                    alert_saved = False
            else:
                drowning_start_time = None
            
            # Tính thời gian đuối nước
            drowning_duration = 0
            if drowning_start_time is not None:
                drowning_duration = time.time() - drowning_start_time
            
            # Xác định xem có nên hiển thị cảnh báo không
            show_alert = drowning_start_time is not None and drowning_duration >= alert_time
            
            # Vẽ bounding box và cảnh báo
            processed_frame = self.draw_bbox(frame, boxes, classes, confidences, show_alert)
            
            # Tự động lưu hình ảnh khi phát hiện đuối nước
            if show_alert and not alert_saved:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.temp_dir, f"drowning_alert_{timestamp}.jpg")
                cv2.imwrite(save_path, processed_frame)
                alert_saved = True
                status_placeholder.warning(f"Cảnh báo! Đã lưu hình ảnh: {save_path}")
                
                # Hiển thị hình ảnh đã lưu
                alert_img = Image.open(save_path)
                alert_placeholder.image(alert_img, caption="⚠️ PHÁT HIỆN ĐUỐI NƯỚC! ⚠️", use_column_width=True)
            
            # Thêm chỉ báo bộ đếm thời gian đuối nước
            if drowning_start_time is not None:
                timer_text = f"Thời gian đuối nước: {drowning_duration:.1f}s / {alert_time:.1f}s"
                cv2.putText(processed_frame, timer_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Hiển thị bộ đếm thời gian trong trạng thái
                status_placeholder.warning(f"⚠️ Có thể đang đuối nước! Thời gian: {drowning_duration:.1f}s / {alert_time:.1f}s")
            
            # Cập nhật thông tin
            info_text = f"""
            ## Kết quả phát hiện:
            - Đuối nước: {drowning_count}
            - Bơi: {swimming_count}
            - Vùng vẫy: {tread_water_count}
            """
            if drowning_count > 0:
                info_text += f"- Độ tin cậy đuối nước cao nhất: {max_drowning_conf:.2f}\n"
            
            info_placeholder.markdown(info_text)
            
            # Chuyển đổi sang RGB để hiển thị
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, caption="Đang xử lý video", use_column_width=True)
        
        # Dọn dẹp
        cap.release()
        os.unlink(tfile.name)
        status_placeholder.success("Xử lý video hoàn tất")
    
    def draw_bbox(self, frame, boxes, classes, confidences, is_drowning=False):
        """Vẽ bounding box lên khung hình"""
        out = frame.copy()
        
        # Màu sắc cho các lớp khác nhau (định dạng BGR)
        class_colors = {
            "swimming": (0, 255, 0),      # Xanh lá
            "tread water": (0, 255, 255), # Vàng
            "drowning": (0, 0, 255)       # Đỏ
        }
        
        # Vẽ bounding box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            label = classes[i]
            
            # Lấy màu dựa vào lớp
            color = class_colors.get(label, (255, 0, 0))  # Mặc định là xanh dương
            
            # Vẽ box với đường viền dày hơn nếu là đuối nước
            thickness = 5 if label == "drowning" else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            
            # Thêm hình chữ nhật đầy màu ở phía trên cho nền văn bản
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), color, -1)
            
            # Chuyển đổi tên lớp sang tiếng Việt
            if label == "drowning":
                vn_label = "ĐUỐI NƯỚC"
            elif label == "swimming":
                vn_label = "BƠI"
            elif label == "tread water":
                vn_label = "VÙNG VẪY"
            else:
                vn_label = label
                
            text = f"{vn_label}: {confidence:.2f}"
            
            # Thêm văn bản
            font_scale = 0.7
            cv2.putText(out, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Thêm văn bản cảnh báo nếu phát hiện đuối nước
        if is_drowning:
            height, width = out.shape[:2]
            warning_text = "PHÁT HIỆN ĐUỐI NƯỚC!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            
            # Thêm văn bản cảnh báo ở trên cùng
            cv2.putText(out, warning_text, (text_x, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Thêm đường viền nhấp nháy xung quanh khung hình
            border_thickness = int(3 + 2 * abs(np.sin(time.time() * 5)))
            cv2.rectangle(out, (5, 5), (width-5, height-5), (0, 0, 255), border_thickness)
        
        return out

def main():
    # Tiêu đề và mô tả
    st.title("Hệ thống phát hiện đuối nước - DROWNING DETECTION")
    st.markdown("""
    **Ứng dụng này sử dụng YOLO để phát hiện các trường hợp đuối nước trong video bơi lội.**
    
    Tải lên video và hệ thống sẽ tự động phân tích để phát hiện người đuối nước.
    """)
    
    # Khởi tạo ứng dụng
    app = DrowningDetectionApp()
    
    # Khởi tạo session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    # Sidebar cho lựa chọn model và tham số
    with st.sidebar:
        st.header("Cài đặt")
        
        # Lựa chọn model
        model_name = st.selectbox(
            "Chọn Model",
            options=list(app.models.keys()),
            index=2  # Model 3 (best2.pt) là model tốt nhất
        )
        model_path = app.models[model_name]
        
        # Hiển thị phiên bản
        st.markdown("---")
        st.caption(f"PyTorch version: {torch.__version__}")
        st.caption(f"Ultralytics version: {ultralytics.__version__}")
        
        # Tự động tải model nếu chưa tải
        if not st.session_state.model_loaded:
            with st.spinner("Đang tải model..."):
                if app.load_model(model_path):
                    st.session_state.model_loaded = True
        
        # Nút tải model thủ công
        if st.button("Tải lại Model"):
            with st.spinner("Đang tải model..."):
                if app.load_model(model_path):
                    st.session_state.model_loaded = True
                    st.session_state.video_processed = False  # Reset trạng thái xử lý
        
        st.divider()
        
        # Các tham số
        st.subheader("Tham số phát hiện")
        conf_threshold = st.slider(
            "Ngưỡng tin cậy (Confidence)",
            min_value=0.1,
            max_value=0.9,
            value=app.conf_threshold,
            step=0.05,
            help="Ngưỡng tối thiểu để phát hiện đối tượng"
        )
        
        alert_conf = st.slider(
            "Ngưỡng cảnh báo (Alert)",
            min_value=0.1,
            max_value=0.9,
            value=app.alert_conf,
            step=0.05,
            help="Ngưỡng tin cậy để kích hoạt cảnh báo đuối nước"
        )
        
        alert_time = st.slider(
            "Thời gian cảnh báo (giây)",
            min_value=1.0,
            max_value=10.0,
            value=app.alert_time,
            step=0.5,
            help="Thời gian tối thiểu phát hiện đuối nước liên tục trước khi cảnh báo"
        )
        
        # Thay đổi tham số reset trạng thái đã xử lý video
        if conf_threshold != app.conf_threshold or alert_conf != app.alert_conf or alert_time != app.alert_time:
            app.conf_threshold = conf_threshold
            app.alert_conf = alert_conf
            app.alert_time = alert_time
            st.session_state.video_processed = False
    
    # Khu vực nội dung chính
    # Tải lên video
    st.subheader("Tải lên video")
    video_file = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"])
    
    if video_file is not None:
        # Kiểm tra xem có phải video mới không
        if st.session_state.current_video != video_file.name:
            st.session_state.current_video = video_file.name
            st.session_state.video_processed = False
        
        # Hiển thị thông tin file
        file_details = {"Tên file": video_file.name, "Loại file": video_file.type, "Kích thước": f"{video_file.size / (1024*1024):.2f} MB"}
        st.write(file_details)
        
        # Xem trước video
        st.video(video_file)
        
        # Tự động xử lý nếu chưa xử lý
        if not st.session_state.video_processed:
            if app.model is None:
                st.error("Model chưa được tải. Vui lòng đợi model tải xong hoặc nhấn 'Tải lại Model'.")
            else:
                with st.spinner("Đang tự động xử lý video..."):
                    # Cần reset vị trí của file video
                    video_file.seek(0)
                    app.process_video(video_file, conf_threshold, alert_conf, alert_time)
                    st.session_state.video_processed = True
        
        # Nút xử lý lại
        if st.button("Xử lý lại"):
            st.session_state.video_processed = False
            video_file.seek(0)  # Reset vị trí file
            with st.spinner("Đang xử lý video..."):
                app.process_video(video_file, conf_threshold, alert_conf, alert_time)
                st.session_state.video_processed = True
    
    # Hiển thị phần hướng dẫn
    with st.expander("Hướng dẫn sử dụng"):
        st.markdown("""
        ### Cách sử dụng:
        1. **Chọn model** từ thanh bên trái (Model 3 - best2.pt là model tốt nhất)
        2. **Điều chỉnh các tham số** phát hiện:
           - **Ngưỡng tin cậy**: Đặt thấp (0.25) để phát hiện nhiều đối tượng hơn, đặt cao (0.5+) cho độ chính xác cao hơn
           - **Ngưỡng cảnh báo**: Đặt cao (0.65+) để tránh cảnh báo giả
           - **Thời gian cảnh báo**: Thời gian cần phát hiện đuối nước liên tục trước khi cảnh báo
        3. **Tải lên video** để phân tích
        4. Hệ thống sẽ tự động xử lý và hiển thị kết quả
        
        ### Ý nghĩa màu sắc:
        - **Xanh lá**: Người đang bơi bình thường
        - **Vàng**: Người đang vùng vẫy/bơi tại chỗ
        - **Đỏ**: Người đang đuối nước
        
        ### Khi có cảnh báo:
        - Viền đỏ nhấp nháy sẽ xuất hiện quanh khung hình
        - Thông báo "PHÁT HIỆN ĐUỐI NƯỚC!" được hiển thị
        - Hình ảnh sẽ được tự động lưu
        """)
        
    # Footer
    st.markdown("---")
    st.caption("© Hệ thống phát hiện đuối nước - Sử dụng YOLOv8 cho phát hiện đuối nước trong thời gian thực")

if __name__ == "__main__":
    main() 