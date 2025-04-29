from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import serial
import serial.tools.list_ports

class DrowningDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowning Detection Application")
        self.root.geometry("1280x800")
        
        # Các mô hình có sẵn
        self.models = {
            "Model 1 (best.pt)": "runs/drowning_detection/weights/best.pt",
            "Model 2 (best1.pt)": "runs/drowning_detection/weights/best1.pt",
            "Model 3 (best2.pt)": "runs/drowning_detection/weights/best2.pt",
            "Model 4 (blurry.pt)": "runs/drowning_detection/weights/blurry.pt", #mô hình mờ
            "Model 5 (best3.pt)": "runs/drowning_detection/weights/best3.pt",
            "Model 6 (model.pt)": "runs/drowning_detection/weights/model.pt"   #mô hình tốt nhất 
        }
        
        # Giá trị mặc định
        self.model_path = list(self.models.values())[0]
        self.conf_threshold = 0.25
        self.alert_conf = 0.65
        self.alert_time = 5.0
        
        # Biến xử lý video
        self.cap = None
        self.is_playing = False
        self.thread = None
        self.current_frame = None
        
        # Biến kết nối Arduino
        self.arduino = None
        self.arduino_connected = False
        self.arduino_ports = []
        self.previous_alert_state = False
        
        # Tạo giao diện
        self.create_widgets()
        
        # Quét cổng Arduino khi khởi động
        self.scan_arduino_ports()
        
    def create_widgets(self):
        # Khung chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bảng điều khiển (bên trái)
        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Lựa chọn mô hình
        ttk.Label(control_panel, text="Select Model:").pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar(value=list(self.models.keys())[0])
        model_dropdown = ttk.Combobox(control_panel, textvariable=self.model_var, 
                                      values=list(self.models.keys()), state="readonly", width=25)
        model_dropdown.pack(anchor=tk.W, pady=(0, 10))
        model_dropdown.bind("<<ComboboxSelected>>", self.update_model)
        
        # Thanh trượt ngưỡng độ tin cậy
        ttk.Label(control_panel, text=f"Confidence Threshold: {self.conf_threshold}").pack(anchor=tk.W, pady=(10, 5))
        self.conf_slider = ttk.Scale(control_panel, from_=0.1, to=0.9, length=200, 
                                    orient=tk.HORIZONTAL, value=self.conf_threshold,
                                    command=self.update_conf)
        self.conf_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Thanh trượt ngưỡng cảnh báo
        ttk.Label(control_panel, text=f"Alert Confidence: {self.alert_conf}").pack(anchor=tk.W, pady=(10, 5))
        self.alert_conf_slider = ttk.Scale(control_panel, from_=0.1, to=0.9, length=200, 
                                          orient=tk.HORIZONTAL, value=self.alert_conf,
                                          command=self.update_alert_conf)
        self.alert_conf_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Thanh trượt thời gian cảnh báo
        ttk.Label(control_panel, text=f"Alert Time (seconds): {self.alert_time}").pack(anchor=tk.W, pady=(10, 5))
        self.alert_time_slider = ttk.Scale(control_panel, from_=1.0, to=10.0, length=200, 
                                          orient=tk.HORIZONTAL, value=self.alert_time,
                                          command=self.update_alert_time)
        self.alert_time_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Phần Arduino
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_panel, text="Arduino Connection", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 5))
        
        # Lựa chọn cổng Arduino
        ttk.Label(control_panel, text="Select Arduino Port:").pack(anchor=tk.W, pady=(5, 5))
        self.arduino_port_var = tk.StringVar(value="Not connected")
        self.arduino_dropdown = ttk.Combobox(control_panel, textvariable=self.arduino_port_var, 
                                           state="readonly", width=25)
        self.arduino_dropdown.pack(anchor=tk.W, pady=(0, 5))
        
        # Nút điều khiển kết nối Arduino
        arduino_button_frame = ttk.Frame(control_panel)
        arduino_button_frame.pack(anchor=tk.W, pady=(0, 10), fill=tk.X)
        
        self.connect_btn = ttk.Button(arduino_button_frame, text="Connect", command=self.connect_arduino)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.scan_btn = ttk.Button(arduino_button_frame, text="Scan Ports", command=self.scan_arduino_ports)
        self.scan_btn.pack(side=tk.LEFT)
        
        # Trạng thái Arduino
        self.arduino_status_var = tk.StringVar(value="Arduino Status: Not connected")
        ttk.Label(control_panel, textvariable=self.arduino_status_var).pack(anchor=tk.W, pady=(0, 10))
        
        # Điều khiển video
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Nút chọn video
        self.select_video_btn = ttk.Button(control_panel, text="Select Video", command=self.select_video)
        self.select_video_btn.pack(anchor=tk.W, pady=(10, 5), fill=tk.X)
        
        # Nút phát/tạm dừng
        self.play_btn = ttk.Button(control_panel, text="Play Video", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Nút dừng
        self.stop_btn = ttk.Button(control_panel, text="Stop Video", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Nút lưu kết quả
        self.save_btn = ttk.Button(control_panel, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Trạng thái
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.status_var = tk.StringVar(value="Ready. Please select a video file.")
        ttk.Label(control_panel, textvariable=self.status_var, wraplength=200).pack(anchor=tk.W, pady=10)
        
        # Thông tin phát hiện
        self.info_text = tk.Text(control_panel, height=10, width=25, state=tk.DISABLED)
        self.info_text.pack(anchor=tk.W, pady=(10, 0), fill=tk.BOTH, expand=True)
        
        # Hiển thị video (bên phải)
        self.video_frame = ttk.LabelFrame(main_frame, text="Video", padding=10)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas cho hiển thị video
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Tải mô hình ban đầu
        self.load_model()
    
    def scan_arduino_ports(self):
        """Quét các cổng COM có sẵn cho thiết bị Arduino"""
        self.arduino_ports = []
        available_ports = list(serial.tools.list_ports.comports())
        
        if not available_ports:
            self.arduino_dropdown.config(values=["No ports available"])
            self.arduino_port_var.set("No ports available")
            return
        
        # Tạo danh sách tên cổng
        port_names = []
        for port in available_ports:
            port_name = f"{port.device} - {port.description}"
            port_names.append(port_name)
            self.arduino_ports.append(port.device)
        
        # Cập nhật giá trị dropdown
        self.arduino_dropdown.config(values=port_names)
        self.arduino_port_var.set(port_names[0] if port_names else "No ports available")
        
        # Cập nhật trạng thái
        self.status_var.set(f"Found {len(port_names)} serial port(s)")
    
    def connect_arduino(self):
        """Kết nối với cổng Arduino đã chọn"""  # Import here for debugging
       

        if not self.arduino_ports:
            self.show_error("No Arduino ports available")
            return
    
        selected_index = self.arduino_dropdown.current()
        if selected_index < 0 or selected_index >= len(self.arduino_ports):
            self.show_error("Please select a valid port")
            return
    
        selected_port = self.arduino_ports[selected_index]
    
        # Đóng kết nối hiện có nếu đang mở
        if self.arduino is not None and self.arduino.is_open:
            try:
                self.arduino.close()
            except Exception as e:
                self.show_error(f"Error closing previous connection: {str(e)}")
    
    # Thử mở kết nối mới
        try:
            self.arduino = serial.Serial(selected_port, 9600, timeout=1)
            time.sleep(2)  # Đợi Arduino khởi động lại
            self.arduino_connected = True
            self.arduino_status_var.set(f"Arduino Status: Connected to {selected_port}")
            self.connect_btn.config(text="Disconnect")
            self.connect_btn.config(command=self.disconnect_arduino)
            self.status_var.set(f"Successfully connected to Arduino on {selected_port}")
        except Exception as e:
            self.show_error(f"Failed to connect to Arduino: {str(e)}")
            self.arduino_connected = False
            self.arduino = None
    
    def disconnect_arduino(self):
        """Ngắt kết nối với Arduino"""
        if self.arduino is not None and self.arduino.is_open:
            try:
                # Gửi lệnh dừng cảnh báo trước khi ngắt kết nối
                self.send_arduino_command("STOP_ALERT")
                time.sleep(0.5)
                self.arduino.close()
                self.arduino_connected = False
                self.arduino_status_var.set("Arduino Status: Not connected")
                self.connect_btn.config(text="Connect")
                self.connect_btn.config(command=self.connect_arduino)
                self.status_var.set("Arduino disconnected")
            except Exception as e:
                self.show_error(f"Error disconnecting from Arduino: {str(e)}")
        else:
            self.arduino_connected = False
            self.arduino_status_var.set("Arduino Status: Not connected")
            self.connect_btn.config(text="Connect")
            self.connect_btn.config(command=self.connect_arduino)
    
    def send_arduino_command(self, command):
        """Gửi lệnh đến Arduino"""
        if not self.arduino_connected or self.arduino is None:
            return
        
        try:
            self.arduino.write(f"{command}\n".encode())
            # Tuỳ chọn: Đọc phản hồi từ Arduino
            # response = self.arduino.readline().decode().strip()
            # print(f"Arduino Response: {response}")
        except Exception as e:
            self.show_error(f"Error sending command to Arduino: {str(e)}")
            self.arduino_connected = False
            self.arduino_status_var.set("Arduino Status: Connection lost")
            self.connect_btn.config(text="Connect")
            self.connect_btn.config(command=self.connect_arduino)
    
    def update_model(self, event=None):
        model_name = self.model_var.get()
        self.model_path = self.models[model_name]
        self.load_model()
    
    def update_conf(self, value):
        self.conf_threshold = float(value)
        # Cập nhật nhãn để hiển thị giá trị hiện tại
        for child in self.conf_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Confidence Threshold" in child.cget("text"):
                child.config(text=f"Confidence Threshold: {self.conf_threshold:.2f}")
    
    def update_alert_conf(self, value):
        self.alert_conf = float(value)
        # Cập nhật nhãn để hiển thị giá trị hiện tại
        for child in self.alert_conf_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Alert Confidence" in child.cget("text"):
                child.config(text=f"Alert Confidence: {self.alert_conf:.2f}")
    
    def update_alert_time(self, value):
        self.alert_time = float(value)
        # Cập nhật nhãn để hiển thị giá trị hiện tại
        for child in self.alert_time_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Alert Time" in child.cget("text"):
                child.config(text=f"Alert Time (seconds): {self.alert_time:.2f}")
    
    def load_model(self):
        self.status_var.set(f"Loading model from {self.model_path}...")
        
        # Vô hiệu hóa các nút trong quá trình tải
        self.select_video_btn.config(state=tk.DISABLED)
        
        # Tải mô hình trong một luồng riêng biệt
        def load_model_thread():
            try:
                self.model = YOLO(self.model_path)
                # Lên lịch cập nhật GUI trên luồng chính
                self.root.after(0, self.model_loaded)
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.root.after(0, lambda msg=error_msg: self.show_error(msg))
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def model_loaded(self):
        self.status_var.set(f"Model loaded successfully. Please select a video file.")
        self.select_video_btn.config(state=tk.NORMAL)
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"Video selected: {os.path.basename(file_path)}")
            
            # Mở video
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.show_error(f"Error opening video: {file_path}")
                return
            
            # Hiển thị khung hình đầu tiên
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                
                # Kích hoạt nút phát
                self.play_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
            else:
                self.show_error("Could not read video frame")
    
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play Video")
            
            # Dừng cảnh báo nếu video tạm dừng
            if self.arduino_connected:
                self.send_arduino_command("STOP_ALERT")
                self.previous_alert_state = False
        else:
            self.is_playing = True
            self.play_btn.config(text="Pause Video")
            
            # Bắt đầu xử lý video trong một luồng nếu chưa chạy
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.process_video, daemon=True)
                self.thread.start()
    
    def stop_video(self):
        self.is_playing = False
        self.play_btn.config(text="Play Video")
        
        # Dừng cảnh báo khi video dừng
        if self.arduino_connected:
            self.send_arduino_command("STOP_ALERT")
            self.previous_alert_state = False
        
        # Đặt lại video về đầu
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
    
    def save_result(self):
        if self.current_frame is None:
            self.show_error("No frame to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_frame)
                self.status_var.set(f"Frame saved to {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"Error saving frame: {str(e)}")
    
    def process_video(self):
        if self.cap is None:
            return
            
        # Biến cho phát hiện đuối nước
        drowning_start_time = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Đặt lại video nếu ở cuối
        if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        self.status_var.set("Processing video...")
        
        # Cho tự động lưu hình ảnh
        alert_saved = False
        save_dir = "drowning_alerts"
        os.makedirs(save_dir, exist_ok=True)
        
        while self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                # Kết thúc video
                self.is_playing = False
                self.play_btn.config(text="Play Video")
                self.status_var.set("End of video")
                self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                
                # Dừng cảnh báo Arduino ở cuối video
                if self.arduino_connected:
                    self.send_arduino_command("STOP_ALERT")
                    self.previous_alert_state = False
                break
                
            # Chạy phát hiện
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            
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
                
                # Đếm số lượng phát hiện theo lớp
                if class_name == "drowning":
                    drowning_count += 1
                    conf_value = conf.item()
                    max_drowning_conf = max(max_drowning_conf, conf_value)
                    if conf_value > self.alert_conf:
                        is_drowning_detected = True
                        high_conf_drowning = True
                elif class_name == "swimming":
                    swimming_count += 1
                elif class_name == "tread water":
                    tread_water_count += 1
            
            # Cập nhật bộ hẹn giờ đuối nước
            current_time = time.time()
            if high_conf_drowning:
                if drowning_start_time is None:
                    drowning_start_time = current_time
                    alert_saved = False  # Đặt lại cờ đã lưu khi phát hiện đuối nước mới
            else:
                drowning_start_time = None
            
            # Tính thời gian đuối nước
            drowning_duration = 0
            if drowning_start_time is not None:
                drowning_duration = current_time - drowning_start_time
            
            # Xác định xem có nên hiển thị cảnh báo hay không
            show_alert = drowning_start_time is not None and drowning_duration >= self.alert_time
            
            # Cập nhật Arduino với trạng thái cảnh báo
            if self.arduino_connected and show_alert != self.previous_alert_state:
                if show_alert:
                    self.send_arduino_command("DROWNING_ALERT")
                else:
                    self.send_arduino_command("STOP_ALERT")
                self.previous_alert_state = show_alert
            
            # Vẽ hộp giới hạn và cảnh báo
            processed_frame = self.draw_bbox(frame, boxes, classes, confidences, show_alert)
            
            # Tự động lưu hình ảnh khi kích hoạt cảnh báo và chưa được lưu
            if show_alert and not alert_saved:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"drowning_alert_{timestamp}.jpg")
                cv2.imwrite(save_path, processed_frame)
                alert_saved = True
                self.status_var.set(f"Alert! Image saved to {save_path}")
            
            # Thêm chỉ báo hẹn giờ đuối nước nếu phát hiện đuối nước
            if drowning_start_time is not None:
                timer_text = f"Drowning Timer: {drowning_duration:.1f}s / {self.alert_time:.1f}s"
                cv2.putText(processed_frame, timer_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Cập nhật thông tin phát hiện
            self.root.after(0, lambda: self.update_info(drowning_count, swimming_count, tread_water_count, 
                                                       max_drowning_conf, show_alert))
            
            # Lưu khung hình hiện tại
            self.current_frame = processed_frame
            
            # Hiển thị khung hình
            self.root.after(0, lambda f=processed_frame: self.display_frame(f))
            
            # Kích hoạt nút lưu
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
            # Độ trễ nhẹ để không quá tải hệ thống
            time.sleep(0.01)
    
    def draw_bbox(self, frame, boxes, classes, confidences, is_drowning=False):
        """Vẽ các hộp giới hạn trên khung hình"""
        out = frame.copy()
        
        # Màu sắc cho các lớp khác nhau (định dạng BGR)
        class_colors = {
            "swimming": (0, 255, 0),      # Xanh lá
            "tread water": (0, 255, 255), # Vàng
            "drowning": (0, 0, 255)       # Đỏ
        }
        
        # Vẽ hộp giới hạn
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            label = classes[i]
            
            # Lấy màu dựa trên lớp
            color = class_colors.get(label, (255, 0, 0))  # Mặc định là xanh dương
            
            # Vẽ hộp với đường viền dày hơn nếu đuối nước
            thickness = 5 if label == "drowning" else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            
            # Thêm một hình chữ nhật đầy màu ở phía trên cho nền văn bản
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), color, -1)
            text = f"{label}: {confidence:.2f}"
            
            # Thêm văn bản
            font_scale = 0.7
            cv2.putText(out, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Thêm văn bản cảnh báo nếu có cảnh báo đuối nước
        if is_drowning:
            height, width = out.shape[:2]
            warning_text = "DROWNING DETECTED!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            
            # Thêm văn bản cảnh báo ở phía trên
            cv2.putText(out, warning_text, (text_x, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Thêm đường viền nhấp nháy xung quanh khung hình
            border_thickness = int(3 + 2 * abs(np.sin(time.time() * 5)))
            cv2.rectangle(out, (5, 5), (width-5, height-5), (0, 0, 255), border_thickness)
        
        return out
    
    def update_info(self, drowning, swimming, tread_water, max_drowning_conf, alert):
        """Cập nhật hộp văn bản thông tin"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, f"Detection Results:\n\n")
        self.info_text.insert(tk.END, f"Drowning: {drowning}\n")
        self.info_text.insert(tk.END, f"Swimming: {swimming}\n")
        self.info_text.insert(tk.END, f"Tread Water: {tread_water}\n\n")
        
        if drowning > 0:
            self.info_text.insert(tk.END, f"Max Drowning Conf: {max_drowning_conf:.2f}\n")
        
        if alert:
            self.info_text.insert(tk.END, "\n⚠️ DROWNING ALERT! ⚠️\n", "alert")
            self.info_text.tag_configure("alert", foreground="red", font=("Arial", 12, "bold"))
            
            # Thêm trạng thái Arduino nếu đã kết nối
            if self.arduino_connected:
                self.info_text.insert(tk.END, "\nArduino alarm activated!", "arduino")
                self.info_text.tag_configure("arduino", foreground="blue", font=("Arial", 10, "bold"))
        
        self.info_text.config(state=tk.DISABLED)
    
    def display_frame(self, frame):
        """Hiển thị khung hình trên canvas"""
        if frame is None:
            return
            
        # Chuyển đổi từ BGR sang RGB cho tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Thay đổi kích thước để phù hợp với canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Tính tỉ lệ khung hình
            frame_height, frame_width = rgb_frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas rộng hơn mức cần thiết
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas cao hơn mức cần thiết
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Thay đổi kích thước khung hình
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Tạo một hình ảnh PIL và sau đó là một PhotoImage
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Cập nhật canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
            self.canvas.image = img_tk  # Giữ tham chiếu để ngăn việc thu gom rác
    
    def show_error(self, message):
        """Hiển thị thông báo lỗi"""
        messagebox.showerror("Error", message)
        self.status_var.set(f"Error: {message}")

def main():
    root = tk.Tk()
    app = DrowningDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
