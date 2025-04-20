import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from PIL import Image
import tempfile
import torch  # Thêm import torch

# Thêm dòng này để cho phép tải các model của ultralytics
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

class DrowningDetectionApp:
    def __init__(self):
        # Available models
        self.models = {
            "Model 1 (best.pt)": "runs/drowning_detection/weights/best.pt",
            "Model 2 (best1.pt)": "runs/drowning_detection/weights/best1.pt",
            "Model 3 (best2.pt)": "runs/drowning_detection/weights/best2.pt",
            "Model 4 (blurry.pt)": "runs/drowning_detection/weights/blurry.pt", # blurry model
            "Model 5 (best3.pt)": "runs/drowning_detection/weights/best3.pt",
            "Model 6 (model.pt)": "runs/drowning_detection/weights/model.pt"    # best model by far
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
            # Hai cách tải model: 
            # 1. Sử dụng weights_only=False trong phiên bản PyTorch 2.6+
            try:
                # Cách 1: load model với cài đặt weights_only=False (không khuyên khích)
                # Chỉ sử dụng nếu bạn tin tưởng nguồn file model
                self.model = YOLO(model_path)
            except Exception as e:
                # Nếu cách 1 không thành công, thử cách 2
                st.warning("Trying alternative loading method...")
                # Cách 2: Sử dụng add_safe_globals() (được khuyên dùng)
                # Đã thêm ở đầu file: torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
                self.model = YOLO(model_path)
                
            st.success(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def process_video(self, video_file, conf_threshold, alert_conf, alert_time):
        """Process video for drowning detection"""
        if self.model is None:
            st.error("Please load a model first")
            return
        
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error opening video file")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Variables for drowning detection
        drowning_start_time = None
        current_time = time.time()
        alert_saved = False
        show_alert = False
        frame_count = 0
        
        # Streamlit placeholders
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        info_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        status_placeholder.info("Processing video...")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        drowning_metric = col1.empty()
        swimming_metric = col2.empty()
        tread_water_metric = col3.empty()
        
        # Display initial metrics
        drowning_metric.metric(label="Drowning", value=0)
        swimming_metric.metric(label="Swimming", value=0)
        tread_water_metric.metric(label="Tread Water", value=0)
        
        # Process the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)[0]
            
            # Process detection results
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
                
                # Get class name
                class_id = int(cls)
                class_name = self.model.names[class_id]
                
                # Add to lists
                boxes.append([x1, y1, x2, y2])
                classes.append(class_name)
                confidences.append(conf.item())
                
                # Count detections by class
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
            
            # Update metrics
            drowning_metric.metric(label="Drowning", value=drowning_count)
            swimming_metric.metric(label="Swimming", value=swimming_count)
            tread_water_metric.metric(label="Tread Water", value=tread_water_count)
            
            # Update drowning timer
            if high_conf_drowning:
                if drowning_start_time is None:
                    drowning_start_time = time.time()
                    alert_saved = False
            else:
                drowning_start_time = None
            
            # Calculate drowning duration
            drowning_duration = 0
            if drowning_start_time is not None:
                drowning_duration = time.time() - drowning_start_time
            
            # Determine if alert should be shown
            show_alert = drowning_start_time is not None and drowning_duration >= alert_time
            
            # Draw bounding boxes and alerts
            processed_frame = self.draw_bbox(frame, boxes, classes, confidences, show_alert)
            
            # Auto-save image when alert triggers and hasn't been saved yet
            if show_alert and not alert_saved:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.temp_dir, f"drowning_alert_{timestamp}.jpg")
                cv2.imwrite(save_path, processed_frame)
                alert_saved = True
                status_placeholder.warning(f"Alert! Image saved to {save_path}")
                
                # Display saved image in alert placeholder
                alert_img = Image.open(save_path)
                alert_placeholder.image(alert_img, caption="⚠️ DROWNING ALERT! ⚠️", use_column_width=True)
            
            # Add drowning timer indicator if detecting drowning
            if drowning_start_time is not None:
                timer_text = f"Drowning Timer: {drowning_duration:.1f}s / {alert_time:.1f}s"
                cv2.putText(processed_frame, timer_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show timer in status
                status_placeholder.warning(f"⚠️ Possible drowning detected! Timer: {drowning_duration:.1f}s / {alert_time:.1f}s")
            
            # Update info
            info_text = f"""
            ## Detection Results:
            - Drowning: {drowning_count}
            - Swimming: {swimming_count}
            - Tread Water: {tread_water_count}
            """
            if drowning_count > 0:
                info_text += f"- Max Drowning Confidence: {max_drowning_conf:.2f}\n"
            
            info_placeholder.markdown(info_text)
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, caption="Video Processing", use_column_width=True)
        
        # Clean up
        cap.release()
        os.unlink(tfile.name)
        status_placeholder.success("Video processing complete")
    
    def draw_bbox(self, frame, boxes, classes, confidences, is_drowning=False):
        """Draw bounding boxes on the frame"""
        out = frame.copy()
        
        # Colors for different classes (BGR format)
        class_colors = {
            "swimming": (0, 255, 0),      # Green
            "tread water": (0, 255, 255), # Yellow
            "drowning": (0, 0, 255)       # Red
        }
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            label = classes[i]
            
            # Get color based on class
            color = class_colors.get(label, (255, 0, 0))  # Default to blue
            
            # Draw box with thicker borders if drowning
            thickness = 5 if label == "drowning" else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            
            # Add a filled rectangle at the top for text background
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), color, -1)
            text = f"{label}: {confidence:.2f}"
            
            # Add text
            font_scale = 0.7
            cv2.putText(out, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Add warning text if drowning alert
        if is_drowning:
            height, width = out.shape[:2]
            warning_text = "DROWNING DETECTED!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            
            # Add warning text at the top
            cv2.putText(out, warning_text, (text_x, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Add flashing border around the frame
            border_thickness = int(3 + 2 * abs(np.sin(time.time() * 5)))
            cv2.rectangle(out, (5, 5), (width-5, height-5), (0, 0, 255), border_thickness)
        
        return out

def main():
    # Set page config
    st.set_page_config(
        page_title="Drowning Detection System",
        page_icon="🏊‍♂️",
        layout="wide"
    )
    
    # Title and description
    st.title("Drowning Detection System")
    st.markdown("""
    This application uses YOLO to detect drowning incidents in swimming videos.
    Upload a video and the system will automatically analyze it for possible drowning events.
    """)
    
    # Initialize app
    app = DrowningDetectionApp()
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            options=list(app.models.keys()),
            index=5  # Model 6 (model.pt) là mô hình tốt nhất
        )
        model_path = app.models[model_name]
        
        # Automatically load model if not already loaded
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                if app.load_model(model_path):
                    st.session_state.model_loaded = True
        
        # Manual model loading button
        if st.button("Load Selected Model"):
            with st.spinner("Loading model..."):
                if app.load_model(model_path):
                    st.session_state.model_loaded = True
                    st.session_state.video_processed = False  # Reset processed state
        
        st.divider()
        
        # Parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=app.conf_threshold,
            step=0.05
        )
        
        alert_conf = st.slider(
            "Alert Confidence",
            min_value=0.1,
            max_value=0.9,
            value=app.alert_conf,
            step=0.05
        )
        
        alert_time = st.slider(
            "Alert Time (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=app.alert_time,
            step=0.5
        )
        
        # Parameter changes reset video processed state
        if conf_threshold != app.conf_threshold or alert_conf != app.alert_conf or alert_time != app.alert_time:
            st.session_state.video_processed = False
    
    # Main content area
    # File uploader for video
    st.subheader("Upload Video")
    video_file = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"])
    
    if video_file is not None:
        # Check if it's a new video
        if st.session_state.current_video != video_file.name:
            st.session_state.current_video = video_file.name
            st.session_state.video_processed = False
        
        # Display file info
        file_details = {"Filename": video_file.name, "FileType": video_file.type, "FileSize": f"{video_file.size / (1024*1024):.2f} MB"}
        st.write(file_details)
        
        # Video preview
        st.video(video_file)
        
        # Automatically process if not processed yet
        if not st.session_state.video_processed:
            if app.model is None:
                st.error("Model not loaded. Please wait for the model to load or click 'Load Selected Model'.")
            else:
                with st.spinner("Automatically processing video..."):
                    # Need to reset the position of the video file
                    video_file.seek(0)
                    app.process_video(video_file, conf_threshold, alert_conf, alert_time)
                    st.session_state.video_processed = True
        
        # Optional manual processing button
        if st.button("Process Again"):
            st.session_state.video_processed = False
            video_file.seek(0)  # Reset file position
            with st.spinner("Processing video..."):
                app.process_video(video_file, conf_threshold, alert_conf, alert_time)
                st.session_state.video_processed = True

if __name__ == "__main__":
    main() 