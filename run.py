from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading

class DrowningDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowning Detection Application")
        self.root.geometry("1280x800")
        
        # Available models
        self.models = {
            "Model 1 (best.pt)": "runs/drowning_detection/weights/best.pt",
            "Model 2 (best1.pt)": "runs/drowning_detection/weights/best1.pt",
            "Model 3 (best2.pt)": "runs/drowning_detection/weights/best2.pt"
        }
        
        # Default values
        self.model_path = list(self.models.values())[0]
        self.conf_threshold = 0.25
        self.alert_conf = 0.65
        self.alert_time = 5.0
        
        # Video processing variables
        self.cap = None
        self.is_playing = False
        self.thread = None
        self.current_frame = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (left side)
        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Model selection
        ttk.Label(control_panel, text="Select Model:").pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar(value=list(self.models.keys())[0])
        model_dropdown = ttk.Combobox(control_panel, textvariable=self.model_var, 
                                      values=list(self.models.keys()), state="readonly", width=25)
        model_dropdown.pack(anchor=tk.W, pady=(0, 10))
        model_dropdown.bind("<<ComboboxSelected>>", self.update_model)
        
        # Confidence threshold slider
        ttk.Label(control_panel, text=f"Confidence Threshold: {self.conf_threshold}").pack(anchor=tk.W, pady=(10, 5))
        self.conf_slider = ttk.Scale(control_panel, from_=0.1, to=0.9, length=200, 
                                    orient=tk.HORIZONTAL, value=self.conf_threshold,
                                    command=self.update_conf)
        self.conf_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Alert confidence threshold slider
        ttk.Label(control_panel, text=f"Alert Confidence: {self.alert_conf}").pack(anchor=tk.W, pady=(10, 5))
        self.alert_conf_slider = ttk.Scale(control_panel, from_=0.1, to=0.9, length=200, 
                                          orient=tk.HORIZONTAL, value=self.alert_conf,
                                          command=self.update_alert_conf)
        self.alert_conf_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Alert time slider
        ttk.Label(control_panel, text=f"Alert Time (seconds): {self.alert_time}").pack(anchor=tk.W, pady=(10, 5))
        self.alert_time_slider = ttk.Scale(control_panel, from_=1.0, to=10.0, length=200, 
                                          orient=tk.HORIZONTAL, value=self.alert_time,
                                          command=self.update_alert_time)
        self.alert_time_slider.pack(anchor=tk.W, pady=(0, 10))
        
        # Video controls
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Select video button
        self.select_video_btn = ttk.Button(control_panel, text="Select Video", command=self.select_video)
        self.select_video_btn.pack(anchor=tk.W, pady=(10, 5), fill=tk.X)
        
        # Play/Pause button
        self.play_btn = ttk.Button(control_panel, text="Play Video", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Stop button
        self.stop_btn = ttk.Button(control_panel, text="Stop Video", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Save result button
        self.save_btn = ttk.Button(control_panel, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(anchor=tk.W, pady=(5, 5), fill=tk.X)
        
        # Status
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.status_var = tk.StringVar(value="Ready. Please select a video file.")
        ttk.Label(control_panel, textvariable=self.status_var, wraplength=200).pack(anchor=tk.W, pady=10)
        
        # Detection info
        self.info_text = tk.Text(control_panel, height=10, width=25, state=tk.DISABLED)
        self.info_text.pack(anchor=tk.W, pady=(10, 0), fill=tk.BOTH, expand=True)
        
        # Video display (right side)
        self.video_frame = ttk.LabelFrame(main_frame, text="Video", padding=10)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial model loading
        self.load_model()
    
    def update_model(self, event=None):
        model_name = self.model_var.get()
        self.model_path = self.models[model_name]
        self.load_model()
    
    def update_conf(self, value):
        self.conf_threshold = float(value)
        # Update the label to show the current value
        for child in self.conf_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Confidence Threshold" in child.cget("text"):
                child.config(text=f"Confidence Threshold: {self.conf_threshold:.2f}")
    
    def update_alert_conf(self, value):
        self.alert_conf = float(value)
        # Update the label to show the current value
        for child in self.alert_conf_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Alert Confidence" in child.cget("text"):
                child.config(text=f"Alert Confidence: {self.alert_conf:.2f}")
    
    def update_alert_time(self, value):
        self.alert_time = float(value)
        # Update the label to show the current value
        for child in self.alert_time_slider.master.winfo_children():
            if isinstance(child, ttk.Label) and "Alert Time" in child.cget("text"):
                child.config(text=f"Alert Time (seconds): {self.alert_time:.2f}")
    
    def load_model(self):
        self.status_var.set(f"Loading model from {self.model_path}...")
        
        # Disable buttons during loading
        self.select_video_btn.config(state=tk.DISABLED)
        
        # Load model in a separate thread
        def load_model_thread():
            try:
                self.model = YOLO(self.model_path)
                # Schedule GUI update on the main thread
                self.root.after(0, self.model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error loading model: {str(e)}"))
        
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
            
            # Open video
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.show_error(f"Error opening video: {file_path}")
                return
            
            # Display first frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                
                # Enable play button
                self.play_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
            else:
                self.show_error("Could not read video frame")
    
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play Video")
        else:
            self.is_playing = True
            self.play_btn.config(text="Pause Video")
            
            # Start video processing in a thread if not already running
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.process_video, daemon=True)
                self.thread.start()
    
    def stop_video(self):
        self.is_playing = False
        self.play_btn.config(text="Play Video")
        
        # Reset video to beginning
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
            
        # Variables for drowning detection
        drowning_start_time = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Reset video if at end
        if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        self.status_var.set("Processing video...")
        
        while self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                # End of video
                self.is_playing = False
                self.play_btn.config(text="Play Video")
                self.status_var.set("End of video")
                self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                break
                
            # Run detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            
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
                    if conf_value > self.alert_conf:
                        is_drowning_detected = True
                        high_conf_drowning = True
                elif class_name == "swimming":
                    swimming_count += 1
                elif class_name == "tread water":
                    tread_water_count += 1
            
            # Update drowning timer
            current_time = time.time()
            if high_conf_drowning:
                if drowning_start_time is None:
                    drowning_start_time = current_time
            else:
                drowning_start_time = None
            
            # Calculate drowning duration
            drowning_duration = 0
            if drowning_start_time is not None:
                drowning_duration = current_time - drowning_start_time
            
            # Determine if alert should be shown
            show_alert = drowning_start_time is not None and drowning_duration >= self.alert_time
            
            # Draw bounding boxes and alerts
            processed_frame = self.draw_bbox(frame, boxes, classes, confidences, show_alert)
            
            # Add drowning timer indicator if detecting drowning
            if drowning_start_time is not None:
                timer_text = f"Drowning Timer: {drowning_duration:.1f}s / {self.alert_time:.1f}s"
                cv2.putText(processed_frame, timer_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update detection info
            self.root.after(0, lambda: self.update_info(drowning_count, swimming_count, tread_water_count, 
                                                       max_drowning_conf, show_alert))
            
            # Store the current frame
            self.current_frame = processed_frame
            
            # Display the frame
            self.root.after(0, lambda f=processed_frame: self.display_frame(f))
            
            # Enable the save button
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
            # Slight delay to not overload the system
            time.sleep(0.01)
    
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
    
    def update_info(self, drowning, swimming, tread_water, max_drowning_conf, alert):
        """Update the information text box"""
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
        
        self.info_text.config(state=tk.DISABLED)
    
    def display_frame(self, frame):
        """Display a frame on the canvas"""
        if frame is None:
            return
            
        # Convert from BGR to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate aspect ratio
            frame_height, frame_width = rgb_frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas is wider than needed
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas is taller than needed
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize the frame
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a PIL Image and then a PhotoImage
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update the canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference to prevent garbage collection
    
    def show_error(self, message):
        """Display an error message"""
        messagebox.showerror("Error", message)
        self.status_var.set(f"Error: {message}")

def main():
    root = tk.Tk()
    app = DrowningDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 