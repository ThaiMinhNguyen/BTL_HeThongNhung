from ultralytics import YOLO
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

class DrowningDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowning Detection")
        self.root.geometry("1200x800")
        
        # Model path
        self.model_path = 'runs/drowning_detection/weights/best2.pt'
        
        # Confidence threshold
        self.conf_threshold = 0.25
        
        # Create GUI components
        self.create_widgets()
        
        # Load model
        self.load_model()
        
    def load_model(self):
        # Create a progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Loading Model")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Add a progress bar
        progress_label = tk.Label(progress_window, text=f"Loading model from {self.model_path}...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=250, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()
        
        # Load model in a separate thread to prevent GUI from freezing
        def load_model_thread():
            try:
                self.model = YOLO(self.model_path)
                # Schedule GUI update on the main thread
                self.root.after(0, lambda: self.model_loaded(progress_window))
            except Exception as e:
                self.root.after(0, lambda: self.model_load_error(progress_window, str(e)))
        
        thread = threading.Thread(target=load_model_thread)
        thread.daemon = True
        thread.start()
    
    def model_loaded(self, progress_window):
        progress_window.destroy()
        self.status_label.config(text="Model loaded successfully. Click 'Select Image' to begin.")
        
    def model_load_error(self, progress_window, error_message):
        progress_window.destroy()
        messagebox.showerror("Error Loading Model", f"Failed to load model: {error_message}")
        self.status_label.config(text="Error loading model.")
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for controls
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Select image button
        self.select_btn = tk.Button(control_frame, text="Select Image", command=self.select_image, width=15, height=2)
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        # Confidence threshold slider
        threshold_frame = tk.Frame(control_frame)
        threshold_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(threshold_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        
        self.threshold_var = tk.DoubleVar(value=self.conf_threshold)
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.1, to=1.0, resolution=0.05, 
                                         orient=tk.HORIZONTAL, variable=self.threshold_var, 
                                         command=self.update_threshold, length=200)
        self.threshold_slider.pack()
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Loading model...", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        # Frame for displaying image
        self.image_frame = tk.Frame(main_frame, bg="black")
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label for displaying image
        self.image_label = tk.Label(self.image_frame, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame for detection results
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.X, pady=10)
        
        # Label for detection results
        self.results_label = tk.Label(self.results_frame, text="Detection Results:", anchor=tk.W)
        self.results_label.pack(anchor=tk.W)
        
        # Text widget for detailed results
        self.results_text = tk.Text(self.results_frame, height=5, width=50)
        self.results_text.pack(fill=tk.X, expand=True)
    
    def update_threshold(self, value):
        self.conf_threshold = float(value)
        # If an image is already loaded, reprocess it with the new threshold
        if hasattr(self, 'current_image'):
            self.process_image(self.current_image)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.status_label.config(text=f"Processing image: {os.path.basename(file_path)}")
            
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", f"Could not read image: {file_path}")
                self.status_label.config(text="Error reading image.")
                return
            
            # Store the current image
            self.current_image = image
            
            # Process the image
            self.process_image(image)
    
    def process_image(self, image):
        # Run YOLOv8 inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        
        # Process detection results
        boxes = []
        classes = []
        confidences = []
        
        # Extract detection information
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
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Detected {len(boxes)} objects in the image:\n")
        
        class_counts = {"swimming": 0, "tread water": 0, "drowning": 0}
        
        for i, cls in enumerate(classes):
            confidence = confidences[i]
            self.results_text.insert(tk.END, f"  - {cls}: {confidence:.2f}\n")
            
            # Count detections by class
            if cls in class_counts:
                class_counts[cls] += 1
        
        # Add summary
        self.results_text.insert(tk.END, f"\nSummary: ")
        for cls, count in class_counts.items():
            if count > 0:
                self.results_text.insert(tk.END, f"{cls}: {count}, ")
        
        # Draw bounding boxes
        if len(boxes) > 0:
            output_image = self.draw_bbox(image, boxes, classes, confidences)
            self.display_image(output_image)
            self.status_label.config(text=f"Found {len(boxes)} object(s). Confidence threshold: {self.conf_threshold}")
        else:
            self.display_image(image)
            self.status_label.config(text="No objects detected in the image.")
    
    def draw_bbox(self, image, boxes, classes, confidences):
        """Draw bounding boxes on the image"""
        out = image.copy()
        
        # Colors for different classes (BGR format)
        class_colors = {
            "swimming": (0, 255, 0),     # Green
            "tread water": (0, 255, 255), # Yellow
            "drowning": (0, 0, 255)      # Red
        }
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            label = classes[i]
            
            # Get color based on class
            color = class_colors.get(label, (255, 0, 0))  # Default to blue
            
            # Draw box
            thickness = 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            
            # Add a filled rectangle at the top for text background
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), color, -1)
            text = f"{label}: {confidence:.2f}"
            
            # Add text
            font_scale = 0.6
            cv2.putText(out, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        return out
    
    def display_image(self, cv_image):
        """Display an OpenCV image in the GUI"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Get the frame dimensions
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # Get the image dimensions
        img_height, img_width = rgb_image.shape[:2]
        
        # Calculate the scaling factor to fit the image in the frame
        # while maintaining aspect ratio
        width_ratio = frame_width / img_width
        height_ratio = frame_height / img_height
        scale_factor = min(width_ratio, height_ratio)
        
        if scale_factor < 1:
            # Only scale down, not up
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update the image label
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowningDetectionApp(root)
    root.mainloop() 