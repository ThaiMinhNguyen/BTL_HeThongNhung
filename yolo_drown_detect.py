from ultralytics import YOLO
import cv2
import argparse
import os
import numpy as np
import time
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Drowning detection using YOLOv8')
    parser.add_argument('--source', type=str, default=None, help='Source video file or webcam index (0, 1, etc.)')
    parser.add_argument('--model', type=str, default='runs/drowning_detection/weights/best.pt', 
                        help='Path to the trained model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--alert_conf', type=float, default=0.65, help='Confidence threshold for drowning alerts')
    parser.add_argument('--alert_time', type=float, default=5.0, help='Minimum time in seconds for drowning alert')
    parser.add_argument('--save', action='store_true', help='Save video output')
    parser.add_argument('--show', action='store_true', help='Show detection results')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Output video path')
    return parser.parse_args()

def draw_bbox(frame, boxes, classes, confidences, is_drowning=False):
    """
    Draw bounding boxes on the frame in a style similar to DrownDetect.py
    
    Args:
        frame: Input frame
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        classes: List of class names for each box
        confidences: List of confidence scores for each box
        is_drowning: Boolean indicating if drowning is detected
    
    Returns:
        Frame with bounding boxes drawn
    """
    # Make a copy of the frame
    out = frame.copy()
    
    # Print minimal debug information (only if drowning alert)
    if is_drowning:
        print(f"Drowning alert active - Drawing {len(boxes)} bounding boxes")
    
    # Colors for drawing
    box_color = (0, 0, 255) if is_drowning else (0, 255, 0)  # Red if drowning, Green otherwise
    text_color = (255, 255, 255)  # White text
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        # Skip debug info for each box
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        label = classes[i]
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw box with thicker borders if drowning
        thickness = 5 if is_drowning else 3  # Increased thickness for better visibility
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        
        # Add a filled rectangle at the top for text background
        if label == "drowning":
            # Use red for drowning class regardless of alert status
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), (0, 0, 255), -1)
            text = f"DROWNING: {confidence:.2f}"
        else:
            cv2.rectangle(out, (x1, y1 - 30), (x1 + 160, y1), (0, 255, 0), -1)
            text = f"{label}: {confidence:.2f}"
        
        # Add text with larger font
        font_scale = 0.7
        cv2.putText(out, text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
    # Add warning text if drowning
    if is_drowning:
        height, width = out.shape[:2]
        warning_text = "DROWNING DETECTED!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        # Position text in the center top
        text_x = (width - text_size[0]) // 2
        
        # Add warning text at the top
        cv2.putText(out, warning_text, (text_x, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Add flashing border around the frame
        border_thickness = int(3 + 2 * abs(np.sin(time.time() * 5)))
        cv2.rectangle(out, (5, 5), (width-5, height-5), (0, 0, 255), border_thickness)
    
    return out

def main():
    args = parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    print("Model loaded successfully!")
    
    # Set source to webcam if not specified
    if args.source is None:
        args.source = 0  # Use default webcam
        args.show = True  # Force show if using webcam
    elif args.source.isdigit():
        args.source = int(args.source)
    elif not os.path.exists(args.source):
        print(f"Error: Source file {args.source} does not exist!")
        return
    
    # Open video source
    print(f"Opening video source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if saving
    out = None
    if args.save:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
        print(f"Saving output to {args.output_path}")
    
    # Detection loop
    drowning_start_time = None
    start_time = time.time()
    frame_count = 0
    
    # Variables for printing results every 2 seconds
    last_print_time = time.time()
    print_interval = 2  # Print every 2 seconds
    
    # Calculate frames threshold based on time and FPS
    alert_frames_threshold = int(args.alert_time * fps)
    print(f"Alert will trigger after {args.alert_time} seconds ({alert_frames_threshold} frames) of continuous drowning detection with confidence > {args.alert_conf}")
    
    print("Starting detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Run YOLOv8 inference
        results = model(frame, conf=args.conf, verbose=False)[0]
        
        # Process detection results
        boxes = []
        classes = []
        confidences = []
        is_drowning_detected = False
        high_conf_drowning = False
        drowning_count = 0
        swimming_count = 0
        max_drowning_conf = 0
        
        # Minimal debug info - only print number of detections every 20 frames
        if frame_count % 20 == 0:
            print(f"Frame {frame_count}: Detected {len(results.boxes)} objects")
        
        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_id = int(cls)
            class_name = model.names[class_id]
            
            # Add to lists
            boxes.append([x1, y1, x2, y2])
            classes.append(class_name)
            confidences.append(conf.item())
            
            # Count detections by class
            if class_name == "drowning":
                drowning_count += 1
                conf_value = conf.item()
                max_drowning_conf = max(max_drowning_conf, conf_value)
                if conf_value > args.alert_conf:  # Only consider high confidence detections for alert
                    is_drowning_detected = True
                    high_conf_drowning = True
            else:
                swimming_count += 1
        
        # Make sure we have detections before proceeding
        if len(boxes) == 0:
            # Only print "no detections" message every 30 frames to reduce console output
            if frame_count % 30 == 0:
                print("No detections in recent frames")
            # Add empty frame handling
            processed_frame = frame.copy()
            
            # Add status text when no detections
            cv2.putText(processed_frame, "No detections", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Reset drowning timer if no detections
            drowning_start_time = None
        else:
            # Update drowning timer
            if high_conf_drowning:
                if drowning_start_time is None:
                    drowning_start_time = current_time
                    print(f"High confidence drowning detected (conf={max_drowning_conf:.2f}). Starting timer.")
            else:
                if drowning_start_time is not None:
                    print("Drowning no longer detected with high confidence. Resetting timer.")
                drowning_start_time = None
            
            # Calculate drowning duration
            drowning_duration = 0
            if drowning_start_time is not None:
                drowning_duration = current_time - drowning_start_time
            
            # Determine if alert should be shown
            show_alert = drowning_start_time is not None and drowning_duration >= args.alert_time
            
            # Draw bounding boxes in a style similar to DrownDetect.py
            processed_frame = draw_bbox(frame, boxes, classes, confidences, show_alert)
            
            # Add drowning timer indicator if detecting drowning
            if drowning_start_time is not None:
                timer_text = f"Drowning Timer: {drowning_duration:.1f}s / {args.alert_time:.1f}s"
                cv2.putText(processed_frame, timer_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
            cv2.putText(processed_frame, fps_text, (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Reset FPS counter every second
            if elapsed_time > 1:
                start_time = current_time
                frame_count = 0
        
        # Print detection results less frequently to reduce console output
        if current_time - last_print_time >= print_interval:
            # Get current status
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_status = "DROWNING ALERT!" if (drowning_start_time is not None and drowning_duration >= args.alert_time) else "Normal"
            
            # Define last_status as a global variable if it doesn't exist
            if not hasattr(main, 'last_status'):
                main.last_status = ""
                
            # Only print if status changed or drowning alert is active
            if current_status != main.last_status or current_status == "DROWNING ALERT!":
                if drowning_start_time is not None:
                    print(f"[{timestamp}] Status: {current_status} | Drowning: {drowning_count} | Max Conf: {max_drowning_conf:.2f} | Timer: {drowning_duration:.1f}s/{args.alert_time:.1f}s")
                else:
                    print(f"[{timestamp}] Status: {current_status} | Drowning: {drowning_count} | Swimming: {swimming_count}")
                main.last_status = current_status
            
            last_print_time = current_time
        
        # Display detection count on frame
        status_text = f"Drowning: {drowning_count} | Swimming: {swimming_count}"
        cv2.putText(processed_frame, status_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame if show is enabled
        if args.show:
            cv2.imshow("Drowning Detection", processed_frame)
        
        # Save frame if save is enabled
        if args.save and out is not None:
            out.write(processed_frame)
        
        # Break loop if 'q' is pressed
        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    if args.show:
        cv2.destroyAllWindows()
    
    print("Detection finished!")

if __name__ == "__main__":
    main() 