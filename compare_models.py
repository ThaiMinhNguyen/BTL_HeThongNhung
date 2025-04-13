from ultralytics import YOLO
import argparse
import time
import tabulate
import os
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple YOLOv8 drowning detection models')
    parser.add_argument('--models', nargs='+', default=['runs/drowning_detection/weights/best.pt', 
                                                        'runs/drowning_detection/weights/best1.pt'], 
                        help='Paths to model files to compare')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='Path to the data configuration file')
    parser.add_argument('--video', type=str, default=None, 
                        help='Optional: Path to test video to evaluate inference speed')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold')
    return parser.parse_args()

def evaluate_models(model_paths, data_path, conf_threshold):
    """Evaluate models on validation dataset and return metrics."""
    results = []
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        
        # Load model
        try:
            model = YOLO(model_path)
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            
            # Evaluate model
            metrics = model.val(data=data_path, conf=conf_threshold)
            
            # Extract metrics
            map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
            map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0
            precision = float(metrics.box.p) if hasattr(metrics.box, 'p') else 0.0
            recall = float(metrics.box.r) if hasattr(metrics.box, 'r') else 0.0
            
            # Nếu các giá trị là mảng, lấy giá trị đầu tiên
            if hasattr(map50, '__len__') and len(map50) > 0:
                map50 = float(map50[0])
            if hasattr(map50_95, '__len__') and len(map50_95) > 0:
                map50_95 = float(map50_95[0])
            if hasattr(precision, '__len__') and len(precision) > 0:
                precision = float(precision[0])
            if hasattr(recall, '__len__') and len(recall) > 0:
                recall = float(recall[0])
            
            # Add to results
            results.append({
                'Model': os.path.basename(model_path),
                'Size (MB)': f"{model_size:.2f}",
                'mAP@0.5': f"{map50:.4f}",
                'mAP@0.5:0.95': f"{map50_95:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}"
            })
            
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            results.append({
                'Model': os.path.basename(model_path),
                'Size (MB)': "Error",
                'mAP@0.5': "Error",
                'mAP@0.5:0.95': "Error",
                'Precision': "Error",
                'Recall': "Error"
            })
    
    return results

def benchmark_inference_speed(model_paths, video_path, conf_threshold, num_frames=200):
    """Benchmark inference speed on a video."""
    if not video_path:
        return []
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return []
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames = min(num_frames, frame_count)
    
    # Read frames into memory
    frames = []
    for _ in range(actual_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nBenchmarking inference speed for {model_name}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Warm up
            _ = model(frames[0], conf=conf_threshold, verbose=False)
            
            # Benchmark
            start_time = time.time()
            
            for frame in tqdm(frames, desc=f"Processing frames with {model_name}"):
                _ = model(frame, conf=conf_threshold, verbose=False)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            fps = actual_frames / total_time
            ms_per_frame = (total_time / actual_frames) * 1000
            
            results.append({
                'Model': model_name,
                'FPS': f"{fps:.2f}",
                'ms/frame': f"{ms_per_frame:.2f}",
                'Frames': actual_frames
            })
            
        except Exception as e:
            print(f"Error benchmarking {model_path}: {e}")
            results.append({
                'Model': model_name,
                'FPS': "Error",
                'ms/frame': "Error",
                'Frames': actual_frames
            })
    
    return results

def main():
    args = parse_args()
    
    # Print model paths being compared
    print("Comparing models:")
    for i, model_path in enumerate(args.models):
        print(f"{i+1}. {model_path}")
    
    # Evaluate models on validation dataset
    eval_results = evaluate_models(args.models, args.data, args.conf)
    
    # Print evaluation results as table
    print("\nModel Evaluation Results:")
    headers = ['Model', 'Size (MB)', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    table = [[result[h] for h in headers] for result in eval_results]
    print(tabulate.tabulate(table, headers, tablefmt="grid"))
    
    # If video path is provided, benchmark inference speed
    if args.video:
        speed_results = benchmark_inference_speed(args.models, args.video, args.conf)
        
        # Print speed results as table
        print("\nInference Speed Benchmark:")
        headers = ['Model', 'FPS', 'ms/frame', 'Frames']
        table = [[result[h] for h in headers] for result in speed_results]
        print(tabulate.tabulate(table, headers, tablefmt="grid"))
    
    # Print conclusion
    if eval_results and all('Error' not in result['mAP@0.5'] for result in eval_results):
        best_map = max(eval_results, key=lambda x: float(x['mAP@0.5']))
        print(f"\nConclusion: Based on mAP@0.5, the best model is: {best_map['Model']}")
        
        if args.video and speed_results and all('Error' not in result['FPS'] for result in speed_results):
            best_fps = max(speed_results, key=lambda x: float(x['FPS']))
            print(f"The fastest model is: {best_fps['Model']} with {best_fps['FPS']} FPS")

if __name__ == "__main__":
    main() 