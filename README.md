# Drowning Detection with YOLOv8

This project uses YOLOv8 to detect drowning incidents in swimming pools.

## Environment Setup

1. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python train.py
```

The training results will be saved in the `runs/detect/drowning_detection/` directory.

## Evaluation

To evaluate the model performance:
```bash
python evaluate.py --model runs/detect/drowning_detection/weights/best.pt
```

Options:
- `--model`: Path to the trained model
- `--data`: Path to the data configuration file
- `--iou`: IOU threshold for evaluation
- `--save_dir`: Directory to save evaluation results

## Inference

### Using YOLOv8 Model with Advanced Drowning Alerts

For real-time drowning detection with alert system:
```bash
# Using webcam
python yolo_drown_detect.py --show

# Using a video file
python yolo_drown_detect.py --source path/to/video.mp4 --show --save

# Using a specific model with custom confidence threshold
python yolo_drown_detect.py --model runs/detect/drowning_detection/weights/best.pt --conf 0.25 --output_path output_video.mp4
```

Options:
- `--source`: Source video file or webcam index (0, 1, etc.)
- `--model`: Path to the trained model
- `--conf`: Confidence threshold
- `--save`: Save video output
- `--show`: Show detection results
- `--output_path`: Custom path for output video

This script includes:
- Real-time drowning detection
- Visual alerts with red borders
- Warning text when drowning is detected
- FPS counter
- Configurable alert threshold

### Using YOLOv8 Model (Basic Version)

Basic inference without the alert system:
```bash
# Using webcam
python predict.py --show

# Using a video file
python predict.py --source path/to/video.mp4 --show --save

# Using the trained model
python predict.py --model runs/detect/drowning_detection/weights/best.pt --conf 0.25
```

### Using Alternative CNN Model

This project also includes an alternative CNN-based drowning detection model using `cvlib` for person detection:

```bash
# Using webcam
python DrownDetect.py --source 0

# Using a video file (place video in 'videos' folder)
python DrownDetect.py --source video_name.mp4
```

Note: This approach requires the following files:
- `model.pth`: The trained CNN model weights
- `lb.pkl`: Label binarizer for class mapping

## Project Structure

```
Drown_Detect/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── models/
├── runs/
├── videos/
├── venv/
├── requirements.txt
├── predict.py
├── evaluate.py
├── DrownDetect.py
├── yolo_drown_detect.py
└── train.py
``` 