# Task 4: Object Detection and Tracking
Real-time object detection using **YOLOv8** and multi-object tracking using **SORT algorithm**.
## Requirements
- Set up real-time video input using webcam or video file (OpenCV)
- Use pre-trained YOLOv8 model for object detection
- Process each frame to detect objects and draw bounding boxes
- Apply SORT tracking with consistent IDs
- Display output with labels and tracking IDs in real time
## Installation
```bash
git clone https://github.com/your-username/Task4_ObjectDetectionAndTracking.git
cd Task4_ObjectDetectionAndTracking
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt