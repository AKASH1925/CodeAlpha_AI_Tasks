"""
YOLOv8 Object Detection Module.
Provides a wrapper around YOLOv8 for real-time object detection
with support for multiple model sizes and configurations.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
class ObjectDetector:
    """
    YOLOv8-based object detector for real-time detection.
    Supports various YOLOv8 model sizes (n, s, m, l, x) and
    provides both detection and optional tracking capabilities.
    """
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None
    ):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.model = self._load_model()
        self.class_names = self.COCO_CLASSES
    def _load_model(self) -> YOLO:
        print(f"Loading YOLOv8 model: {self.model_path}")
        try:
            model = YOLO(self.model_path)
            print(f"Model loaded successfully on device: {self.device}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    def detect(
        self,
        frame: np.ndarray,
        return_scores: bool = False
    ) -> np.ndarray:
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            verbose=False
        )
        detections = []
        scores = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    if return_scores:
                        detections.append([x1, y1, x2, y2, class_id, score])
                    else:
                        detections.append([x1, y1, x2, y2, class_id])
                        scores.append(score)
        if return_scores:
            return np.array(detections) if detections else np.empty((0, 6))
        else:
            return np.array(detections) if detections else np.empty((0, 5)), np.array(scores)
    def detect_batch(
        self,
        frames: List[np.ndarray],
        return_scores: bool = False
    ) -> List[np.ndarray]:
        results = []
        for frame in frames:
            results.append(self.detect(frame, return_scores))
        return results
    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")
    def get_class_names_dict(self) -> Dict[int, str]:
        return self.class_names.copy()
    def update_config(
        self,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ):
        if confidence is not None:
            self.confidence = confidence
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if classes is not None:
            self.classes = classes
    def benchmark(self, frame: np.ndarray, num_iterations: int = 100) -> float:
        import time
        for _ in range(10):
            self.detect(frame)
        start_time = time.time()
        for _ in range(num_iterations):
            self.detect(frame)
        end_time = time.time()
        avg_time = ((end_time - start_time) / num_iterations) * 1000
        return avg_time
if __name__ == "__main__":
    detector = ObjectDetector(model_path="yolov8n.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, scores = detector.detect(dummy_frame)
    print(f"Detected {len(detections)} objects")
    print(f"Class names: {detector.get_class_names_dict()}")