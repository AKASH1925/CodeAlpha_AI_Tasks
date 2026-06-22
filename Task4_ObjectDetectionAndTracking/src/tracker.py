"""
Object Tracker Module.
Combines YOLOv8 detection with SORT tracking for real-time
object detection and tracking pipeline.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from .detector import ObjectDetector
from utils.sort import Sort
class ObjectTracker:
    """
    Real-time object detection and tracking pipeline.
    Combines YOLOv8 detector with SORT tracker for robust
    multi-object tracking with consistent IDs.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        max_age: int = 30,
        min_hits: int = 3,
        tracker_iou_threshold: float = 0.3
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            device=device,
            classes=classes
        )
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=tracker_iou_threshold
        )
        self.frame_count = 0
        self.total_tracks = 0
    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.frame_count += 1
        detections, scores = self.detector.detect(frame, return_scores=False)
        if len(detections) > 0:
            det_with_scores = np.column_stack([detections[:, :4], scores])
        else:
            det_with_scores = np.empty((0, 5))
        tracks = self.tracker.update(det_with_scores)
        if len(tracks) > 0:
            self.total_tracks = max(self.total_tracks, int(tracks[:, 4].max()) + 1)
        return tracks, detections
    def detect_only(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.detector.detect(frame, return_scores=True)
    def get_class_name(self, class_id: int) -> str:
        return self.detector.get_class_name(class_id)
    def get_class_names(self) -> Dict[int, str]:
        return self.detector.get_class_names_dict()
    def update_config(
        self,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ):
        self.detector.update_config(confidence, iou_threshold, classes)
    def reset(self):
        self.tracker = Sort(
            max_age=self.tracker.max_age,
            min_hits=self.tracker.min_hits,
            iou_threshold=self.tracker.iou_threshold
        )
        self.frame_count = 0
    def get_stats(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "total_tracks": self.total_tracks,
            "active_tracks": len(self.tracker.trackers)
        }
class VideoProcessor:
    """
    Video processing pipeline for object detection and tracking.
    Handles video input/output, frame processing, and visualization.
    """
    def __init__(
        self,
        tracker: ObjectTracker,
        source: int = 0,
        output_path: Optional[str] = None,
        display_size: Tuple[int, int] = (1280, 720)
    ):
        self.tracker = tracker
        self.source = source
        self.output_path = output_path
        self.display_size = display_size
        self.cap = None
        self.writer = None
        self.is_running = False
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_size[1])
        if self.output_path:
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, fps,
                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
        self.is_running = True
        print(f"Video capture started from source: {self.source}")
    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("Video capture stopped")
    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    def write_frame(self, frame: np.ndarray):
        if self.writer:
            self.writer.write(frame)
    def display_frame(
        self,
        frame: np.ndarray,
        window_name: str = "Object Detection & Tracking"
    ):
        display_frame = cv2.resize(frame, self.display_size)
        cv2.imshow(window_name, display_frame)
    def get_video_info(self) -> dict:
        if self.cap is None:
            return {}
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }