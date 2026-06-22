"""
Object Detection and Tracking System
A real-time object detection and tracking system using YOLOv8 and SORT algorithm.
"""
from .detector import ObjectDetector
from .tracker import ObjectTracker, VideoProcessor
__version__ = "1.0.0"
__author__ = "CodeAlpha Intern"
__all__ = [
    "ObjectDetector",
    "ObjectTracker",
    "VideoProcessor"
]