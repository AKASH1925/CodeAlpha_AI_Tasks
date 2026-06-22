"""
Example usage script demonstrating the Object Detection and Tracking system.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
def example_basic_detection():
    print("=" * 50)
    print("Example 1: Basic Object Detection")
    print("=" * 50)
    detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, scores = detector.detect(frame, return_scores=True)
    print(f"Detected {len(detections)} objects")
    for i, (det, score) in enumerate(zip(detections, scores)):
        x1, y1, x2, y2, class_id = det
        class_name = detector.get_class_name(int(class_id))
        print(f"  {i+1}. {class_name}: {score:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
def example_tracking():
    print("\n" + "=" * 50)
    print("Example 2: Object Tracking")
    print("=" * 50)
    tracker = ObjectTracker(model_path="yolov8n.pt", confidence=0.5, device="cpu")
    num_frames = 10
    print(f"Simulating {num_frames} frames...")
    for frame_num in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks, detections = tracker.update(frame)
        print(f"Frame {frame_num + 1}: {len(detections)} detections, {len(tracks)} active tracks")
    stats = tracker.get_stats()
    print(f"\nTotal unique tracks: {stats['total_tracks']}")
def example_webcam():
    print("\n" + "=" * 50)
    print("Example 3: Real-time Webcam Tracking")
    print("=" * 50)
    print("Press 'q' to quit")
    tracker = ObjectTracker(model_path="yolov8n.pt", confidence=0.5, device="cpu")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tracks, detections = tracker.update(frame)
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Webcam Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    example_basic_detection()
    example_tracking()
    # Uncomment for webcam: example_webcam()
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)