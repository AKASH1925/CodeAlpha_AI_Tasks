"""
Main application for Object Detection and Tracking.
This module provides the main entry point for running real-time
object detection and tracking on video streams.
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from src.tracker import ObjectTracker, VideoProcessor
from utils.visualization import (
    draw_bboxes,
    draw_bboxes_with_scores,
    draw_fps,
    draw_detection_stats,
    create_tracker_legend
)
class Application:
    """
    Main application class for Object Detection and Tracking.
    Orchestrates the detection, tracking, and visualization pipeline.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        source: int = 0,
        output_path: str = None,
        display_size: tuple = (1280, 720),
        show_fps: bool = True,
        show_labels: bool = True,
        show_tracking_id: bool = True
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.source = source
        self.output_path = output_path
        self.display_size = display_size
        self.show_fps = show_fps
        self.show_labels = show_labels
        self.show_tracking_id = show_tracking_id
        self.tracker = ObjectTracker(
            model_path=model_path,
            confidence=confidence,
            device=device
        )
        self.video_processor = VideoProcessor(
            tracker=self.tracker,
            source=source,
            output_path=output_path,
            display_size=display_size
        )
        self.fps = 0
        self.frame_times = []
    def run(self):
        print("=" * 60)
        print("Object Detection and Tracking System")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Confidence: {self.confidence}")
        print(f"Device: {self.device}")
        print(f"Source: {self.source}")
        if self.output_path:
            print(f"Output: {self.output_path}")
        print("=" * 60)
        try:
            self.video_processor.start()
            video_info = self.video_processor.get_video_info()
            print(f"Video Info: {video_info}")
            print("\nPress 'q' to quit")
            print("Press 'p' to pause/resume")
            print("Press 'r' to reset tracker")
            print("-" * 60)
            paused = False
            while self.video_processor.is_running:
                if not paused:
                    frame = self.video_processor.read_frame()
                    if frame is None:
                        print("\nEnd of video stream")
                        break
                    start_time = time.time()
                    tracks, detections = self.tracker.update(frame)
                    class_names = self.tracker.get_class_names()
                    output_frame = frame.copy()
                    if len(tracks) > 0:
                        for track in tracks:
                            x1, y1, x2, y2, track_id = track
                            class_id = -1
                            for det in detections:
                                det_x1, det_y1, det_x2, det_y2, det_class = det
                                track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                det_center = ((det_x1 + det_x2) / 2, (det_y1 + det_y2) / 2)
                                distance = np.sqrt(
                                    (track_center[0] - det_center[0])**2 +
                                    (track_center[1] - det_center[1])**2
                                )
                                if distance < 50:
                                    class_id = int(det_class)
                                    break
                            color = (0, 255, 0)
                            cv2.rectangle(
                                output_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color, 2
                            )
                            label_parts = []
                            if self.show_tracking_id:
                                label_parts.append(f"ID:{int(track_id)}")
                            if self.show_labels and class_id in class_names:
                                label_parts.append(class_names[class_id])
                            label = " ".join(label_parts) if label_parts else ""
                            if label:
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                cv2.rectangle(
                                    output_frame,
                                    (int(x1), int(y1) - text_height - 10),
                                    (int(x1) + text_width, int(y1)),
                                    color, -1
                                )
                                cv2.putText(
                                    output_frame, label,
                                    (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2
                                )
                    end_time = time.time()
                    frame_time = end_time - start_time
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)
                    self.fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
                    if self.show_fps:
                        output_frame = draw_fps(output_frame, self.fps)
                    output_frame = draw_detection_stats(
                        output_frame, len(detections), len(tracks), self.fps
                    )
                    self.video_processor.write_frame(output_frame)
                    self.video_processor.display_frame(output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'}")
                elif key == ord('r'):
                    self.tracker.reset()
                    print("\nTracker reset")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            stats = self.tracker.get_stats()
            print("\n" + "=" * 60)
            print("Final Statistics:")
            print(f"  Frames processed: {stats['frame_count']}")
            print(f"  Total unique tracks: {stats['total_tracks']}")
            print(f"  Average FPS: {self.fps:.1f}")
            print("=" * 60)
            self.video_processor.stop()
def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection and Tracking System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model path")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Display width")
    parser.add_argument("--height", type=int, default=720, help="Display height")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS counter")
    parser.add_argument("--no-labels", action="store_true", help="Hide class labels")
    parser.add_argument("--no-tracking-id", action="store_true", help="Hide tracking IDs")
    return parser.parse_args()
def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    app = Application(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        source=source,
        output_path=args.output,
        display_size=(args.width, args.height),
        show_fps=not args.no_fps,
        show_labels=not args.no_labels,
        show_tracking_id=not args.no_tracking_id
    )
    app.run()
if __name__ == "__main__":
    main()