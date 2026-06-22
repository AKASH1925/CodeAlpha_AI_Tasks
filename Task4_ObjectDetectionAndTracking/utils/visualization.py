"""
Visualization utilities for Object Detection and Tracking.
Provides functions for drawing bounding boxes, labels, FPS, and
creating visual legends for tracked objects.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (255, 0, 128),
]
def get_color(track_id: int) -> Tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]
def draw_bboxes(
    frame: np.ndarray,
    detections: np.ndarray,
    class_names: Dict[int, str],
    show_labels: bool = True,
    show_confidence: bool = True,
    show_tracking_id: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    output_frame = frame.copy()
    if len(detections) == 0:
        return output_frame
    for detection in detections:
        if len(detection) == 6:
            x1, y1, x2, y2, track_id, class_id = detection
            class_id = int(class_id)
        elif len(detection) == 5:
            x1, y1, x2, y2, track_id = detection
            class_id = -1
        else:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        color = get_color(track_id)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, line_thickness)
        label_parts = []
        if show_tracking_id:
            label_parts.append(f"ID:{track_id}")
        if show_labels and class_id in class_names:
            label_parts.append(class_names[class_id])
        if label_parts:
            label = " ".join(label_parts)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color, -1
            )
            cv2.putText(
                output_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), line_thickness
            )
    return output_frame
def draw_bboxes_with_scores(
    frame: np.ndarray,
    detections: np.ndarray,
    class_names: Dict[int, str],
    scores: Optional[np.ndarray] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_tracking_id: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    output_frame = frame.copy()
    if len(detections) == 0:
        return output_frame
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, track_id = detection[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        color = get_color(track_id)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, line_thickness)
        corner_length = 15
        cv2.line(output_frame, (x1, y1), (x1 + corner_length, y1), color, line_thickness + 1)
        cv2.line(output_frame, (x1, y1), (x1, y1 + corner_length), color, line_thickness + 1)
        cv2.line(output_frame, (x2, y1), (x2 - corner_length, y1), color, line_thickness + 1)
        cv2.line(output_frame, (x2, y1), (x2, y1 + corner_length), color, line_thickness + 1)
        cv2.line(output_frame, (x1, y2), (x1 + corner_length, y2), color, line_thickness + 1)
        cv2.line(output_frame, (x1, y2), (x1, y2 - corner_length), color, line_thickness + 1)
        cv2.line(output_frame, (x2, y2), (x2 - corner_length, y2), color, line_thickness + 1)
        cv2.line(output_frame, (x2, y2), (x2, y2 - corner_length), color, line_thickness + 1)
        label_parts = []
        if show_tracking_id:
            label_parts.append(f"ID:{track_id}")
        if show_labels and len(detection) > 5:
            class_id = int(detection[5])
            if class_id in class_names:
                label_parts.append(class_names[class_id])
        if show_confidence and scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color, -1
            )
            cv2.putText(
                output_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), line_thickness
            )
    return output_frame
def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    output_frame = frame.copy()
    fps_text = f"FPS: {fps:.1f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        output_frame,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + 5),
        (0, 0, 0), -1
    )
    cv2.putText(
        output_frame, fps_text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )
    return output_frame
def create_tracker_legend(
    frame: np.ndarray,
    track_ids: List[int],
    class_names: Dict[int, str],
    class_ids: Optional[List[int]] = None,
    position: Tuple[int, int] = (10, 60),
    font_scale: float = 0.5,
    thickness: int = 1
) -> np.ndarray:
    output_frame = frame.copy()
    if not track_ids:
        return output_frame
    line_height = 20
    legend_height = len(track_ids) * line_height + 10
    legend_width = 200
    cv2.rectangle(
        output_frame,
        (position[0], position[1]),
        (position[0] + legend_width, position[1] + legend_height),
        (0, 0, 0), -1
    )
    for i, track_id in enumerate(track_ids):
        color = get_color(track_id)
        class_name = "Object"
        if class_ids and i < len(class_ids) and class_ids[i] in class_names:
            class_name = class_names[class_ids[i]]
        y_pos = position[1] + 15 + i * line_height
        cv2.rectangle(
            output_frame,
            (position[0] + 5, y_pos - 10),
            (position[0] + 20, y_pos),
            color, -1
        )
        text = f"ID {track_id}: {class_name}"
        cv2.putText(
            output_frame, text, (position[0] + 25, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
    return output_frame
def draw_detection_stats(
    frame: np.ndarray,
    num_detections: int,
    num_tracks: int,
    fps: float,
    position: Tuple[int, int] = (10, 30)
) -> np.ndarray:
    output_frame = frame.copy()
    stats_height = 80
    cv2.rectangle(
        output_frame,
        (position[0], position[1]),
        (position[0] + 250, position[1] + stats_height),
        (0, 0, 0), -1
    )
    cv2.putText(
        output_frame, f"FPS: {fps:.1f}",
        (position[0] + 10, position[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    cv2.putText(
        output_frame, f"Detections: {num_detections}",
        (position[0] + 10, position[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        output_frame, f"Active Tracks: {num_tracks}",
        (position[0] + 10, position[1] + 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )
    return output_frame