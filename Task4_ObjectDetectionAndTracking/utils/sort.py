"""
SORT (Simple Online and Realtime Tracking) Implementation
A simplified implementation of the SORT algorithm for object tracking.
SORT uses Kalman filtering for state estimation and the Hungarian algorithm
for data association.
Reference: Bewley, A., et al. "Simple online and realtime tracking." ICIP 2016.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
def linear_assignment(cost_matrix):
    if cost_matrix.size == 0:
        return [], np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = list(range(cost_matrix.shape[0]))
    unmatched_cols = list(range(cost_matrix.shape[1]))
    for r, c in zip(row_indices, col_indices):
        matches.append((r, c))
        unmatched_rows.remove(r)
        unmatched_cols.remove(c)
    return matches, unmatched_rows, unmatched_cols
def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - wh
    return wh / union
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        cost_matrix = 1 - iou_matrix
        if cost_matrix.max() > 1.0:
            cost_matrix[cost_matrix > 1.0] = 1.0
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    else:
        row_indices, col_indices = np.array([]), np.array([])
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))
    for r, c in zip(row_indices, col_indices):
        if iou_matrix[r, c] < iou_threshold:
            continue
        matches.append((int(r), int(c)))
        unmatched_detections.remove(int(r))
        unmatched_trackers.remove(int(c))
    return matches, unmatched_detections, unmatched_trackers
class KalmanBoxTracker:
    """
    Kalman filter-based bounding box tracker.
    State vector: [x, y, s, r, dx, dy, ds]
    """
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R *= 10.0
        self.kf.R[2, 2] *= 10.0
        self.kf.R[3, 3] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits_list = []
        self.time_since_update_list = []
    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[x], [y], [s], [r]])
    def _convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(max(0, x[2] * x[3]))
        h = x[2] / w if w > 0 else 0
        if score is None:
            return np.array([
                x[0] - w / 2.0, x[1] - h / 2.0,
                x[0] + w / 2.0, x[1] + h / 2.0
            ]).reshape((1, 4))
        else:
            return np.array([
                x[0] - w / 2.0, x[1] - h / 2.0,
                x[0] + w / 2.0, x[1] + h / 2.0, score
            ]).reshape((1, 5))
    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._convert_x_to_bbox(self.kf.x)
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)
class Sort:
    """
    SORT tracker - Simple Online and Realtime Tracking.
    Maintains multiple KalmanBoxTrackers and handles
    association of detections to existing tracks using IOU.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    def update(self, detections):
        self.frame_count += 1
        predicted_boxes = []
        for tracker in self.trackers:
            pred = tracker.predict()[0]
            predicted_boxes.append(pred)
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))
        if len(detections) > 0:
            if detections.shape[1] == 5:
                det_boxes = detections[:, :4]
            else:
                det_boxes = detections
        else:
            det_boxes = np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.iou_threshold
        )
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])
        for det_idx in unmatched_dets:
            tracker = KalmanBoxTracker(detections[det_idx])
            self.trackers.append(tracker)
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]
        output = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits and tracker.time_since_update == 0:
                bbox = tracker.get_state()[0]
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], tracker.id])
        return np.array(output) if output else np.empty((0, 5))
if __name__ == "__main__":
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    det1 = np.array([[100, 100, 200, 200, 0.9]])
    det2 = np.array([[300, 300, 400, 400, 0.8]])
    tracks = tracker.update(det1)
    print(f"Frame 1 tracks: {tracks}")
    det1_moved = np.array([[105, 105, 205, 205, 0.9]])
    tracks = tracker.update(det1_moved)
    print(f"Frame 2 tracks: {tracks}")