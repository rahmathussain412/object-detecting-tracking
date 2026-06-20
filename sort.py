"""
sort.py
--------
A compact implementation of SORT (Simple Online and Realtime Tracking):
constant-velocity Kalman filter motion prediction combined with
IOU-based Hungarian assignment between predicted tracks and new detections.

This reflects the core ideas behind:
    Bewley, A. et al. "Simple Online and Realtime Tracking." ICIP 2016.

It has no deep appearance model (unlike DeepSORT), so it is faster and
has fewer dependencies, at the cost of being more easily confused when
objects overlap or occlude one another.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """Intersection-over-Union between two boxes in [x1, y1, x2, y2] format."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0.0


def bbox_to_state(bbox):
    """[x1,y1,x2,y2] -> [center_x, center_y, area, aspect_ratio] column vector."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ratio = w / float(h) if h != 0 else 0.0
    return np.array([cx, cy, area, ratio]).reshape((4, 1))


def state_to_bbox(state):
    """[center_x, center_y, area, aspect_ratio] -> [x1, y1, x2, y2]."""
    area, ratio = max(state[2], 0), max(state[3], 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w if w != 0 else 0
    cx, cy = state[0], state[1]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class Track:
    """A single tracked object, with its own constant-velocity Kalman filter."""
    _next_id = 1

    def __init__(self, bbox, class_name="object"):
        # State: [cx, cy, area, aspect_ratio, vcx, vcy, varea]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_state(bbox)

        self.id = Track._next_id
        Track._next_id += 1
        self.class_name = class_name
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return state_to_bbox(self.kf.x[:4].reshape(-1))

    def update(self, bbox, class_name=None):
        self.time_since_update = 0
        self.hits += 1
        if class_name:
            self.class_name = class_name
        self.kf.update(bbox_to_state(bbox))

    def get_state(self):
        return state_to_bbox(self.kf.x[:4].reshape(-1))


class Sort:
    """Tracker manager: handles association, track creation, aging, and removal."""

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections):
        """
        detections: list of (bbox=[x1,y1,x2,y2], confidence, class_name)
        returns: list of (x1, y1, x2, y2, track_id, class_name) for confirmed tracks
                 visible in the current frame.
        """
        predicted_boxes = [t.predict() for t in self.tracks]

        matches, unmatched_dets, _ = self._associate(detections, predicted_boxes)

        for trk_idx, det_idx in matches:
            bbox, _, cls_name = detections[det_idx]
            self.tracks[trk_idx].update(bbox, cls_name)

        for det_idx in unmatched_dets:
            bbox, _, cls_name = detections[det_idx]
            self.tracks.append(Track(bbox, cls_name))

        
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                x1, y1, x2, y2 = t.get_state()
                results.append((x1, y1, x2, y2, t.id, t.class_name))
        return results

    def _associate(self, detections, predicted_boxes):
        if len(predicted_boxes) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predicted_boxes)))

        iou_matrix = np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)
        for t, trk_box in enumerate(predicted_boxes):
            for d, (det_box, _, _) in enumerate(detections):
                iou_matrix[t, d] = iou(trk_box, det_box)

        row_idx, col_idx = linear_sum_assignment(-iou_matrix)  # maximize IOU

        matches, matched_trk, matched_det = [], set(), set()
        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_trk.add(r)
                matched_det.add(c)

        unmatched_trks = [i for i in range(len(predicted_boxes)) if i not in matched_trk]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]

        return matches, unmatched_dets, unmatched_trks
