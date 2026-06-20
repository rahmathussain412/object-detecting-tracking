"""
main_sort.py
-------------
Alternative pipeline: YOLOv8 detection + classic SORT tracking
(Kalman filter motion model + IOU/Hungarian matching), using the
from-scratch tracker in sort.py instead of DeepSORT.

Lighter weight than main.py (no appearance/re-identification model),
useful when extra dependencies aren't available or when objects are
unlikely to occlude each other heavily.

Usage:
    python main_sort.py --source 0
    python main_sort.py --source video.mp4 --save output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking (YOLOv8 + SORT)")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Path or name of the YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Confidence threshold for detections")
    parser.add_argument("--save", type=str, default=None,
                         help="Optional path to save the annotated output video")
    return parser.parse_args()


def get_video_source(source: str):
    return int(source) if source.isdigit() else source


def generate_color(seed: int):
    rng = np.random.RandomState(seed % (2 ** 31))
    return tuple(int(c) for c in rng.randint(0, 255, size=3))


def draw_track(frame, x1, y1, x2, y2, label, track_id):
    color = generate_color(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} | ID {track_id}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    args = parse_args()

    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model)

    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    source = get_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))

    prev_time = time.time()
    print("[INFO] Starting video stream... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            detections.append(([x1, y1, x2, y2], float(box.conf[0]), model.names[cls_id]))

        tracked_objects = tracker.update(detections)

        for x1, y1, x2, y2, track_id, cls_name in tracked_objects:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            draw_track(frame, x1, y1, x2, y2, cls_name, track_id)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Object Detection and Tracking (YOLOv8 + SORT)", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
