"""
Real-Time Object Detection and Tracking
----------------------------------------
Internship Project: Object Detection and Tracking using YOLOv8 + DeepSORT

Pipeline:
1. Capture video frames from a webcam or video file using OpenCV.
2. Run each frame through a pre-trained YOLOv8 model to detect objects.
3. Draw bounding boxes and class labels for each detection.
4. Feed detections into a DeepSORT tracker to maintain consistent IDs
   for each object across frames.
5. Display the annotated video stream in real time, with an option to
   save the result to disk.

Usage:
    python main.py --source 0                     # webcam
    python main.py --source video.mp4              # video file
    python main.py --source 0 --save output.mp4    # save annotated output
    python main.py --source 0 --classes 0 2        # only track classes 0 (person) and 2 (car)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking (YOLOv8 + DeepSORT)")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Path or name of the YOLO model weights (auto-downloads if not present)")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Confidence threshold for detections")
    parser.add_argument("--save", type=str, default=None,
                         help="Optional path to save the annotated output video")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                         help="Optional list of COCO class IDs to keep (default: all classes)")
    return parser.parse_args()


def get_video_source(source: str):
    """Return an int (webcam index) if source is numeric, else treat it as a file path."""
    return int(source) if source.isdigit() else source


def generate_color(seed: int):
    """Deterministic BGR color for a given tracking ID, so each ID keeps a stable color."""
    rng = np.random.RandomState(seed % (2 ** 31))
    return tuple(int(c) for c in rng.randint(0, 255, size=3))


def draw_track(frame, x1, y1, x2, y2, label, track_id):
    color = generate_color(abs(hash(str(track_id))))
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

    print("[INFO] Initializing DeepSORT tracker...")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    
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

        
        results = model(frame, conf=args.conf, classes=args.classes, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
    _name)
            detections.append(([x1, y1, w, h], conf, model.names[cls_id]))

        
        tracks = tracker.update_tracks(detections, frame=frame)

        
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class() or "object"
            draw_track(frame, x1, y1, x2, y2, label, track.track_id)

        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Object Detection and Tracking (YOLOv8 + DeepSORT)", frame)
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
