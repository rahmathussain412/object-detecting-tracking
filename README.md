# Real-Time Object Detection and Tracking

A real-time computer vision pipeline that detects objects in a video stream
(webcam or file) using a pre-trained **YOLOv8** model and assigns each
detected object a persistent **tracking ID** as it moves across frames.

## Project Structure

```
object-detection-tracking/
├── main.py            # Primary pipeline: YOLOv8 + DeepSORT
├── main_sort.py        # Alternative pipeline: YOLOv8 + custom SORT
├── sort.py             # From-scratch SORT tracker (Kalman filter + Hungarian matching)
├── requirements.txt    # Python dependencies
└── README.md
```

## How It Works

1. **Video input** — `cv2.VideoCapture` reads frames from a webcam (index `0`)
   or a video file, frame by frame.
2. **Detection** — Each frame is passed to a YOLOv8 model (pre-trained on the
   COCO dataset), which returns bounding boxes, class labels, and confidence
   scores for every object found.
3. **Tracking** — The raw detections (which have no memory between frames)
   are handed to a tracker:
   - **DeepSORT** (`main.py`) combines a Kalman filter for motion prediction
     with a deep appearance embedding, so it can re-identify objects even
     after brief occlusion.
   - **SORT** (`main_sort.py` + `sort.py`) uses only a constant-velocity
     Kalman filter and IOU-based Hungarian matching — simpler and faster,
     but more easily confused when boxes overlap.
4. **Visualization** — Each tracked object is drawn with a bounding box,
   class label, and a stable tracking ID, plus a live FPS counter.

## Setup

```bash

python -m venv venv
source venv/bin/activate      


pip install -r requirements.txt
```

The first run will auto-download the YOLOv8 nano weights (`yolov8n.pt`, ~6MB).
You can swap in a bigger/more accurate model (`yolov8s.pt`, `yolov8m.pt`, etc.)
via the `--model` flag.

## Usage

**DeepSORT pipeline (recommended):**
```bash
python main.py --source 0                      
python main.py --source path/to/video.mp4       
python main.py --source 0 --save output.mp4     
python main.py --source 0 --classes 0 2         
```

**Custom SORT pipeline (lighter, fewer dependencies):**
```bash
python main_sort.py --source 0
```

Press **`q`** at any time to stop the stream.

### Useful flags
| Flag | Description | Default |
|---|---|---|
| `--source` | Webcam index or video file path | `0` |
| `--model` | YOLO weights file | `yolov8n.pt` |
| `--conf` | Detection confidence threshold | `0.4` |
| `--save` | Path to save annotated video | `None` |
| `--classes` | COCO class IDs to restrict detection to (main.py only) | all classes |

## Notes for the Report / Viva

- **Why YOLO?** It's a single-stage detector — it predicts boxes and classes
  in one forward pass, which is what makes real-time speed possible (compared
  to two-stage detectors like Faster R-CNN, which are typically more accurate
  but slower).
- **Why a tracker at all?** A detector alone has no concept of identity —
  re-running detection on every frame gives you boxes, not *which* box in
  frame 2 corresponds to which box in frame 1. SORT/DeepSORT solve this with
  motion prediction (Kalman filter) and an assignment step (Hungarian
  algorithm matching predicted positions to new detections).
- **DeepSORT vs SORT:** DeepSORT adds a learned appearance embedding on top
  of SORT's motion model, so it can still match an object to its correct ID
  after it's briefly hidden behind something else — SORT alone would likely
  assign it a new ID.

## Possible Extensions
- Swap YOLOv8 for a custom-trained model on a domain-specific dataset.
- Add per-class object counting or zone-crossing alerts.
- Export tracked trajectories to a CSV for downstream analytics.
