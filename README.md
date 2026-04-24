# Vehicle Counting System

A real-time vehicle detection, tracking, and counting system built with **Python**, **YOLOv8**, and **SQLite**.

---

## Project Structure

```
vehicle_counting_system/
├── main.py                        # ← Entry point (run this)
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py                # All tuneable parameters
├── data/
│   ├── MVI_20012/                 # Dataset image frames
│   ├── MVI_20065/
│   └── MVI_39051/
├── database/
│   ├── __init__.py
│   └── database_manager.py        # SQLite: sessions, frame counts, daily summary
├── src/
│   ├── __init__.py
│   ├── video_converter.py         # Images → MP4
│   ├── detector.py                # YOLOv8 vehicle detection
│   ├── tracker.py                 # Centroid + IoU multi-object tracker
│   ├── counter.py                 # Horizontal line-crossing counter
│   ├── visualizer.py              # Bounding boxes + HUD overlay
│   └── pipeline.py                # Orchestrates all steps
├── output/                        # Generated videos, JSON, log (auto-created)
└── models/                        # YOLO weights cache (auto-downloaded)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on any Video File (MP4, AVI, etc.)

```bash
python main.py --input input/video.mp4
```

### 3. Run on an Image File

```bash
python main.py --input path/to/image.jpg
```

### 4. Run on a Dataset Folder (default behavior)

```bash
python main.py --input MVI_20012
```

---

## Output Files

Output filenames are generated dynamically based on the input name (e.g., `video_counted.mp4`).

| File | Description |
|---|---|
| `output/<name>_counted.mp4` | Annotated video with detections & counts |
| `output/<name>_counted.jpg` | Annotated image (if image input) |
| `output/daily_summary.json` | JSON export of cumulative daily counts |
| `database/vehicle_counts.db` | SQLite database with per-run session data |

---

## Dynamic Input Handling

The system automatically detects the input type:
- **Video:** Performs full detection + tracking + line-crossing counting.
- **Image:** Performs detection and per-frame counting only.
- **Dataset:** Converts images to video first, then runs the video pipeline.


---

## How It Works

| Step | Module | Description |
|---|---|---|
| 1 | `video_converter.py` | Sorts and assembles JPEG frames into MP4 via OpenCV |
| 2 | `detector.py` | YOLOv8n detects `car`, `motorcycle`, `bus`, `truck` per frame |
| 3 | `tracker.py` | IoU-based greedy matcher assigns persistent integer IDs |
| 4 | `counter.py` | Detects centroid crossing a horizontal line; counts once per vehicle |
| 5 | `visualizer.py` | Draws boxes, ID badges, counting line, and HUD panel |
| 6 | `pipeline.py` | Writes annotated frames to output video; commits to SQLite |
| 7 | `database_manager.py` | Stores per-frame counts + daily summary in SQLite |

---

## Configuration (`config/settings.py`)

| Parameter | Default | Description |
|---|---|---|
| `DATASET_FOLDER` | `MVI_20012` | Which data folder to process |
| `VIDEO_FPS` | `25` | Output video frame rate |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO weight file (auto-downloaded) |
| `YOLO_CONF` | `0.40` | Detection confidence threshold |
| `LINE_POSITION` | `0.55` | Counting line as fraction of frame height |
| `MAX_DISAPPEARED` | `30` | Frames before a lost track is pruned |
| `IOU_THRESHOLD` | `0.30` | Min IoU to match detection → existing track |

---

## SQLite Schema

```sql
sessions        -- one row per run
vehicle_counts  -- per-frame cumulative counts (FK → sessions)
daily_summary   -- daily rollup by dataset (upserted each run)
```
