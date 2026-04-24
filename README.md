# 🚗 Vehicle Counting System AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-red?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-green?style=for-the-badge&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**Real-time vehicle detection, tracking, and counting system powered by YOLOv8**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## ✨ Features

- 🎯 **Real-time Vehicle Detection** - State-of-the-art YOLOv8 nano model for fast inference
- 📍 **Multi-Object Tracking** - Centroid and IoU-based tracking with persistent vehicle IDs
- 📊 **Intelligent Counting** - Line-crossing detection for accurate vehicle counts
- 🎬 **Multi-format Support** - Process videos (MP4, AVI), images, or image datasets
- 💾 **SQLite Integration** - Persistent session tracking and daily summaries
- ⚙️ **Configurable Parameters** - Easy tuning of detection, tracking, and counting thresholds
- 📈 **Visual Analytics** - Real-time HUD overlay with detection boxes and counts
- 🚀 **High Performance** - Optimized pipeline for processing large video datasets

---

## 📋 Project Structure

```
vehicle_counting_system/
├── main.py                        # Entry point - Run this!
├── requirements.txt               # Python dependencies
├── config/
│   └── settings.py                # Centralized configuration
├── src/
│   ├── video_converter.py         # Images → MP4 conversion
│   ├── detector.py                # YOLOv8 vehicle detection
│   ├── tracker.py                 # Multi-object tracking engine
│   ├── counter.py                 # Line-crossing counter
│   ├── visualizer.py              # Annotation & visualization
│   └── pipeline.py                # Main orchestration logic
├── database/
│   └── database_manager.py        # SQLite management
├── data/                          # Dataset storage
├── models/                        # YOLO weights cache
└── output/                        # Generated results
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/SabihaMishu/Vehicle_Counting_System-AI.git
cd Vehicle_Counting_System-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Video File
```bash
python main.py --input path/to/video.mp4
```

### 4. Run on Image
```bash
python main.py --input path/to/image.jpg
```

### 5. Run on Dataset (Default)
```bash
python main.py --input MVI_20012
```

---

## 📊 Output Files

| Output | Description |
|--------|-------------|
| `output/<name>_counted.mp4` | 🎬 Annotated video with detections and counts |
| `output/<name>_counted.jpg` | 🖼️ Annotated image (for image inputs) |
| `output/daily_summary.json` | 📈 JSON export of cumulative daily counts |
| `database/vehicle_counts.db` | 💾 SQLite database with session data |

---

## 🏗️ How It Works

The system follows a modular pipeline architecture:

```
Input (Video/Image/Dataset)
    ↓
[1] Video Converter → Convert image frames to MP4
    ↓
[2] YOLOv8 Detector → Detect vehicles (car, motorcycle, bus, truck)
    ↓
[3] Multi-Object Tracker → Assign persistent IDs with IoU matching
    ↓
[4] Line-Crossing Counter → Count vehicles crossing detection line
    ↓
[5] Visualizer → Draw annotations & HUD overlay
    ↓
[6] Database Manager → Store results in SQLite
    ↓
Output (Annotated Video + Analytics)
```

### Component Breakdown

| Component | Module | Purpose |
|-----------|--------|---------|
| **Detection** | `detector.py` | YOLOv8 nano model for vehicle detection |
| **Tracking** | `tracker.py` | IoU-based greedy matching for persistent IDs |
| **Counting** | `counter.py` | Horizontal line-crossing detection |
| **Visualization** | `visualizer.py` | Bounding boxes, IDs, counting line, HUD |
| **Pipeline** | `pipeline.py` | Orchestrates entire workflow |
| **Database** | `database_manager.py` | SQLite session & summary storage |

---

## ⚙️ Configuration

All parameters can be tuned in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_FOLDER` | `MVI_20012` | Input dataset folder |
| `VIDEO_FPS` | `25` | Output video frame rate |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (nano - fast & lightweight) |
| `YOLO_CONF` | `0.40` | Detection confidence threshold |
| `LINE_POSITION` | `0.55` | Counting line (% of frame height) |
| `MAX_DISAPPEARED` | `30` | Frames before track removal |
| `IOU_THRESHOLD` | `0.30` | Min IoU for track matching |

---

## 📦 Dependencies

- **ultralytics** - YOLOv8 detection framework
- **opencv-python** - Video/image processing
- **numpy** - Numerical computing
- **sqlite3** - Database management (built-in)

See `requirements.txt` for complete list.

---

## 🎯 Supported Vehicle Classes

The system detects and counts:
- 🚗 Cars
- 🏍️ Motorcycles
- 🚌 Buses
- 🚚 Trucks

---

## 📈 Performance Metrics

- **Detection Speed**: Real-time inference on CPU/GPU
- **Tracking Accuracy**: Persistent ID assignment with IoU matching
- **Database**: Efficient SQLite storage with indexed queries
- **Output**: Configurable video quality and FPS

---

## 🔧 Troubleshooting

**Issue: Out of memory on large videos?**
- Reduce `VIDEO_FPS` in settings
- Process shorter video segments

**Issue: Missing detections?**
- Lower `YOLO_CONF` threshold (more detections, may include false positives)
- Ensure good lighting in video

**Issue: Counting inaccuracies?**
- Adjust `LINE_POSITION` for better line placement
- Tune `IOU_THRESHOLD` for track matching

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📧 Contact

For questions or feedback, reach out on GitHub: [@SabihaMishu](https://github.com/SabihaMishu)

---

<div align="center">

**Made with ❤️ for intelligent traffic monitoring**

</div>
