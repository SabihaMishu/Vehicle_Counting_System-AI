"""
settings.py
-----------
Central configuration for the Vehicle Counting System.
All tuneable parameters live here so nothing is hard-coded elsewhere.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH    = os.path.join(BASE_DIR, "database", "vehicle_counts.db")

# ── Dataset folder (which MVI_* folder to process) ────────────────────────
DATASET_FOLDER = "MVI_20012"        # change to MVI_20065 / MVI_39051 as needed
IMAGE_PATTERN  = "img*.jpg"         # glob pattern for frame images
VIDEO_FPS      = 25                 # frames-per-second for the synthesised video

# ── YOLO ──────────────────────────────────────────────────────────────────
YOLO_MODEL        = "yolov8n.pt"    # nano model; swap for yolov8s.pt / yolov8m.pt
YOLO_CONF         = 0.40            # minimum detection confidence
YOLO_IOU          = 0.45            # NMS IoU threshold
VEHICLE_CLASSES   = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# ── Tracker ───────────────────────────────────────────────────────────────
MAX_DISAPPEARED   = 30              # frames before a track is dropped
MAX_DISTANCE      = 80              # max centroid distance (px) for IoU assignment
IOU_THRESHOLD     = 0.30            # min IoU to associate detection→track

# ── Counting line (Vertical) ──────────────────────────────────────────────
# Expressed as a fraction of frame width (0.0 = left, 1.0 = right).
# The line spans the full height of the frame for left-to-right traffic.
LINE_POSITION     = 0.50            # 50 % across the frame (center)
CROSS_MARGIN      = 8               # pixel band around the line

# ── Visualisation ─────────────────────────────────────────────────────────
BOX_COLORS = {
    "car":        (0,   200, 0),    # green
    "motorcycle": (255, 128, 0),    # orange
    "bus":        (0,   100, 255),  # blue
    "truck":      (0,   0,   220),  # dark-blue
    "unknown":    (180, 180, 180),  # grey
}
LINE_COLOR        = (0, 255, 255)   # cyan counting line
TEXT_COLOR        = (255, 255, 255)
OVERLAY_ALPHA     = 0.45            # transparency of the HUD overlay panel
