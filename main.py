"""
main.py
-------
Dynamic entry point for the Vehicle Counting System.
Updated for VERTICAL line crossing and simplified counting (no directions).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# ── allow imports from project root ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config.settings     import (
    DATA_DIR, OUTPUT_DIR, DB_PATH,
    VIDEO_FPS, IMAGE_PATTERN,
    YOLO_MODEL, YOLO_CONF, YOLO_IOU,
    MAX_DISAPPEARED, IOU_THRESHOLD,
    LINE_POSITION, CROSS_MARGIN,
    OVERLAY_ALPHA,
)
from src.video_converter import images_to_video
from src.detector        import VehicleDetector
from src.tracker         import VehicleTracker
from src.counter         import LineCounter
from src.visualizer      import Visualizer
from src.pipeline        import VehiclePipeline
from src.image_processor import ImageProcessor
from src.input_resolver  import resolve, InputType
from database.database_manager import DatabaseManager

import cv2


# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log"), mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ── helpers ───────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _get_media_dims(path: str, is_video: bool) -> tuple:
    if is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return 0, 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    else:
        img = cv2.imread(path)
        if img is None: return 0, 0
        h, w = img.shape[:2]
        return w, h


def _print_summary(counts: Dict[str, int], output_path: str) -> None:
    border = "─" * 40
    print(f"\n{'':>4}┌{border}┐")
    print(f"{'':>4}│{'  VEHICLE COUNTING RESULTS':^40}│")
    print(f"{'':>4}├{border}┤")
    for label, cnt in counts.items():
        print(f"{'':>4}│  {label.upper():<15} {cnt:>18}  │")
    print(f"{'':>4}├{border}┤")
    print(f"{'':>4}│  {'TOTAL':<15} {sum(counts.values()):>18}  │")
    print(f"{'':>4}└{border}┘")
    print(f"\n  ✅  Output saved to:\n      {output_path}\n")


# ── main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vertical Vehicle Counting System")
    parser.add_argument("--input", required=True, help="Path to video/image or dataset name.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip img2vid conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(OUTPUT_DIR)
    _ensure_dir(os.path.dirname(DB_PATH))

    try:
        res = resolve(args.input, DATA_DIR, OUTPUT_DIR)
    except Exception as e:
        log.error(str(e))
        return

    log.info("Input Type: %s | Source: %s", res.input_type.name, res.source_path)

    if res.needs_conversion:
        if not args.skip_convert or not os.path.exists(res.video_path):
            images_to_video(res.source_path, res.video_path, fps=VIDEO_FPS, pattern=IMAGE_PATTERN)

    w, h = _get_media_dims(
        res.video_path if res.input_type != InputType.IMAGE else res.source_path,
        is_video=(res.input_type != InputType.IMAGE)
    )
    
    # CALCULATE VERTICAL LINE X COORDINATE
    line_x = int(w * LINE_POSITION)
    log.info("Resolution: %dx%d | Vertical Line at x=%d (%.0f%%)", w, h, line_x, LINE_POSITION*100)

    detector   = VehicleDetector(model_name=YOLO_MODEL, conf=YOLO_CONF, iou=YOLO_IOU)
    visualizer = Visualizer(line_x=line_x, overlay_alpha=OVERLAY_ALPHA)

    final_counts = {}
    output_path = ""

    with DatabaseManager(DB_PATH) as db:
        if res.input_type == InputType.IMAGE:
            output_path = os.path.join(OUTPUT_DIR, f"{res.output_stem}_counted.jpg")
            processor = ImageProcessor(detector, visualizer, db, res.dataset_name)
            processor.run(res.source_path, output_path)
            final_counts = processor.counts
        else:
            output_path = os.path.join(OUTPUT_DIR, f"{res.output_stem}_counted.mp4")
            tracker = VehicleTracker(max_disappeared=MAX_DISAPPEARED)
            counter = LineCounter(line_x=line_x, cross_margin=CROSS_MARGIN)
            
            pipeline = VehiclePipeline(
                detector=detector, tracker=tracker, counter=counter,
                visualizer=visualizer, db=db, dataset=res.dataset_name,
            )
            pipeline.run(res.video_path, output_path)
            final_counts = counter.counts

        summary = db.get_daily_summary()
        with open(os.path.join(OUTPUT_DIR, "daily_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    _print_summary(final_counts, output_path)


if __name__ == "__main__":
    main()
