"""
image_processor.py
------------------
Handles single image processing: detection and counting.
Updated to use simple class-based tallying (no directions).
"""

import logging
import os
from pathlib import Path
from typing import Dict

import cv2

from src.detector import VehicleDetector
from src.tracker import Track
from src.visualizer import Visualizer
from database.database_manager import DatabaseManager

log = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(
        self,
        detector: VehicleDetector,
        visualizer: Visualizer,
        db: DatabaseManager,
        dataset: str,
    ) -> None:
        self._detector = detector
        self._viz = visualizer
        self._db = db
        self._dataset = dataset
        self._counts: Dict[str, int] = {}

    @property
    def counts(self) -> Dict[str, int]:
        return self._counts

    def run(self, input_path: str, output_path: str) -> None:
        frame = cv2.imread(input_path)
        frame = cv2.resize(frame, (1280, 720))  # Resize the frame
        if frame is None: raise RuntimeError(f"Cannot read image: {input_path}")

        log.info("Processing image: %s", input_path)

        detections = self._detector.detect(frame)
        
        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        fake_tracks: Dict[int, Track] = {}
        
        for i, det in enumerate(detections):
            label = det.label if det.label in counts else "car"
            counts[label] += 1
            fake_tracks[i] = Track(track_id=i, label=det.label, bbox=det.bbox, centroid=det.centroid)

        self._counts = counts
        annotated = self._viz.draw(frame, fake_tracks, counts, 1, 0.0)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)

        session_id = self._db.create_session(self._dataset)
        self._db.insert_frame_count(session_id, 1, 0.0, counts)
        self._db.close_session(session_id, sum(counts.values()))
        self._db.upsert_daily_summary(self._dataset, counts)
        
        log.info("Image counts: %s | Total: %d", counts, sum(counts.values()))
        print(f"Detected vehicles: {len(detections)}")  # Print the number of detected vehicles