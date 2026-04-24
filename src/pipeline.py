"""
pipeline.py
-----------
Orchestrates the end-to-end processing pipeline for video inputs.
Updated for simple counting and vertical line crossing.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2

from src.detector    import VehicleDetector
from src.tracker     import VehicleTracker
from src.counter     import LineCounter
from src.visualizer  import Visualizer
from database.database_manager import DatabaseManager

log = logging.getLogger(__name__)

_DB_COMMIT_INTERVAL = 50


class VehiclePipeline:
    def __init__(
        self,
        detector:   VehicleDetector,
        tracker:    VehicleTracker,
        counter:    LineCounter,
        visualizer: Visualizer,
        db:         DatabaseManager,
        dataset:    str,
    ) -> None:
        self._detector   = detector
        self._tracker    = tracker
        self._counter    = counter
        self._viz        = visualizer
        self._db         = db
        self._dataset    = dataset

    def run(self, input_video: str, output_video: str) -> None:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {input_video}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log.info("Video: %s (%dx%d, %.1f fps)", input_video, width, height, fps)

        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not writer.isOpened(): raise RuntimeError(f"VideoWriter failed: {output_video}")

        session_id = self._db.create_session(self._dataset)
        frame_num  = 0
        t_start    = time.monotonic()

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_num += 1
                timestamp  = frame_num / fps

                detections = self._detector.track(frame)
                tracks     = self._tracker.update(detections)
                
                self._counter.update(tracks)
                self._counter.cleanup_stale_tracks(set(tracks.keys()))

                self._db.insert_frame_count(session_id, frame_num, timestamp, self._counter.counts)
                if frame_num % _DB_COMMIT_INTERVAL == 0: self._db.batch_commit()

                elapsed  = time.monotonic() - t_start
                live_fps = frame_num / elapsed if elapsed > 0 else 0.0
                annotated = self._viz.draw(frame, tracks, self._counter.counts, frame_num, live_fps)
                writer.write(annotated)

                if frame_num % 100 == 0:
                    log.info("Frame %d/%d | Total Count: %d", frame_num, total, self._counter.grand_total)

        finally:
            self._db.batch_commit()
            cap.release()
            writer.release()

        self._db.close_session(session_id, self._counter.grand_total)
        self._db.upsert_daily_summary(self._dataset, self._counter.counts)

        elapsed = time.monotonic() - t_start
        log.info("Pipeline done. %d frames in %.1fs.", frame_num, elapsed)
        log.info("Final counts: %s", self._counter.counts)
