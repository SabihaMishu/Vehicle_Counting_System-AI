"""
tracker.py
----------
Modern tracker wrapper using YOLOv8's built-in tracking IDs.
Instead of manual IoU matching, it simply maps YOLO track IDs
to persistent Track objects.
"""

import logging
from typing import Dict, List, Tuple

from src.detector import Detection

log = logging.getLogger(__name__)


class Track:
    """State for a single tracked vehicle."""
    def __init__(self, track_id: int, label: str, bbox: Tuple[int, int, int, int], centroid: Tuple[int, int]):
        self.track_id    = track_id
        self.label       = label
        self.bbox        = bbox
        self.centroid    = centroid
        self.centroids   = [centroid]
        self.disappeared = 0

    def update(self, det: Detection) -> None:
        self.bbox       = det.bbox
        self.centroid   = det.centroid
        self.disappeared = 0
        self.centroids.append(det.centroid)


class VehicleTracker:
    """
    Maintains a set of active tracks by mapping YOLO's persistent IDs
    to our internal metadata objects.
    """

    def __init__(self, max_disappeared: int = 30) -> None:
        self._tracks: Dict[int, Track] = {}
        self._max_disappeared = max_disappeared

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """
        Update the track state using detections that already contain track IDs.
        """
        active_ids = []

        for det in detections:
            if det.track_id is None:
                continue
            
            tid = det.track_id
            active_ids.append(tid)

            if tid in self._tracks:
                self._tracks[tid].update(det)
            else:
                # Register new track
                self._tracks[tid] = Track(
                    track_id=tid,
                    label=det.label,
                    bbox=det.bbox,
                    centroid=det.centroid
                )

        # Age and prune tracks that weren't in current detections
        current_active = set(active_ids)
        to_delete = []
        for tid, track in self._tracks.items():
            if tid not in current_active:
                track.disappeared += 1
                if track.disappeared > self._max_disappeared:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self._tracks[tid]
            log.debug("Track %d pruned", tid)

        return self._tracks
