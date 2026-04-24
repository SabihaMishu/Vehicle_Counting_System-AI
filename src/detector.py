"""
detector.py
-----------
YOLO-based vehicle detector.
Wraps Ultralytics YOLOv8 and returns only vehicle detections in a
clean, framework-agnostic format.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

# COCO class IDs for vehicles we care about
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}          # car, motorcycle, bus, truck

# Map COCO class_id → label string
_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """One vehicle detection in a single frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    class_id:   int
    track_id:   Optional[int] = None

    @property
    def centroid(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


class VehicleDetector:
    """
    Loads a YOLOv8 model and runs inference on individual frames.

    Parameters
    ----------
    model_name : str   Path or name of the YOLO model (e.g. 'yolov8n.pt').
    conf       : float Minimum confidence threshold.
    iou        : float NMS IoU threshold.
    device     : str   'cpu', 'cuda', '0', etc.  Empty string = auto.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.3,
        iou: float = 0.45,
        device: str = "",
    ) -> None:
        log.info("Loading YOLO model: %s", model_name)
        self._model = YOLO(model_name)
        self._conf  = conf
        self._iou   = iou
        self._device = device or None
        log.info("Detector ready (conf=%.2f, iou=%.2f)", conf, iou)

    # ── public API ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on *frame* (BGR numpy array).

        Returns a list of :class:`Detection` objects, filtered to
        vehicle classes only.
        """
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
            classes=list(_VEHICLE_CLASS_IDS),
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in _VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=_CLASS_NAMES.get(cls_id, "unknown"),
                        confidence=float(box.conf[0].item()),
                        class_id=cls_id,
                    )
                )
        return detections
    def track(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO tracking on *frame*.
        Returns a list of :class:`Detection` objects with track IDs.
        """
        results = self._model.track(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
            persist=True,
            classes=list(_VEHICLE_CLASS_IDS),
            tracker="botsort.yaml" # BoT-SORT is a modern successor to DeepSORT
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in _VEHICLE_CLASS_IDS:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get track ID if available
                tid = None
                if box.id is not None:
                    tid = int(box.id[0].item())
                
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=_CLASS_NAMES.get(cls_id, "unknown"),
                        confidence=float(box.conf[0].item()),
                        class_id=cls_id,
                        track_id=tid
                    )
                )
        return detections
