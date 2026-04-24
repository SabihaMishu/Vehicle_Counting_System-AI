"""
visualizer.py
-------------
All drawing responsibilities: bounding boxes, track IDs, the vertical 
counting line, and the simplified HUD overlay panel.
"""

import logging
from typing import Dict

import cv2
import numpy as np

from src.tracker import Track

log = logging.getLogger(__name__)

# ── colour palette (BGR) ────────────────────────────────────────────────────
_BOX_COLORS: Dict[str, tuple] = {
    "car":        (0,   200, 0),
    "motorcycle": (0,   128, 255),
    "bus":        (255, 100, 0),
    "truck":      (0,   0,   220),
    "unknown":    (180, 180, 180),
}
_LINE_COLOR  = (0, 255, 255)  # cyan
_TEXT_COLOR  = (255, 255, 255)
_HUD_COLOR   = (15,  15,  30)  # dark navy
_FONT        = cv2.FONT_HERSHEY_DUPLEX
_FONT_SMALL  = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """
    Draws all visual elements onto a frame.

    Parameters
    ----------
    line_x       : int   x-pixel of the vertical counting line.
    overlay_alpha: float Transparency of the HUD overlay.
    """

    def __init__(self, line_x: int, overlay_alpha: float = 0.45) -> None:
        self._line_x = line_x
        self._alpha  = overlay_alpha

    def draw(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Track],
        counts: Dict[str, int],
        frame_num: int,
        fps: float,
    ) -> np.ndarray:
        frame = self._draw_line(frame)
        frame = self._draw_tracks(frame, tracks)
        frame = self._draw_hud(frame, counts, frame_num, fps)
        return frame

    def _draw_line(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # Vertical line from top to bottom
        cv2.line(frame, (self._line_x, 0), (self._line_x, h), _LINE_COLOR, 2)
        cv2.putText(
            frame, "COUNTING LINE",
            (self._line_x + 10, 30),
            _FONT_SMALL, 0.6, _LINE_COLOR, 2, cv2.LINE_AA,
        )
        return frame

    def _draw_tracks(self, frame: np.ndarray, tracks: Dict[int, Track]) -> np.ndarray:
        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue

            color = _BOX_COLORS.get(track.label, _BOX_COLORS["unknown"])
            x1, y1, x2, y2 = track.bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            badge_text = f"{track.label.upper()} #{tid}"
            cv2.putText(frame, badge_text, (x1, y1 - 10), _FONT_SMALL, 0.5, color, 1, cv2.LINE_AA)
            cv2.circle(frame, track.centroid, 4, color, -1)
        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        counts: Dict[str, int],
        frame_num: int,
        fps: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w, panel_h = 240, 180
        margin = 10
        px, py = margin, margin

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), _HUD_COLOR, -1)
        cv2.addWeighted(overlay, self._alpha, frame, 1 - self._alpha, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), _LINE_COLOR, 1)

        # Content
        cv2.putText(frame, "VEHICLE COUNTER", (px + 10, py + 25), _FONT, 0.6, _LINE_COLOR, 1, cv2.LINE_AA)
        cv2.line(frame, (px + 10, py + 30), (px + panel_w - 10, py + 30), _LINE_COLOR, 1)

        y_off = py + 55
        for label, cnt in counts.items():
            color = _BOX_COLORS.get(label, _BOX_COLORS["unknown"])
            text = f"{label.upper():<12} {cnt:>5}"
            cv2.putText(frame, text, (px + 10, y_off), _FONT_SMALL, 0.5, color, 1, cv2.LINE_AA)
            y_off += 25

        cv2.line(frame, (px + 10, y_off), (px + panel_w - 10, y_off), (100, 100, 100), 1)
        total_text = f"TOTAL        {sum(counts.values()):>5}"
        cv2.putText(frame, total_text, (px + 10, y_off + 25), _FONT, 0.6, _TEXT_COLOR, 1, cv2.LINE_AA)

        return frame
