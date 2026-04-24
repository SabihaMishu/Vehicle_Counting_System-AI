"""
counter.py
----------
Counts vehicles that cross a vertical line (x-coordinate).
Simplified logic: any crossing is counted once.
No IN/OUT direction filtering as requested.
"""

import logging
from typing import Dict, Set

from src.tracker import Track

log = logging.getLogger(__name__)

# Side constants
_LEFT  = "left"
_RIGHT = "right"
_ON    = "on"


class LineCounter:
    """
    Detects when tracked vehicles cross a vertical counting line
    and tallies them by vehicle class.
    
    Parameters
    ----------
    line_x       : int   Pixel x-coordinate of the vertical counting line.
    cross_margin : int   Pixel band around the line used to suppress jitter.
    """

    def __init__(self, line_x: int, cross_margin: int = 8) -> None:
        self._line_x       = line_x
        self._margin       = cross_margin
        self._prev_side:   Dict[int, str]  = {}   # track_id → last side
        self._counted:     Dict[int, bool] = {}   # track_id → already counted?
        self.counts: Dict[str, int] = {
            "car": 0, "motorcycle": 0, "bus": 0, "truck": 0
        }

    @property
    def grand_total(self) -> int:
        return sum(self.counts.values())

    def update(self, tracks: Dict[int, Track]) -> None:
        """
        Call once per frame with the current set of active tracks.
        """
        for tid, track in tracks.items():
            # Check the X coordinate for vertical line crossing
            cx = track.centroid[0]
            side = self._classify(cx)

            if tid not in self._prev_side:
                self._prev_side[tid] = side
                self._counted[tid]   = False
                continue

            prev = self._prev_side[tid]

            # A valid crossing: was on one clear side, now on the other
            if (
                not self._counted[tid]
                and side != _ON
                and prev != _ON
                and prev != side
            ):
                label = track.label if track.label in self.counts else "car"
                self.counts[label] += 1
                self._counted[tid]  = True
                log.info(
                    "Track %d (%s) crossed vertical line at x=%d | total: %d",
                    tid, label, cx, self.grand_total,
                )

            # Update previous side (ignore ON band to prevent multiple counts)
            if side != _ON:
                self._prev_side[tid] = side

    def cleanup_stale_tracks(self, active_ids: Set[int]) -> None:
        """Remove state for tracks that are no longer active."""
        stale = set(self._prev_side) - active_ids
        for tid in stale:
            self._prev_side.pop(tid, None)
            self._counted.pop(tid, None)

    def _classify(self, cx: int) -> str:
        if cx < self._line_x - self._margin:
            return _LEFT
        if cx > self._line_x + self._margin:
            return _RIGHT
        return _ON
