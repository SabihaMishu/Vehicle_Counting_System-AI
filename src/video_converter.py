"""
video_converter.py
------------------
Converts a folder of sequentially named images into an MP4 video file.
"""

import glob
import logging
import os
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def images_to_video(
    image_dir: str,
    output_path: str,
    fps: int = 25,
    pattern: str = "img*.jpg",
) -> str:
    """
    Read all images matching *pattern* inside *image_dir* (sorted),
    write them as an MP4 to *output_path*, and return the output path.

    Raises
    ------
    FileNotFoundError  – if *image_dir* does not exist or has no matching frames.
    RuntimeError       – if OpenCV cannot open the VideoWriter.
    """
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    frames = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not frames:
        raise FileNotFoundError(
            f"No images matching '{pattern}' found in {image_dir}"
        )

    # Read first frame to derive resolution
    first = cv2.imread(frames[0])
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    h, w = first.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")

    log.info(
        "Converting %d frames (%dx%d @ %d fps) → %s",
        len(frames), w, h, fps, output_path,
    )

    for idx, path in enumerate(frames):
        frame = cv2.imread(path)
        if frame is None:
            log.warning("Skipping unreadable frame %d: %s", idx, path)
            continue
        writer.write(frame)

    writer.release()
    log.info("Video saved: %s", output_path)
    return output_path
