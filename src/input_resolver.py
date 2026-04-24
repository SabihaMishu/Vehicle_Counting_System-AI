"""
input_resolver.py
-----------------
Determines the type of input the user has provided and resolves all
relevant paths so that the rest of the system can stay agnostic about
where the data came from.

Supported input types
---------------------
  VIDEO   - any MP4 / AVI / MOV / MKV file
  IMAGE   - any JPG / PNG / BMP file
  DATASET - a folder whose name matches a subfolder in data/
"""

import os
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path


class InputType(Enum):
    VIDEO   = auto()
    IMAGE   = auto()
    DATASET = auto()


_VIDEO_EXT  = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
_IMAGE_EXT  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class ResolvedInput:
    """Everything the pipeline needs to know about the user's input."""
    input_type:  InputType
    source_path: str          # original file / folder path
    video_path:  str          # path to the video to process (may need conversion)
    output_stem: str          # base name used to build output file names
    dataset_name: str         # label stored in the DB (same as output_stem)
    needs_conversion: bool    # True when source is an image dataset folder


def resolve(raw_input: str, data_dir: str, output_dir: str) -> ResolvedInput:
    """
    Accept whatever the user typed (file path, folder name, absolute path)
    and return a fully resolved :class:`ResolvedInput`.

    Parameters
    ----------
    raw_input  : The --input argument value from the CLI.
    data_dir   : The project's data/ directory (for dataset folders).
    output_dir : Where converted / output videos are placed.
    """
    p = Path(raw_input)

    # ── 1. Absolute or relative path to an EXISTING file ──────────────────
    if p.is_file():
        ext = p.suffix.lower()
        if ext in _VIDEO_EXT:
            stem = p.stem
            return ResolvedInput(
                input_type=InputType.VIDEO,
                source_path=str(p),
                video_path=str(p),
                output_stem=stem,
                dataset_name=stem,
                needs_conversion=False,
            )
        if ext in _IMAGE_EXT:
            stem = p.stem
            return ResolvedInput(
                input_type=InputType.IMAGE,
                source_path=str(p),
                video_path=str(p),          # not used for image mode
                output_stem=stem,
                dataset_name=stem,
                needs_conversion=False,
            )
        raise ValueError(f"Unsupported file type: {p.suffix}")

    # ── 2. Existing DIRECTORY (dataset folder) ─────────────────────────────
    # Accept full path OR just the folder name (relative to data/)
    if p.is_dir():
        stem  = p.name
        vpath = os.path.join(output_dir, f"{stem}_input.mp4")
        return ResolvedInput(
            input_type=InputType.DATASET,
            source_path=str(p),
            video_path=vpath,
            output_stem=stem,
            dataset_name=stem,
            needs_conversion=True,
        )

    # ── 3. Bare folder name inside data/ ──────────────────────────────────
    candidate = Path(data_dir) / raw_input
    if candidate.is_dir():
        stem  = raw_input
        vpath = os.path.join(output_dir, f"{stem}_input.mp4")
        return ResolvedInput(
            input_type=InputType.DATASET,
            source_path=str(candidate),
            video_path=vpath,
            output_stem=stem,
            dataset_name=stem,
            needs_conversion=True,
        )

    raise FileNotFoundError(
        f"Input not found: '{raw_input}'\n"
        "  Provide: a video file path, an image file path, "
        "or a dataset folder name inside data/"
    )
