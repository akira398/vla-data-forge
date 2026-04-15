"""
File I/O utilities.

Shared helpers for reading and writing episodes, JSONL files, and images.
All functions work with pathlib.Path and accept str paths too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def save_jsonl(
    records: List[Dict[str, Any]],
    path: Union[str, Path],
    mode: str = "w",
) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """
    Lazily iterate over a JSONL file, yielding one dict per line.

    Usage
    -----
    for record in load_jsonl("episodes.jsonl"):
        process(record)
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl_lines(path: Union[str, Path]) -> int:
    """Count lines in a JSONL file without loading all content."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Episode JSON (single-file format)
# ---------------------------------------------------------------------------


def save_episode_json(episode_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a single episode dict as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode_dict, f, indent=2, ensure_ascii=False)


def load_episode_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def save_image(arr: np.ndarray, path: Union[str, Path], quality: int = 95) -> None:
    """Save a uint8 numpy array as PNG or JPEG based on extension."""
    from PIL import Image as PILImage
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = PILImage.fromarray(arr.astype(np.uint8))
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, format="JPEG", quality=quality)
    else:
        img.save(path, format="PNG")


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image file as a uint8 numpy array (H, W, 3)."""
    from PIL import Image as PILImage
    return np.array(PILImage.open(path).convert("RGB"), dtype=np.uint8)


def resize_image(
    arr: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Resize image array to (H, W) using bilinear interpolation."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(arr.astype(np.uint8))
    pil = pil.resize((size[1], size[0]), PILImage.BILINEAR)
    return np.array(pil)
