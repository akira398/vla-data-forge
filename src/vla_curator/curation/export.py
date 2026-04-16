"""
Export writers for curated datasets.

Supported formats
-----------------
JSONL   — One JSON object per line, human-readable, streamable.
          Images saved as separate PNG files; paths stored in the JSON.
          This is the default and recommended format for research use.

HDF5    — (Placeholder) Planned for training-scale data where I/O throughput
          is the bottleneck.  Will store images as uint8 arrays and use
          chunked/compressed storage for actions.

Design
------
Exporters write to a directory, not a single file.  For JSONL:
  output_dir/
    episodes.jsonl      — one episode per line
    images/
      <episode_id>/
        step_000.png
        step_001.png
        ...
    metadata.json       — dataset-level stats and config

The image directory layout mirrors episode_id so it is easy to inspect.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

from ..schemas.interleaved import InterleavedEpisode

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    JSONL = "jsonl"
    HDF5 = "hdf5"
    RLDS = "rlds"


class BaseExporter(ABC):
    """Abstract base for all exporters."""

    @abstractmethod
    def export_episode(self, episode: InterleavedEpisode, split: str = "train") -> None:
        ...

    def export_dataset(
        self,
        episodes: Iterator[InterleavedEpisode],
        total: Optional[int] = None,
    ) -> int:
        """Export all episodes. Returns number written."""
        count = 0
        for ep in tqdm(episodes, total=total, desc="Exporting", unit="ep"):
            self.export_episode(ep)
            count += 1
        logger.info("Exported %d episodes.", count)
        return count

    @abstractmethod
    def write_metadata(self, stats: Dict[str, Any]) -> None:
        ...


class JSONLExporter(BaseExporter):
    """
    Export curated episodes to JSONL format.

    Parameters
    ----------
    output_dir : Path
        Root directory for output files.
    save_images : bool
        If True, encode images as base64 in the JSONL (not recommended
        for large datasets — use False and save to disk instead).
    image_dir : Optional[Path]
        Where to save frame PNG files when save_images=False.
        Defaults to output_dir/images.
    """

    def __init__(
        self,
        output_dir: Path,
        save_images: bool = False,
        image_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        self.image_dir = image_dir or (self.output_dir / "images")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not save_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.output_dir / "episodes.jsonl"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def export_episode(self, episode: InterleavedEpisode) -> None:
        """Serialize one episode to the JSONL file and save its images to disk."""
        if not self.save_images:
            self._save_images_to_disk(episode)

        d = episode.to_dict(include_images=self.save_images)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def write_metadata(self, stats: Dict[str, Any]) -> None:
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("Metadata written to %s", meta_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_images_to_disk(self, episode: InterleavedEpisode) -> None:
        """
        Save each step's image as a PNG file and update the path in the
        episode's observation in-place.
        """
        ep_dir = self.image_dir / _safe_path_component(episode.episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)

        for step in episode.steps:
            obs = step.observation
            img = obs.load_image()
            if img is None:
                continue
            img_path = ep_dir / f"step_{step.step_index:05d}.png"
            if not img_path.exists():
                PILImage.fromarray(img.astype(np.uint8)).save(img_path)
            # Update the step to reference the saved path
            obs.image = None
            obs.image_path = str(img_path)


def _safe_path_component(s: str) -> str:
    """Convert an episode ID to a safe directory name."""
    return s.replace("/", "_").replace("\\", "_").replace(":", "_").strip("_")


# ---------------------------------------------------------------------------
# HDF5 exporter (placeholder)
# ---------------------------------------------------------------------------


class HDF5Exporter(BaseExporter):
    """
    Placeholder for HDF5 export.

    TODO: Implement using h5py with:
      - One HDF5 file per episode  OR  one file per shard of N episodes
      - Datasets: images (uint8, chunked), actions (float32), states (float32)
      - String datasets: task_description, reasoning fields
      - Attributes: alignment_metadata, provenance, schema_version

    HDF5 is preferred for training at scale because:
    - Random access without loading the full file
    - Compression (gzip/lzf) reduces storage
    - Compatible with PyTorch Dataset / TensorFlow Dataset loaders
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        raise NotImplementedError(
            "HDF5 export is not yet implemented.  Use JSONLExporter for now.\n"
            "See the docstring in this class for the planned implementation."
        )

    def export_episode(self, episode: InterleavedEpisode) -> None:
        raise NotImplementedError

    def write_metadata(self, stats: Dict[str, Any]) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_exporter(
    fmt: ExportFormat,
    output_dir: Path,
    **kwargs: Any,
) -> BaseExporter:
    """Instantiate the correct exporter for the given format."""
    if fmt == ExportFormat.JSONL:
        return JSONLExporter(output_dir, **kwargs)
    elif fmt == ExportFormat.HDF5:
        return HDF5Exporter(output_dir)
    elif fmt == ExportFormat.RLDS:
        from .rlds_export import RLDSExporter
        return RLDSExporter(output_dir, variants=kwargs.get("variants"))
    else:
        raise ValueError(f"Unknown export format: {fmt!r}")
