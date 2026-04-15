"""
Embodied-CoT dataset reader.

Loads the ``Embodied-CoT/embodied_features_bridge`` HuggingFace dataset and
converts it to ``ECoTEpisode`` objects.

Dataset assumptions
-------------------
The HF dataset is RLDS-style: each row in the Arrow table represents one
*episode*, and each episode has a ``steps`` column that is a sequence of dicts.

Expected step-level columns (best-effort detection, see ``COLUMN_MAP``):
  observation/image_0             bytes or uint8 array  (H, W, 3)
  action                          float32 (7,)
  language_instruction            str
  reasoning/task_reasoning        str   (may be missing for raw episodes)
  reasoning/subtask_reasoning     str
  reasoning/move_reasoning        str
  reasoning/gripper_reasoning     str
  reasoning/attribute_reasoning   str
  reasoning/spatial_reasoning     str
  is_first / is_last              bool

If the real column structure differs, update ``COLUMN_MAP`` below — the rest
of the reader is column-name-agnostic.

HuggingFace datasets uses Apache Arrow under the hood and may return images as:
  - numpy uint8 arrays          (most common)
  - PIL Image objects           (when decode=True and PIL installed)
  - bytes / encoded JPEG/PNG    (when decode=False)
We handle all three cases in ``_decode_image``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from ..config import ECoTDatasetConfig
from ..schemas.embodied_cot import (
    ECoTEpisode,
    ECoTObservation,
    ECoTStep,
    ReasoningTrace,
)
from .base import DatasetReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name mapping
# ---------------------------------------------------------------------------

# Maps internal field names → possible HF column paths (checked in order).
# Paths use "/" to represent nested dicts: "observation/image_0" means
# step["observation"]["image_0"].
COLUMN_MAP: Dict[str, List[str]] = {
    "image":               ["observation/image_0", "image_0", "image"],
    "action":              ["action"],
    "instruction":         ["language_instruction", "instruction", "task"],
    "task_reasoning":      ["reasoning/task_reasoning", "task_reasoning"],
    "subtask_reasoning":   ["reasoning/subtask_reasoning", "subtask_reasoning"],
    "move_reasoning":      ["reasoning/move_reasoning", "move_reasoning"],
    "gripper_reasoning":   ["reasoning/gripper_reasoning", "gripper_reasoning"],
    "attribute_reasoning": ["reasoning/attribute_reasoning", "attribute_reasoning"],
    "spatial_reasoning":   ["reasoning/spatial_reasoning", "spatial_reasoning"],
    "is_first":            ["is_first"],
    "is_last":             ["is_last"],
}


def _resolve(step: Dict[str, Any], candidates: List[str]) -> Any:
    """
    Try each candidate path against a step dict, returning the first match.
    Supports nested paths with "/" separator.  Returns None if nothing matches.
    """
    for path in candidates:
        parts = path.split("/")
        val = step
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break
        if val is not None:
            return val
    return None


def _decode_image(raw: Any) -> Optional[np.ndarray]:
    """Convert HF image representations to uint8 numpy arrays."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw.astype(np.uint8)
    # PIL Image
    try:
        from PIL import Image as PILImage
        if isinstance(raw, PILImage.Image):
            return np.array(raw.convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    # Bytes / encoded image
    if isinstance(raw, (bytes, bytearray)):
        import io
        from PIL import Image as PILImage
        return np.array(PILImage.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    # Dict with "bytes" key (HF datasets Image feature)
    if isinstance(raw, dict) and "bytes" in raw and raw["bytes"]:
        import io
        from PIL import Image as PILImage
        return np.array(
            PILImage.open(io.BytesIO(raw["bytes"])).convert("RGB"), dtype=np.uint8
        )
    logger.debug("Could not decode image of type %s", type(raw))
    return None


def _parse_step(
    step: Dict[str, Any],
    step_index: int,
    image_size: Optional[tuple[int, int]],
) -> ECoTStep:
    """Parse a single HF step dict into an ECoTStep."""
    # Image
    raw_image = _resolve(step, COLUMN_MAP["image"])
    image = _decode_image(raw_image)
    if image is not None and image_size is not None:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        pil = pil.resize((image_size[1], image_size[0]), PILImage.BILINEAR)
        image = np.array(pil)

    obs = ECoTObservation(step_index=step_index, image=image)

    # Action
    raw_action = _resolve(step, COLUMN_MAP["action"])
    if raw_action is not None:
        action = np.asarray(raw_action, dtype=np.float32).flatten()
    else:
        action = np.zeros(7, dtype=np.float32)
        logger.warning("Step %d has no action — defaulting to zeros.", step_index)

    # Reasoning
    task_r      = _resolve(step, COLUMN_MAP["task_reasoning"])
    subtask_r   = _resolve(step, COLUMN_MAP["subtask_reasoning"])
    move_r      = _resolve(step, COLUMN_MAP["move_reasoning"])
    gripper_r   = _resolve(step, COLUMN_MAP["gripper_reasoning"])
    attribute_r = _resolve(step, COLUMN_MAP["attribute_reasoning"])
    spatial_r   = _resolve(step, COLUMN_MAP["spatial_reasoning"])

    reasoning: Optional[ReasoningTrace] = None
    if any([task_r, subtask_r, move_r, gripper_r, attribute_r, spatial_r]):
        reasoning = ReasoningTrace(
            task_reasoning=task_r,
            subtask_reasoning=subtask_r,
            move_reasoning=move_r,
            gripper_reasoning=gripper_r,
            attribute_reasoning=attribute_r,
            spatial_reasoning=spatial_r,
        )

    is_first = bool(_resolve(step, COLUMN_MAP["is_first"]) or (step_index == 0))
    is_last  = bool(_resolve(step, COLUMN_MAP["is_last"]) or False)

    return ECoTStep(
        step_index=step_index,
        observation=obs,
        action=action,
        reasoning=reasoning,
        is_first=is_first,
        is_last=is_last,
    )


def _parse_episode(
    row: Dict[str, Any],
    episode_index: int,
    image_size: Optional[tuple[int, int]],
) -> ECoTEpisode:
    """Parse a HF dataset row into an ECoTEpisode."""
    # Episode ID
    episode_id = str(
        row.get("episode_id")
        or row.get("episode_metadata", {}).get("file_path", f"ep_{episode_index:06d}")
        or f"ep_{episode_index:06d}"
    )

    # Steps — handle both flat and nested formats
    raw_steps = row.get("steps", [])
    if not raw_steps:
        # Some versions store steps as top-level lists
        raw_steps = row.get("trajectory", [])

    steps: List[ECoTStep] = []

    # Language instruction — try to get from first step or episode metadata
    instruction = (
        row.get("language_instruction")
        or row.get("task", "")
    )

    for i, raw_step in enumerate(raw_steps):
        step = _parse_step(raw_step, step_index=i, image_size=image_size)
        # Instruction from step if not at episode level
        if not instruction:
            instruction = str(_resolve(raw_step, COLUMN_MAP["instruction"]) or "")
        steps.append(step)

    if steps:
        steps[0].is_first = True
        steps[-1].is_last = True

    metadata: Dict[str, Any] = {}
    if "episode_metadata" in row:
        metadata["episode_metadata"] = row["episode_metadata"]

    return ECoTEpisode(
        episode_id=episode_id,
        language_instruction=instruction,
        steps=steps,
        metadata=metadata,
        source_dataset="embodied_features_bridge",
    )


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------


class ECoTDatasetReader(DatasetReader[ECoTEpisode]):
    """
    Reader for the Embodied-CoT / embodied_features_bridge HF dataset.

    Parameters
    ----------
    config : ECoTDatasetConfig
        Dataset configuration (split, max_episodes, image_size, etc.)

    Usage
    -----
    from vla_curator.datasets import ECoTDatasetReader
    from vla_curator.config import ECoTDatasetConfig

    cfg = ECoTDatasetConfig(split="train", max_episodes=10)
    reader = ECoTDatasetReader(cfg)
    for episode in reader:
        print(episode)
    """

    dataset_name = "embodied_features_bridge"

    def __init__(self, config: ECoTDatasetConfig) -> None:
        self.config = config
        self._hf_dataset: Any = None   # lazy-loaded
        self._ids: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self) -> Any:
        """Load the HuggingFace dataset (downloads if not cached)."""
        try:
            import datasets as hf_datasets
        except ImportError as e:
            raise ImportError(
                "The 'datasets' package is required to load ECoT data. "
                "Install it with: pip install datasets"
            ) from e

        logger.info(
            "Loading HF dataset %s (split=%s)…",
            self.config.hf_repo,
            self.config.split,
        )
        ds = hf_datasets.load_dataset(
            self.config.hf_repo,
            split=self.config.split,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )
        logger.info("Loaded %d rows from %s.", len(ds), self.config.hf_repo)
        return ds

    @property
    def hf_dataset(self) -> Any:
        if self._hf_dataset is None:
            self._hf_dataset = self._load_hf_dataset()
        return self._hf_dataset

    # ------------------------------------------------------------------
    # DatasetReader interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ECoTEpisode]:
        ds = self.hf_dataset
        limit = self.config.max_episodes

        indices = list(range(len(ds)))
        if self.config.shuffle:
            import random
            rng = random.Random(self.config.shuffle_seed)
            rng.shuffle(indices)

        count = 0
        for idx in indices:
            if limit is not None and count >= limit:
                break
            row = ds[idx]
            episode = _parse_episode(row, idx, self.config.image_size)

            if self.config.require_reasoning and not episode.has_any_reasoning():
                continue

            yield episode
            count += 1

    def load_episode(self, episode_id: str) -> Optional[ECoTEpisode]:
        """
        Load a single episode by ID.

        This performs a linear scan since HF Arrow datasets do not have
        random access by custom ID.  For large datasets, build an
        id→index mapping once and cache it.
        """
        for i, row in enumerate(self.hf_dataset):
            ep = _parse_episode(row, i, self.config.image_size)
            if ep.episode_id == episode_id:
                return ep
        return None

    def episode_ids(self) -> List[str]:
        if self._ids is None:
            ids = []
            for i, row in enumerate(self.hf_dataset):
                ep_id = str(
                    row.get("episode_id")
                    or row.get("episode_metadata", {}).get("file_path", f"ep_{i:06d}")
                    or f"ep_{i:06d}"
                )
                ids.append(ep_id)
            self._ids = ids
        return self._ids

    def __len__(self) -> int:
        n = len(self.hf_dataset)
        if self.config.max_episodes is not None:
            return min(n, self.config.max_episodes)
        return n

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update(
            {
                "hf_repo": self.config.hf_repo,
                "split": self.config.split,
                "image_size": self.config.image_size,
            }
        )
        return base
