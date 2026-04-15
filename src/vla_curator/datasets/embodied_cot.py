"""
Embodied-CoT dataset reader.

Loads ``Embodied-CoT/embodied_features_bridge``.

Dataset structure (local download)
------------------------------------
A single JSON file ``embodied_features_bridge.json`` (~1.4 GB).
Top-level is a dict keyed by episode file paths:

    {
      "/path/to/bridge_data/.../out.npy": {
          "task_reasoning":      "...",
          "subtask_reasoning":   "...",
          "move_reasoning":      "...",
          "gripper_reasoning":   "...",
          "attribute_reasoning": "...",
          "spatial_reasoning":   "..."
      },
      ...
    }

Each key is the episode ID (matches Bridge v2 source paths).
Images are NOT stored here — they live in the Bridge v2 .npy / TFRecord files.
The ECoT dataset provides only the reasoning annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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

# Reasoning field names expected in the JSON values
_REASONING_FIELDS = [
    "task_reasoning",
    "subtask_reasoning",
    "move_reasoning",
    "gripper_reasoning",
    "attribute_reasoning",
    "spatial_reasoning",
]


def _parse_step(item: dict, index: int, n_steps: int) -> ECoTStep:
    """
    Parse one step dict from the JSON list.

    Actual field names in embodied_features_bridge.json:
      frame_index          int
      language_instruction str
      task                 str   high-level task description
      plan                 str   multi-step plan
      subtask_reason       str   why this subtask
      subtask              str   current subtask label
      move_reason          str   why this motion
      move                 str   motion description
      gripper              str   future gripper positions (5-step lookahead)
      bboxes               str   visible objects + bounding boxes
      action               list  7-DoF [Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]
    """
    step_index = item.get("frame_index", index)

    raw_action = item.get("action")
    if raw_action is not None:
        action = np.asarray(raw_action, dtype=np.float32).flatten()
    else:
        action = np.zeros(7, dtype=np.float32)

    reasoning = ReasoningTrace(
        task_reasoning=item.get("task") or "",
        subtask_reasoning=item.get("subtask_reason") or "",
        move_reasoning=item.get("move_reason") or "",
        gripper_reasoning=item.get("gripper") or "",
        attribute_reasoning=item.get("bboxes") or "",
        spatial_reasoning=item.get("plan") or "",
        extra={
            "subtask": item.get("subtask") or "",
            "move":    item.get("move") or "",
        },
    )

    return ECoTStep(
        step_index=step_index,
        observation=ECoTObservation(step_index=step_index, image=None),
        action=action,
        reasoning=reasoning,
        is_first=(index == 0),
        is_last=(index == n_steps - 1),
    )


def _parse_entry(episode_id: str, value: Any, ep_index: int) -> ECoTEpisode:
    """Parse one key-value pair from embodied_features_bridge.json."""
    if isinstance(value, list):
        # Standard format: list of per-step dicts
        instruction = ""
        steps = []
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            if not instruction:
                instruction = item.get("language_instruction") or ""
            steps.append(_parse_step(item, i, len(value)))

    elif isinstance(value, dict):
        # Episode-level dict (less common)
        instruction = value.get("language_instruction") or value.get("task") or ""
        steps = [_parse_step(value, 0, 1)]

    else:
        instruction = ""
        steps = []

    if not steps:
        steps = [ECoTStep(
            step_index=0,
            observation=ECoTObservation(step_index=0, image=None),
            action=np.zeros(7, dtype=np.float32),
            reasoning=None,
            is_first=True,
            is_last=True,
        )]

    return ECoTEpisode(
        episode_id=episode_id,
        language_instruction=instruction,
        steps=steps,
        metadata={"source_path": episode_id},
        source_dataset="embodied_features_bridge",
    )


class ECoTDatasetReader(DatasetReader[ECoTEpisode]):
    """
    Reader for the Embodied-CoT / embodied_features_bridge dataset.

    Local usage (recommended)
    -------------------------
    Download once:
        huggingface-cli download Embodied-CoT/embodied_features_bridge \\
            --repo-type dataset --local-dir /datasets/embodied_features_bridge

    Then:
        cfg = ECoTDatasetConfig(local_path=Path("/datasets/embodied_features_bridge"))
        reader = ECoTDatasetReader(cfg)
        for ep in reader:
            print(ep.episode_id, ep.language_instruction)
    """

    dataset_name = "embodied_features_bridge"

    def __init__(self, config: ECoTDatasetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ECoTEpisode]:
        if self.config.local_path is not None:
            yield from self._iter_local()
        else:
            yield from self._iter_hf()

    def _iter_local(self) -> Iterator[ECoTEpisode]:
        """Load from local JSON file(s)."""
        path = self.config.local_path
        if not path.exists():
            raise FileNotFoundError(f"local_path does not exist: {path}")

        # Find JSON files, skip metadata
        json_files = sorted(
            f for f in path.rglob("*.json")
            if not f.name.startswith(".")
            and f.name not in ("dataset_info.json", "dataset_dict.json")
        )
        if not path.is_dir():
            # local_path might point directly to the json file
            json_files = [path] if path.suffix == ".json" else []

        if not json_files:
            raise FileNotFoundError(
                f"No .json files found under {path}\n"
                f"Contents: {[f.name for f in path.iterdir()]}"
            )

        limit = self.config.max_episodes
        count = 0

        for json_file in json_files:
            if limit is not None and count >= limit:
                break

            size_mb = json_file.stat().st_size / 1e6
            logger.info("Loading %s (%.0f MB) — this may take a moment…", json_file.name, size_mb)

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info("Loaded %d entries from %s", len(data), json_file.name)

            # Handle both dict {path: {...}} and list [{...}, ...] top-level formats
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = ((str(i), row) for i, row in enumerate(data))
            else:
                raise ValueError(f"Unexpected JSON top-level type: {type(data)}")

            for ep_index, (key, value) in enumerate(items):
                if limit is not None and count >= limit:
                    break
                episode = _parse_entry(key, value, ep_index)
                if self.config.require_reasoning and not episode.has_any_reasoning():
                    continue
                yield episode
                count += 1

    def _iter_hf(self) -> Iterator[ECoTEpisode]:
        """Stream from HuggingFace Hub."""
        try:
            import datasets as hf_datasets
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        logger.info("Streaming %s from HF Hub…", self.config.hf_repo)
        ds = hf_datasets.load_dataset(
            self.config.hf_repo,
            split=self.config.split,
            streaming=True,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )

        limit = self.config.max_episodes
        count = 0
        for idx, row in enumerate(ds):
            if limit is not None and count >= limit:
                break
            # HF Hub version may have a different structure — use key/value if dict-like
            if "episode_id" in row or "steps" not in row:
                ep_id = row.get("episode_id") or str(idx)
                episode = _parse_entry(ep_id, row, idx)
            else:
                # Legacy RLDS-style rows with "steps" column
                episode = _parse_entry(str(idx), row, idx)
            if self.config.require_reasoning and not episode.has_any_reasoning():
                continue
            yield episode
            count += 1

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def load_episode(self, episode_id: str):
        for ep in self:
            if ep.episode_id == episode_id:
                return ep
        return None

    def episode_ids(self):
        return [ep.episode_id for ep in self]

    def __len__(self) -> int:
        if self.config.max_episodes is not None:
            return self.config.max_episodes
        raise TypeError(
            "Cannot determine length without max_episodes. "
            "Set max_episodes or iterate directly."
        )

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "hf_repo": self.config.hf_repo,
            "local_path": str(self.config.local_path) if self.config.local_path else None,
            "split": self.config.split,
        })
        return base
