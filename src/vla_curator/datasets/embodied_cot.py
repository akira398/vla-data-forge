"""
Embodied-CoT dataset reader.

Loads ``Embodied-CoT/embodied_features_bridge`` (embodied_features_bridge.json).

Actual JSON structure
---------------------
{
  "/nfs/.../numpy_256/bridge_data_v2/env/task/ep/split/out.npy": {  ← file_path (join key)
    "43": {                                                            ← episode_id (string int)
      "metadata": {
        "episode_id":           "43",
        "file_path":            "/nfs/.../out.npy",
        "n_steps":              47,
        "language_instruction": "pick up the block",
        "caption":              "scene description"
      },
      "features": {
        "state_3d":          [[x,y,z], ...],       per-step end-effector 3D coords
        "move_primitive":    ["move forward", ...], per-step motion label
        "gripper_positions": [[x,y], ...]           per-step gripper pixel coords
      },
      "reasoning": {
        "0": {
          "task":          "...",   high-level task description
          "plan":          "...",   multi-step plan
          "subtask":       "...",   current subtask label
          "subtask_reason":"...",   why this subtask
          "move":          "...",   motion description
          "move_reason":   "..."    why this motion
        },
        "1": { ... },
        ...                         sparse — not every step is annotated
      }
    }
  },
  ...
}

Join key with Bridge v2
-----------------------
ECoT file_path (top-level key) corresponds to Bridge v2 source_file after
stripping the absolute /nfs/.../numpy_256/ prefix.
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_reasoning(r: dict) -> ReasoningTrace:
    """Convert one reasoning dict (for a single step) into a ReasoningTrace."""
    return ReasoningTrace(
        task_reasoning=r.get("task") or "",
        subtask_reasoning=r.get("subtask_reason") or "",
        move_reasoning=r.get("move_reason") or "",
        gripper_reasoning="",   # not present in this dataset
        attribute_reasoning="", # not present in this dataset
        spatial_reasoning="",   # not present in this dataset
        extra={
            "plan":    r.get("plan") or "",
            "subtask": r.get("subtask") or "",
            "move":    r.get("move") or "",
        },
    )


def _parse_entry(file_path: str, ep_id_str: str, entry: dict) -> ECoTEpisode:
    """
    Parse one entry from the JSON into an ECoTEpisode.

    Parameters
    ----------
    file_path : str
        Top-level JSON key — the .npy file path (from Bridge v2
        episode_metadata/file_path).
    ep_id_str : str
        Second-level JSON key — the episode ID string (e.g. "43",
        from Bridge v2 episode_metadata/episode_id).
    entry : dict
        The entry dict containing "metadata", "features", "reasoning".
    """
    metadata = entry.get("metadata", {})
    features = entry.get("features", {})
    reasoning_dict = entry.get("reasoning", {})

    n_steps = metadata.get("n_steps") or len(reasoning_dict) or 1
    instruction = metadata.get("language_instruction") or ""
    caption = metadata.get("caption") or ""

    move_primitives = features.get("move_primitive") or []
    gripper_positions = features.get("gripper_positions") or []
    state_3d = features.get("state_3d") or []

    steps: List[ECoTStep] = []
    for i in range(n_steps):
        # Reasoning — only present for annotated steps
        raw_r = reasoning_dict.get(str(i))
        reasoning = _parse_reasoning(raw_r) if raw_r else None

        # State: 3D end-effector position for this step
        state = None
        if i < len(state_3d):
            state = np.array(state_3d[i], dtype=np.float32)

        steps.append(ECoTStep(
            step_index=i,
            observation=ECoTObservation(step_index=i, image=None),
            action=np.zeros(7, dtype=np.float32),  # actions come from Bridge v2
            reasoning=reasoning,
            is_first=(i == 0),
            is_last=(i == n_steps - 1),
        ))

    # Composite key matches the original ECoT format:
    #   file_path + "_" + episode_id  (see MichalZawalski/embodied-CoT dataset.py)
    composite_key = f"{file_path}_{ep_id_str}"

    return ECoTEpisode(
        episode_id=composite_key,       # composite join key with Bridge v2
        language_instruction=instruction,
        steps=steps,
        metadata={
            "ecot_episode_id":  ep_id_str,
            "file_path":        file_path,
            "n_steps":          n_steps,
            "caption":          caption,
            "move_primitives":  move_primitives,
            "gripper_positions": gripper_positions,
        },
        source_dataset="embodied_features_bridge",
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class ECoTDatasetReader(DatasetReader[ECoTEpisode]):
    """
    Reader for the Embodied-CoT / embodied_features_bridge dataset.

    Usage
    -----
    cfg = ECoTDatasetConfig(
        local_path=Path("/datasets/embodied_features_bridge"),
        max_episodes=100,
    )
    for ep in ECoTDatasetReader(cfg):
        print(ep.episode_id)           # Bridge v2 file path
        print(ep.language_instruction)
        annotated = [s for s in ep.steps if s.reasoning is not None]
        print(f"{len(annotated)}/{len(ep.steps)} steps annotated")
    """

    dataset_name = "embodied_features_bridge"

    def __init__(self, config: ECoTDatasetConfig) -> None:
        self.config = config

    def __iter__(self) -> Iterator[ECoTEpisode]:
        if self.config.local_path is not None:
            yield from self._iter_local()
        else:
            yield from self._iter_hf()

    # ------------------------------------------------------------------
    # Local JSON loading
    # ------------------------------------------------------------------

    def _find_json_file(self) -> Path:
        path = self.config.local_path
        if path.is_file() and path.suffix == ".json":
            return path
        candidates = sorted(
            f for f in path.rglob("*.json")
            if not f.name.startswith(".")
            and f.name not in ("dataset_info.json", "dataset_dict.json")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .json files found under {path}\n"
                f"Contents: {[x.name for x in path.iterdir()]}"
            )
        return candidates[0]

    def _iter_local(self) -> Iterator[ECoTEpisode]:
        json_file = self._find_json_file()
        size_mb = json_file.stat().st_size / 1e6
        logger.info("Loading %s (%.0f MB)…", json_file.name, size_mb)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Loaded %d file-path entries.", len(data))

        limit = self.config.max_episodes
        count = 0

        for file_path, episodes in data.items():
            if not isinstance(episodes, dict):
                logger.warning("Unexpected value type for %s: %s", file_path, type(episodes))
                continue

            for ep_id_str, entry in episodes.items():
                if limit is not None and count >= limit:
                    return
                if not isinstance(entry, dict):
                    continue

                episode = _parse_entry(file_path, ep_id_str, entry)

                if self.config.require_reasoning and not episode.has_any_reasoning():
                    continue

                yield episode
                count += 1

    # ------------------------------------------------------------------
    # HF Hub streaming (fallback)
    # ------------------------------------------------------------------

    def _iter_hf(self) -> Iterator[ECoTEpisode]:
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
            episode = _parse_entry(str(idx), str(idx), row)
            if self.config.require_reasoning and not episode.has_any_reasoning():
                continue
            yield episode
            count += 1

    # ------------------------------------------------------------------
    # DatasetReader interface
    # ------------------------------------------------------------------

    def load_episode(self, episode_id: str) -> Optional[ECoTEpisode]:
        for ep in self:
            if ep.episode_id == episode_id:
                return ep
        return None

    def episode_ids(self) -> List[str]:
        return [ep.episode_id for ep in self]

    def __len__(self) -> int:
        if self.config.max_episodes is not None:
            return self.config.max_episodes
        raise TypeError(
            "Cannot determine length without max_episodes set. "
            "Set max_episodes in ECoTDatasetConfig or iterate directly."
        )

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "hf_repo":    self.config.hf_repo,
            "local_path": str(self.config.local_path) if self.config.local_path else None,
            "split":      self.config.split,
        })
        return base
