"""
Schemas for Bridge v2 data.

Bridge v2 (Walke et al. 2023) is a large-scale robot manipulation dataset
collected on a BerkeleyUR5 robot.  It is distributed in RLDS format via
tensorflow_datasets (tfds name: "bridge_dataset") and also as raw HDF5 files.

Standard Bridge v2 step structure (RLDS):
  observation/image_0         uint8 (H, W, 3)   — wrist camera or front camera
  observation/image_1         uint8 (H, W, 3)   — secondary camera (may be zeros)
  observation/state           float32 (7,)       — robot proprioceptive state
  action                      float32 (7,)       — delta EEF + gripper
  language_instruction        str
  is_first / is_last / is_terminal   bool
  reward                      float32
  discount                    float32

We store both cameras because Bridge v2 includes two views and downstream VLA
models may consume either.  ``image_0`` is the primary view (wrist or front,
dataset-dependent) and is the one fed to visualisation by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base import NumpyArrayMixin, RobotAction


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class BridgeObservation(NumpyArrayMixin):
    """
    Multi-camera observation from Bridge v2.

    Both cameras are optional at the schema level to accommodate episodes where
    only one view was collected.  ``state`` is the raw proprioceptive vector;
    it may or may not be used by downstream models.
    """

    step_index: int = 0
    image_0: Optional[np.ndarray] = None       # (H, W, 3) uint8 — primary view
    image_1: Optional[np.ndarray] = None       # (H, W, 3) uint8 — secondary view
    image_0_path: Optional[str] = None
    image_1_path: Optional[str] = None
    state: Optional[np.ndarray] = None         # (7,) float32 proprioception

    def load_image_0(self) -> Optional[np.ndarray]:
        """Return primary image, loading from disk if needed."""
        if self.image_0 is not None:
            return self.image_0
        if self.image_0_path is not None:
            from PIL import Image as PILImage
            return np.array(PILImage.open(self.image_0_path).convert("RGB"))
        return None

    def load_image_1(self) -> Optional[np.ndarray]:
        if self.image_1 is not None:
            return self.image_1
        if self.image_1_path is not None:
            from PIL import Image as PILImage
            return np.array(PILImage.open(self.image_1_path).convert("RGB"))
        return None

    def primary_image(self) -> Optional[np.ndarray]:
        """Alias for load_image_0 — the view used by default."""
        return self.load_image_0()

    def has_image(self) -> bool:
        return self.image_0 is not None or self.image_0_path is not None

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "image_0_path": self.image_0_path,
            "image_1_path": self.image_1_path,
            "state": self.array_to_list(self.state),
        }
        if include_images:
            d["image_0"] = self.array_to_list(self.image_0)
            d["image_1"] = self.array_to_list(self.image_1)
        return d


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


@dataclass
class BridgeStep(NumpyArrayMixin):
    """One timestep in a Bridge v2 trajectory."""

    step_index: int = 0
    observation: BridgeObservation = field(default_factory=BridgeObservation)
    action: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    """7-DoF end-effector action [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]."""
    language_instruction: str = ""
    """
    Instruction is stored per-step because RLDS encodes it that way.
    In practice it is identical across all steps of an episode.
    """
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    reward: float = 0.0
    discount: float = 1.0

    def robot_action(self) -> RobotAction:
        return RobotAction.from_numpy(self.action)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.array_to_list(self.action),
            "language_instruction": self.language_instruction,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
            "discount": self.discount,
            "observation": self.observation.to_dict(include_images=include_images),
        }


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class BridgeEpisode:
    """
    A complete Bridge v2 trajectory.

    ``source_file`` records the original HDF5/TFDS shard path, which is the
    canonical identifier used to join with ECoT annotations.
    """

    episode_id: str = ""
    language_instruction: str = ""
    steps: List[BridgeStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    """Path in the Bridge v2 archive — used as part of join key with ECoT."""
    episode_num: Optional[int] = None
    """Integer episode ID from episode_metadata/episode_id in Bridge v2 TFDS.
    Together with source_file, forms the composite join key with ECoT."""

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, idx: int) -> BridgeStep:
        return self.steps[idx]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_primary_images(self) -> List[Optional[np.ndarray]]:
        return [step.observation.primary_image() for step in self.steps]

    def get_actions(self) -> np.ndarray:
        """Return (T, 7) action array."""
        return np.stack([step.action for step in self.steps], axis=0)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "language_instruction": self.language_instruction,
            "source_file": self.source_file,
            "metadata": self.metadata,
            "steps": [s.to_dict(include_images=include_images) for s in self.steps],
        }

    def __repr__(self) -> str:
        return (
            f"BridgeEpisode(id={self.episode_id!r}, "
            f"steps={len(self.steps)}, "
            f"instruction={self.language_instruction!r})"
        )
