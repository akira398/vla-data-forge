"""
Canonical interleaved schema — the training-ready output of the curation pipeline.

This is the single most important schema in the codebase.  Every upstream data
source (ECoT, Bridge v2, future datasets) eventually gets normalised into
``InterleavedEpisode``.  Downstream VLA training code should only need to
understand this schema.

Design decisions
----------------
1. ``EnrichedObservation`` is a superset of both ECoT and Bridge observations.
   It has slots for future modalities (depth, scene graph) that default to
   empty/invalid ``DepthMap`` / ``SceneGraph`` objects rather than None.
   This means training code can always access ``step.observation.depth_map``
   without a None-check — it just checks ``.valid`` instead.

2. ``AlignedStep`` carries an ``alignment_confidence`` float (0–1).
   When reasoning traces come from ECoT (which may only annotate key frames),
   steps that received a propagated trace (e.g. from nearest-key-frame) get a
   confidence < 1.0 so downstream models can weight or filter them.

3. ``AlignmentMetadata`` records the strategy used to fuse the two datasets.
   This is critical for reproducibility: given the metadata you can re-run the
   alignment and get the same result.

4. ``DataProvenance`` tracks *which* model (provider + model name) generated
   the reasoning traces, so dataset versions remain auditable.

5. ``schema_version`` on ``InterleavedEpisode`` allows future breaking changes
   to the format to be detected at load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .base import NumpyArrayMixin, RobotAction
from .embodied_cot import ReasoningTrace
from .modalities import DepthMap, SceneGraph


# ---------------------------------------------------------------------------
# Alignment strategy enum
# ---------------------------------------------------------------------------


class AlignmentStrategy(str, Enum):
    """
    How reasoning traces from ECoT are assigned to Bridge v2 steps.

    EXACT       Only steps that have an explicit trace carry reasoning.
                All other steps have reasoning=None.
                Best for: precision, avoiding hallucinated trace propagation.

    NEAREST     Each unannotated step inherits the trace from the nearest
                annotated step (by step index).
                Best for: dense training signal when ECoT only annotates
                key frames.

    BROADCAST   The single trace for the entire episode is copied to every step.
                Suitable only when ECoT provides one trace per episode.
    """

    EXACT = "exact"
    NEAREST = "nearest"
    BROADCAST = "broadcast"


# ---------------------------------------------------------------------------
# Enriched observation (the hub for all modalities)
# ---------------------------------------------------------------------------


@dataclass
class EnrichedObservation(NumpyArrayMixin):
    """
    A single observation that may carry multiple sensing modalities.

    Modality availability is signalled by the ``.valid`` flag on each
    placeholder object rather than by None-checking.  This makes
    iteration code cleaner:

        if step.observation.depth_map.valid:
            process_depth(step.observation.depth_map.data)

    Future modalities beyond depth and scene graph should be added here AND
    registered in ``ModalityRegistry`` (see schemas/modalities.py).
    """

    step_index: int = 0
    timestamp: Optional[float] = None         # Seconds from episode start

    # Primary visual modality
    image: Optional[np.ndarray] = None        # (H, W, 3) uint8
    image_path: Optional[str] = None

    # Secondary camera (from Bridge v2)
    image_secondary: Optional[np.ndarray] = None
    image_secondary_path: Optional[str] = None

    # Proprioception
    state: Optional[np.ndarray] = None        # (7,) float32

    # Future modalities — always present, check .valid before using
    depth_map: DepthMap = field(default_factory=DepthMap)
    scene_graph: SceneGraph = field(default_factory=SceneGraph)

    # Open-ended extension: any future modality not yet formalised
    extra_modalities: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self) -> Optional[np.ndarray]:
        if self.image is not None:
            return self.image
        if self.image_path is not None:
            from PIL import Image as PILImage
            return np.array(PILImage.open(self.image_path).convert("RGB"))
        return None

    def load_secondary_image(self) -> Optional[np.ndarray]:
        if self.image_secondary is not None:
            return self.image_secondary
        if self.image_secondary_path is not None:
            from PIL import Image as PILImage
            return np.array(PILImage.open(self.image_secondary_path).convert("RGB"))
        return None

    def active_modalities(self) -> List[str]:
        """Return names of modalities that are populated and valid."""
        active = []
        if self.load_image() is not None:
            active.append("image")
        if self.load_secondary_image() is not None:
            active.append("image_secondary")
        if self.state is not None:
            active.append("state")
        if self.depth_map.valid:
            active.append("depth_map")
        if self.scene_graph.valid:
            active.append("scene_graph")
        active.extend(self.extra_modalities.keys())
        return active

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "image_secondary_path": self.image_secondary_path,
            "state": self.array_to_list(self.state),
            "depth_map": self.depth_map.to_dict(),
            "scene_graph": self.scene_graph.to_dict(),
            "extra_modalities": {k: str(v) for k, v in self.extra_modalities.items()},
        }
        if include_images:
            d["image"] = self.array_to_list(self.image)
        return d


# ---------------------------------------------------------------------------
# Aligned step
# ---------------------------------------------------------------------------


@dataclass
class AlignedStep(NumpyArrayMixin):
    """
    One timestep in the merged interleaved dataset.

    Combines an ``EnrichedObservation`` from Bridge v2 with an optional
    ``ReasoningTrace`` from ECoT.  ``alignment_confidence`` records how
    reliable the association is.
    """

    step_index: int = 0
    observation: EnrichedObservation = field(default_factory=EnrichedObservation)
    action: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    reasoning: Optional[ReasoningTrace] = None
    is_first: bool = False
    is_last: bool = False
    source_dataset: str = ""
    alignment_confidence: float = 1.0
    """
    Confidence that this step's reasoning trace correctly corresponds to
    this observation.  1.0 = direct annotation; < 1.0 = propagated from a
    neighbouring annotated step.
    """

    def robot_action(self) -> RobotAction:
        return RobotAction.from_numpy(self.action)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.array_to_list(self.action),
            "reasoning": self.reasoning.to_dict() if self.reasoning else None,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "source_dataset": self.source_dataset,
            "alignment_confidence": self.alignment_confidence,
            "observation": self.observation.to_dict(include_images=include_images),
        }


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlignmentMetadata:
    """
    Records how the ECoT reasoning was fused with Bridge v2 observations.

    Every ``InterleavedEpisode`` carries one of these so the merge can be
    understood and reproduced.
    """

    strategy: str = AlignmentStrategy.NEAREST.value
    ecot_episode_id: str = ""
    bridge_episode_id: str = ""
    num_steps_ecot: int = 0
    num_steps_bridge: int = 0
    num_aligned_steps: int = 0
    num_annotated_steps: int = 0
    """Steps with a direct (non-propagated) reasoning trace."""
    reasoning_coverage: float = 0.0
    """Fraction of steps that carry *any* reasoning trace after alignment."""
    alignment_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ecot_episode_id": self.ecot_episode_id,
            "bridge_episode_id": self.bridge_episode_id,
            "num_steps_ecot": self.num_steps_ecot,
            "num_steps_bridge": self.num_steps_bridge,
            "num_aligned_steps": self.num_aligned_steps,
            "num_annotated_steps": self.num_annotated_steps,
            "reasoning_coverage": self.reasoning_coverage,
            "alignment_notes": self.alignment_notes,
        }


@dataclass
class DataProvenance:
    """
    Source tracking for a curated episode.

    Storing the model that generated the traces is essential: different model
    versions can produce qualitatively different reasoning, and training runs
    should know exactly what they consumed.
    """

    ecot_source: str = "embodied_features_bridge"
    bridge_source: str = "bridge_v2"
    generation_backend: Optional[str] = None
    """Provider used during trace generation, e.g. 'gemini', 'openai'."""
    generation_model: Optional[str] = None
    """Specific model, e.g. 'gemini-1.5-pro', 'gpt-4o'."""
    curation_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecot_source": self.ecot_source,
            "bridge_source": self.bridge_source,
            "generation_backend": self.generation_backend,
            "generation_model": self.generation_model,
            "curation_version": self.curation_version,
        }


# ---------------------------------------------------------------------------
# Top-level interleaved episode
# ---------------------------------------------------------------------------


@dataclass
class InterleavedEpisode:
    """
    The canonical training-ready episode.

    This is the final output of the curation pipeline.  It combines:
    - Observations + actions from Bridge v2 (high-quality, multi-camera)
    - Reasoning traces from ECoT (generated by a VLM)
    - Full alignment and provenance metadata

    The ``schema_version`` field allows downstream loaders to detect format
    changes.  Bump it when making breaking changes to this class.
    """

    episode_id: str = ""
    """Original Bridge v2 file_path (episode_metadata/file_path)."""
    episode_num: Optional[int] = None
    """Integer episode ID from Bridge v2 (episode_metadata/episode_id)."""
    task_description: str = ""
    steps: List[AlignedStep] = field(default_factory=list)
    alignment_metadata: AlignmentMetadata = field(default_factory=AlignmentMetadata)
    provenance: DataProvenance = field(default_factory=DataProvenance)
    schema_version: str = "1.0"

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, idx: int) -> AlignedStep:
        return self.steps[idx]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_actions(self) -> np.ndarray:
        return np.stack([s.action for s in self.steps], axis=0)

    def get_images(self) -> List[Optional[np.ndarray]]:
        return [s.observation.load_image() for s in self.steps]

    def get_reasoning_traces(self) -> List[Optional[ReasoningTrace]]:
        return [s.reasoning for s in self.steps]

    def has_reasoning(self) -> bool:
        return any(s.reasoning is not None for s in self.steps)

    def reasoning_coverage(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.reasoning is not None) / len(self.steps)

    def high_confidence_steps(self, threshold: float = 0.9) -> List[AlignedStep]:
        """Steps where alignment_confidence >= threshold."""
        return [s for s in self.steps if s.alignment_confidence >= threshold]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task_description": self.task_description,
            "alignment_metadata": self.alignment_metadata.to_dict(),
            "provenance": self.provenance.to_dict(),
            "steps": [s.to_dict(include_images=include_images) for s in self.steps],
        }

    def __repr__(self) -> str:
        return (
            f"InterleavedEpisode(id={self.episode_id!r}, "
            f"steps={len(self.steps)}, "
            f"reasoning={self.reasoning_coverage():.0%}, "
            f"strategy={self.alignment_metadata.strategy!r})"
        )
