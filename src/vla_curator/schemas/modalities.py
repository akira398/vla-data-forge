"""
Optional sensing modality schemas and the modality registry.

Architecture: pluggable modalities
-----------------------------------
The curation pipeline supports a fixed core (RGB images + robot actions) and an
open extension set of *optional modalities* (depth maps, scene graphs, …).

Each modality is represented as a dataclass with:
  - ``valid: bool``    — whether this instance is populated
  - ``data``           — the actual payload (or None if not valid)
  - ``to_dict()``      — serialisation

The ``ModalityRegistry`` maps modality names to:
  - The schema class
  - An optional extractor class (the thing that generates the modality)

How to add a new modality (e.g. optical flow)
----------------------------------------------
1. Define a ``OpticalFlow`` dataclass in this file (follow DepthMap as a template).
2. Define an ``OpticalFlowExtractor(ModalityExtractor)`` in a new file under
   ``src/vla_curator/modalities/optical_flow.py``.
3. Register it: ``ModalityRegistry.register("optical_flow", OpticalFlow, OpticalFlowExtractor)``
4. Add an ``optical_flow: OpticalFlow`` field to ``EnrichedObservation`` in
   ``schemas/interleaved.py``.

The extractor is applied in the per-frame enrichment pipeline
(``curation/enrichment.py`` — reserved for future implementation).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np


# ---------------------------------------------------------------------------
# Depth map
# ---------------------------------------------------------------------------


@dataclass
class DepthMap:
    """
    Per-frame depth map placeholder.

    ``data`` is a (H, W) float32 array in metres (positive = distance from camera).
    ``sensor_type`` describes the source (e.g. 'monocular_est', 'lidar', 'stereo').

    To add depth support:
    1. Implement a ``DepthExtractor`` that populates this dataclass.
    2. Run it in the per-frame enrichment step.
    3. The training data loader will see ``depth_map.valid == True`` and can
       include it as an additional input channel.
    """

    valid: bool = False
    data: Optional[np.ndarray] = None          # (H, W) float32 metres
    data_path: Optional[str] = None            # Path to saved .npy file
    sensor_type: str = "unknown"
    scale_factor: float = 1.0                  # Multiply data to get metres
    metadata: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Optional[np.ndarray]:
        if self.data is not None:
            return self.data
        if self.data_path is not None:
            return np.load(self.data_path)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "data_path": self.data_path,
            "sensor_type": self.sensor_type,
            "scale_factor": self.scale_factor,
        }


# ---------------------------------------------------------------------------
# Scene graph
# ---------------------------------------------------------------------------


@dataclass
class SceneGraphNode:
    """One object in a scene graph."""

    node_id: str = ""
    label: str = ""                            # Object class/name
    bbox: Optional[List[float]] = None         # [x1, y1, x2, y2] normalised
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    """Free-form object attributes: color, material, state, etc."""


@dataclass
class SceneGraphEdge:
    """Directed relationship between two scene graph nodes."""

    source_id: str = ""
    target_id: str = ""
    relation: str = ""                         # e.g. "on_top_of", "grasping"
    confidence: float = 1.0


@dataclass
class SceneGraph:
    """
    Per-frame scene graph placeholder.

    A scene graph captures object identities, their attributes, and the
    spatial / functional relationships between them.  This is richer than
    raw text descriptions and enables symbolic reasoning in downstream models.

    To add scene graph support:
    1. Use a grounded VLM (e.g. GPT-4V with bounding-box prompts) or an
       open-vocabulary detector to populate ``nodes``.
    2. Run a relationship extraction model to populate ``edges``.
    3. Implement a ``SceneGraphExtractor`` and register it.
    """

    valid: bool = False
    nodes: List[SceneGraphNode] = field(default_factory=list)
    edges: List[SceneGraphEdge] = field(default_factory=list)
    source_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def node_by_id(self, node_id: str) -> Optional[SceneGraphNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "bbox": n.bbox,
                    "confidence": n.confidence,
                    "attributes": n.attributes,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
            "source_model": self.source_model,
        }


# ---------------------------------------------------------------------------
# Abstract extractor
# ---------------------------------------------------------------------------


class ModalityExtractor(abc.ABC):
    """
    Abstract base class for anything that computes an optional modality from
    a raw observation (image, state, etc.).

    Implement ``extract`` and register the extractor with ``ModalityRegistry``.
    The per-frame enrichment pipeline calls ``extract`` on each step's
    ``EnrichedObservation``.
    """

    @abc.abstractmethod
    def extract(self, image: np.ndarray, **kwargs) -> Any:
        """
        Args:
            image:   (H, W, 3) uint8 RGB image.
            **kwargs: Additional context (state, previous observation, etc.)

        Returns:
            A modality dataclass instance (e.g. DepthMap, SceneGraph).
        """
        ...

    @property
    @abc.abstractmethod
    def modality_name(self) -> str:
        """Unique name identifying this modality, e.g. 'depth_map'."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ModalitySpec:
    """Metadata about a registered modality."""

    name: str
    schema_class: Type
    extractor_class: Optional[Type[ModalityExtractor]] = None
    description: str = ""


class ModalityRegistry:
    """
    Central registry mapping modality names to their schema and extractor.

    Usage
    -----
    # Register a modality (done once at startup or in a plugin)
    ModalityRegistry.register("depth_map", DepthMap, MyDepthExtractor,
                               description="Monocular depth estimation")

    # Check availability
    if ModalityRegistry.has("depth_map"):
        extractor = ModalityRegistry.get_extractor("depth_map")
        depth = extractor().extract(image)

    # List all registered modalities
    for name, spec in ModalityRegistry.all().items():
        print(name, spec.description)
    """

    _registry: Dict[str, ModalitySpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        schema_class: Type,
        extractor_class: Optional[Type[ModalityExtractor]] = None,
        description: str = "",
    ) -> None:
        cls._registry[name] = ModalitySpec(
            name=name,
            schema_class=schema_class,
            extractor_class=extractor_class,
            description=description,
        )

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def get_spec(cls, name: str) -> ModalitySpec:
        if name not in cls._registry:
            raise KeyError(f"Modality {name!r} not registered. "
                           f"Available: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def get_extractor(cls, name: str) -> Optional[Type[ModalityExtractor]]:
        return cls.get_spec(name).extractor_class

    @classmethod
    def all(cls) -> Dict[str, ModalitySpec]:
        return dict(cls._registry)


# ---------------------------------------------------------------------------
# Pre-register built-in modalities (no extractors yet — placeholder)
# ---------------------------------------------------------------------------

ModalityRegistry.register(
    "depth_map",
    DepthMap,
    extractor_class=None,       # TODO: register MonocularDepthExtractor once implemented
    description=(
        "Per-frame depth map (H×W float32, metres). "
        "Planned: monocular estimation via Depth-Anything or ZoeDepth."
    ),
)

ModalityRegistry.register(
    "scene_graph",
    SceneGraph,
    extractor_class=None,       # TODO: register SceneGraphExtractor once implemented
    description=(
        "Per-frame scene graph with object nodes and relational edges. "
        "Planned: grounded detection + VLM-based relationship extraction."
    ),
)
