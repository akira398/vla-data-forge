"""Schema package — canonical data structures for all dataset types."""

from .base import RobotAction, NumpyArrayMixin
from .embodied_cot import ReasoningTrace, ECoTObservation, ECoTStep, ECoTEpisode
from .bridge_v2 import BridgeObservation, BridgeStep, BridgeEpisode
from .interleaved import (
    EnrichedObservation,
    AlignedStep,
    AlignmentMetadata,
    DataProvenance,
    InterleavedEpisode,
    AlignmentStrategy,
)
from .modalities import DepthMap, SceneGraph, ModalityRegistry

__all__ = [
    "RobotAction",
    "NumpyArrayMixin",
    "ReasoningTrace",
    "ECoTObservation",
    "ECoTStep",
    "ECoTEpisode",
    "BridgeObservation",
    "BridgeStep",
    "BridgeEpisode",
    "EnrichedObservation",
    "AlignedStep",
    "AlignmentMetadata",
    "DataProvenance",
    "InterleavedEpisode",
    "AlignmentStrategy",
    "DepthMap",
    "SceneGraph",
    "ModalityRegistry",
]
