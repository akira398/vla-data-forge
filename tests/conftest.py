"""
Pytest fixtures shared across test modules.

All fixtures use synthetic in-memory data — no network calls, no HF downloads,
no GPU required.  Tests should run in < 5 seconds on any laptop.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from vla_curator.schemas.bridge_v2 import BridgeEpisode, BridgeObservation, BridgeStep
from vla_curator.schemas.embodied_cot import (
    ECoTEpisode,
    ECoTObservation,
    ECoTStep,
    ReasoningTrace,
)
from vla_curator.schemas.interleaved import (
    AlignedStep,
    AlignmentMetadata,
    DataProvenance,
    EnrichedObservation,
    InterleavedEpisode,
)
from vla_curator.schemas.modalities import DepthMap, SceneGraph


def make_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a deterministic random RGB image."""
    rng = np.random.default_rng(42)
    return (rng.integers(0, 255, (h, w, 3))).astype(np.uint8)


def make_action(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, 7).astype(np.float32)


@pytest.fixture
def sample_reasoning() -> ReasoningTrace:
    return ReasoningTrace(
        task_reasoning="Pick up the red cube and place it in the bowl.",
        subtask_reasoning="Move the arm toward the red cube.",
        move_reasoning="Move end-effector left and down by 5cm.",
        gripper_reasoning="Keep gripper open to approach the object.",
        attribute_reasoning="The cube is red, small, and located on the left side.",
        spatial_reasoning="End-effector is 10cm above and right of the cube.",
    )


@pytest.fixture
def sample_ecot_episode(sample_reasoning: ReasoningTrace) -> ECoTEpisode:
    steps = []
    for i in range(10):
        obs = ECoTObservation(step_index=i, image=make_image())
        reasoning = sample_reasoning if i % 3 == 0 else None
        step = ECoTStep(
            step_index=i,
            observation=obs,
            action=make_action(i),
            reasoning=reasoning,
            is_first=(i == 0),
            is_last=(i == 9),
        )
        steps.append(step)
    return ECoTEpisode(
        episode_id="test_episode_001_42",   # composite key: file_path + "_" + episode_id
        language_instruction="Pick up the red cube and place it in the bowl.",
        steps=steps,
        metadata={
            "ecot_episode_id": "42",
            "file_path": "test_episode_001",
        },
        source_dataset="embodied_features_bridge",
    )


@pytest.fixture
def sample_bridge_episode() -> BridgeEpisode:
    steps = []
    for i in range(10):
        obs = BridgeObservation(
            step_index=i,
            image_0=make_image(),
            image_1=make_image(),
            state=np.zeros(7, dtype=np.float32),
        )
        step = BridgeStep(
            step_index=i,
            observation=obs,
            action=make_action(i),
            language_instruction="Pick up the red cube and place it in the bowl.",
            is_first=(i == 0),
            is_last=(i == 9),
        )
        steps.append(step)
    return BridgeEpisode(
        episode_id="test_episode_001",
        language_instruction="Pick up the red cube and place it in the bowl.",
        steps=steps,
        source_file="test_episode_001",
        episode_num=42,
    )


@pytest.fixture
def sample_interleaved_episode(sample_reasoning: ReasoningTrace) -> InterleavedEpisode:
    steps = []
    for i in range(10):
        obs = EnrichedObservation(
            step_index=i,
            image=make_image(),
            state=np.zeros(7, dtype=np.float32),
            depth_map=DepthMap(valid=False),
            scene_graph=SceneGraph(valid=False),
        )
        step = AlignedStep(
            step_index=i,
            observation=obs,
            action=make_action(i),
            reasoning=sample_reasoning if i % 2 == 0 else None,
            is_first=(i == 0),
            is_last=(i == 9),
            source_dataset="bridge_v2",
            alignment_confidence=1.0 if i % 2 == 0 else 0.7,
        )
        steps.append(step)

    return InterleavedEpisode(
        episode_id="test_episode_001",
        task_description="Pick up the red cube and place it in the bowl.",
        steps=steps,
        alignment_metadata=AlignmentMetadata(
            strategy="nearest",
            ecot_episode_id="test_episode_001",
            bridge_episode_id="test_episode_001",
            num_steps_ecot=10,
            num_steps_bridge=10,
            num_aligned_steps=10,
            num_annotated_steps=5,
            reasoning_coverage=1.0,
        ),
        provenance=DataProvenance(
            generation_backend="gemini",
            generation_model="gemini-1.5-pro",
        ),
        schema_version="1.0",
    )
