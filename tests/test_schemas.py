"""
Tests for schema classes.

Covers:
- RobotAction numpy roundtrip
- ReasoningTrace completeness checks
- ECoTEpisode accessors
- InterleavedEpisode serialization
- DepthMap / SceneGraph validity flags
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_curator.schemas.base import RobotAction
from vla_curator.schemas.embodied_cot import ECoTEpisode, ReasoningTrace
from vla_curator.schemas.interleaved import (
    AlignedStep,
    EnrichedObservation,
    InterleavedEpisode,
)
from vla_curator.schemas.modalities import DepthMap, ModalityRegistry, SceneGraph


class TestRobotAction:
    def test_roundtrip(self):
        arr = np.array([0.1, -0.2, 0.05, 0.0, 0.01, -0.01, 1.0], dtype=np.float32)
        action = RobotAction.from_numpy(arr)
        result = action.to_numpy()
        np.testing.assert_allclose(result, arr, atol=1e-6)

    def test_to_list(self):
        arr = np.zeros(7, dtype=np.float32)
        action = RobotAction.from_numpy(arr)
        assert action.to_list() == [0.0] * 7

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            RobotAction.from_numpy(np.zeros(5))

    def test_gripper_field(self):
        action = RobotAction(gripper=0.75)
        assert action.to_numpy()[6] == pytest.approx(0.75)


class TestReasoningTrace:
    def test_complete(self, sample_reasoning):
        assert sample_reasoning.is_complete()

    def test_empty(self):
        t = ReasoningTrace()
        assert t.is_empty()
        assert not t.is_complete()

    def test_partial_not_complete(self):
        t = ReasoningTrace(task_reasoning="do something")
        assert not t.is_complete()
        assert not t.is_empty()

    def test_filled_fields(self, sample_reasoning):
        fields = sample_reasoning.filled_fields()
        assert "task_reasoning" in fields
        assert "move_reasoning" in fields
        assert "plan" in fields
        assert "subtask_reason" in fields
        assert "move_reason" in fields
        assert len(fields) == 9

    def test_to_dict_roundtrip(self, sample_reasoning):
        d = sample_reasoning.to_dict()
        restored = ReasoningTrace.from_dict(d)
        assert restored.task_reasoning == sample_reasoning.task_reasoning
        assert restored.move_reasoning == sample_reasoning.move_reasoning

    def test_extra_field(self):
        t = ReasoningTrace(extra={"custom": "value"})
        assert t.extra["custom"] == "value"


class TestECoTEpisode:
    def test_len(self, sample_ecot_episode):
        assert len(sample_ecot_episode) == 10

    def test_iter(self, sample_ecot_episode):
        steps = list(sample_ecot_episode)
        assert len(steps) == 10

    def test_getitem(self, sample_ecot_episode):
        step = sample_ecot_episode[0]
        assert step.step_index == 0
        assert step.is_first

    def test_get_actions_shape(self, sample_ecot_episode):
        actions = sample_ecot_episode.get_actions()
        assert actions.shape == (10, 7)

    def test_reasoning_coverage(self, sample_ecot_episode):
        coverage = sample_ecot_episode.reasoning_coverage()
        # Steps 0, 3, 6, 9 have reasoning → 4/10 = 0.4
        assert coverage == pytest.approx(0.4)

    def test_has_any_reasoning(self, sample_ecot_episode):
        assert sample_ecot_episode.has_any_reasoning()

    def test_annotated_steps(self, sample_ecot_episode):
        annotated = sample_ecot_episode.annotated_steps()
        assert len(annotated) == 4  # steps 0, 3, 6, 9

    def test_to_dict_no_images(self, sample_ecot_episode):
        d = sample_ecot_episode.to_dict(include_images=False)
        assert d["episode_id"] == "test_episode_001_42"
        assert len(d["steps"]) == 10
        # Images should not be in the output
        for step in d["steps"]:
            assert "image" not in step or step.get("image") is None

    def test_repr(self, sample_ecot_episode):
        r = repr(sample_ecot_episode)
        assert "test_episode_001" in r
        assert "reasoning_coverage" in r.lower() or "40%" in r


class TestInterleavedEpisode:
    def test_reasoning_coverage(self, sample_interleaved_episode):
        cov = sample_interleaved_episode.reasoning_coverage()
        # Steps 0, 2, 4, 6, 8 have reasoning → 5/10 = 0.5
        assert cov == pytest.approx(0.5)

    def test_has_reasoning(self, sample_interleaved_episode):
        assert sample_interleaved_episode.has_reasoning()

    def test_high_confidence_steps(self, sample_interleaved_episode):
        hc = sample_interleaved_episode.high_confidence_steps(threshold=0.9)
        # Steps with confidence=1.0 → even steps (0, 2, 4, 6, 8) = 5 steps
        assert len(hc) == 5

    def test_to_dict_structure(self, sample_interleaved_episode):
        d = sample_interleaved_episode.to_dict()
        assert "schema_version" in d
        assert "alignment_metadata" in d
        assert "provenance" in d
        assert len(d["steps"]) == 10


class TestModalities:
    def test_depth_map_default_invalid(self):
        dm = DepthMap()
        assert not dm.valid
        assert dm.load() is None

    def test_scene_graph_default_invalid(self):
        sg = SceneGraph()
        assert not sg.valid
        assert sg.nodes == []

    def test_modality_registry_has_depth(self):
        assert ModalityRegistry.has("depth_map")
        assert ModalityRegistry.has("scene_graph")

    def test_modality_registry_missing(self):
        with pytest.raises(KeyError):
            ModalityRegistry.get_spec("nonexistent_modality")
