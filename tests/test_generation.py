"""
Tests for the generation pipeline components.

No model API calls — all backend calls are mocked.
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from vla_curator.backends.base import GenerationResult, Prompt
from vla_curator.config import FrameSamplingConfig
from vla_curator.generation.prompt_builder import (
    ECoTPromptBuilder,
    sample_frames_keyframe,
    sample_frames_uniform,
)
from vla_curator.generation.response_parser import ReasoningTraceParser
from vla_curator.generation.trace_postprocessor import (
    TracePostprocessor,
    clean_trace,
    propagate_nearest,
)
from vla_curator.schemas.embodied_cot import ReasoningTrace


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestFrameSampling:
    def test_uniform_fewer_than_n(self, sample_ecot_episode):
        indices = sample_frames_uniform(sample_ecot_episode.steps, n=20)
        assert indices == list(range(10))

    def test_uniform_exact_n(self, sample_ecot_episode):
        indices = sample_frames_uniform(sample_ecot_episode.steps, n=5)
        assert len(indices) == 5
        assert indices[0] == 0
        assert indices[-1] == 9

    def test_uniform_single(self, sample_ecot_episode):
        indices = sample_frames_uniform(sample_ecot_episode.steps, n=1)
        assert len(indices) == 1

    def test_keyframe_includes_endpoints(self, sample_ecot_episode):
        indices = sample_frames_keyframe(sample_ecot_episode.steps)
        assert 0 in indices
        assert 9 in indices


class TestECoTPromptBuilder:
    def test_build_episode_prompt(self, sample_ecot_episode):
        cfg = FrameSamplingConfig(strategy="uniform", num_frames=4)
        builder = ECoTPromptBuilder(frame_sampling=cfg)
        prompt, frame_indices = builder.build_episode_prompt(sample_ecot_episode)

        assert isinstance(prompt, Prompt)
        assert len(frame_indices) == 4
        assert "Pick up" in prompt.text
        assert "test_episode_001" in prompt.metadata["episode_id"]

    def test_prompt_has_images(self, sample_ecot_episode):
        cfg = FrameSamplingConfig(strategy="uniform", num_frames=4)
        builder = ECoTPromptBuilder(frame_sampling=cfg)
        prompt, _ = builder.build_episode_prompt(sample_ecot_episode)
        # All 10 steps have images in fixture
        assert len(prompt.images) == 4

    def test_keyframe_strategy(self, sample_ecot_episode):
        cfg = FrameSamplingConfig(strategy="keyframe", num_frames=8, keyframe_threshold=0.0)
        builder = ECoTPromptBuilder(frame_sampling=cfg)
        _, frame_indices = builder.build_episode_prompt(sample_ecot_episode)
        assert len(frame_indices) >= 2  # At least first and last


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


_VALID_JSON_RESPONSE = json.dumps([
    {
        "frame_index": 0,
        "task_reasoning": "Pick up the cube.",
        "subtask_reasoning": "Approach the cube.",
        "move_reasoning": "Move arm left.",
        "gripper_reasoning": "Keep open.",
        "attribute_reasoning": "Red, small cube.",
        "spatial_reasoning": "Cube is to the left.",
    },
    {
        "frame_index": 5,
        "task_reasoning": "Continuing pick-up.",
        "subtask_reasoning": "Grasp cube.",
        "move_reasoning": "Lower arm.",
        "gripper_reasoning": "Close gripper.",
        "attribute_reasoning": "Red cube.",
        "spatial_reasoning": "Above the cube.",
    },
])

_MARKDOWN_WRAPPED = f"```json\n{_VALID_JSON_RESPONSE}\n```"

_SINGLE_OBJECT = json.dumps({
    "task_reasoning": "Pick and place.",
    "subtask_reasoning": "Approaching.",
    "move_reasoning": "Move left.",
    "gripper_reasoning": "Open.",
})


class TestReasoningTraceParser:
    def setup_method(self):
        self.parser = ReasoningTraceParser()

    def test_parse_episode_valid_json(self):
        pairs = self.parser.parse_episode_response(_VALID_JSON_RESPONSE, [0, 5])
        assert len(pairs) == 2
        assert pairs[0][0] == 0
        assert pairs[0][1].task_reasoning == "Pick up the cube."

    def test_parse_episode_markdown_wrapped(self):
        pairs = self.parser.parse_episode_response(_MARKDOWN_WRAPPED, [0, 5])
        assert len(pairs) == 2

    def test_parse_episode_positional_fallback(self):
        # Response without frame_index — uses positional matching
        resp = json.dumps([{"task_reasoning": "A"}, {"task_reasoning": "B"}])
        pairs = self.parser.parse_episode_response(resp, [3, 7])
        assert pairs[0][0] == 3
        assert pairs[1][0] == 7

    def test_parse_step_single_object(self):
        trace = self.parser.parse_step_response(_SINGLE_OBJECT)
        assert trace.task_reasoning == "Pick and place."
        assert trace.move_reasoning == "Move left."

    def test_parse_step_bad_json_returns_raw(self):
        trace = self.parser.parse_step_response("this is not json at all")
        assert trace.raw_response == "this is not json at all"
        assert trace.is_empty()

    def test_parse_episode_bad_json_fills_all(self):
        pairs = self.parser.parse_episode_response("bad", [0, 3, 6])
        assert len(pairs) == 3
        assert all(t.raw_response == "bad" for _, t in pairs)


# ---------------------------------------------------------------------------
# Trace postprocessor
# ---------------------------------------------------------------------------


class TestTracePostprocessor:
    def test_clean_removes_na(self):
        t = ReasoningTrace(task_reasoning="N/A", move_reasoning="move left")
        cleaned = clean_trace(t)
        assert cleaned.task_reasoning is None
        assert cleaned.move_reasoning == "move left"

    def test_propagate_nearest_forward(self):
        t = ReasoningTrace(task_reasoning="task")
        traces = [t, None, None, None]
        result = propagate_nearest(traces)
        assert all(r is not None for r in result)

    def test_propagate_nearest_backward(self):
        t = ReasoningTrace(task_reasoning="task")
        traces = [None, None, t]
        result = propagate_nearest(traces)
        assert result[0] is not None

    def test_postprocessor_coverage(self):
        pp = TracePostprocessor(propagation="nearest")
        t = ReasoningTrace(task_reasoning="task", move_reasoning="move")
        sparse = {2: t}
        dense = pp.process_episode(sparse, num_steps=5)
        assert len(dense) == 5
        assert all(d is not None for d in dense)
        assert pp.coverage(dense) == 1.0

    def test_postprocessor_exact_no_propagation(self):
        pp = TracePostprocessor(propagation="none")
        t = ReasoningTrace(task_reasoning="task")
        sparse = {2: t}
        dense = pp.process_episode(sparse, num_steps=5)
        assert dense[2] is not None
        assert dense[0] is None
        assert dense[4] is None

    def test_validate_report(self):
        pp = TracePostprocessor(propagation="nearest")
        t = ReasoningTrace(
            task_reasoning="t",
            subtask_reasoning="s",
            move_reasoning="m",
            gripper_reasoning="g",
        )
        traces = [t, t, None, t]
        report = pp.validate(traces)
        assert report["total_steps"] == 4
        assert report["annotated_steps"] == 3
        assert report["coverage"] == pytest.approx(0.75)
