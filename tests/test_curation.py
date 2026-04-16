"""
Tests for the curation pipeline: interleaving, validation, and export.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from vla_curator.config import CurationConfig, ECoTDatasetConfig, BridgeV2DatasetConfig
from vla_curator.curation.export import JSONLExporter
from vla_curator.curation.interleaver import (
    EpisodeInterleaver,
    _make_composite_key,
    _normalize_episode_id,
    _normalize_path,
)
from vla_curator.curation.validator import DatasetValidator, ValidationResult
from vla_curator.datasets.base import DatasetReader
from vla_curator.schemas.bridge_v2 import BridgeEpisode
from vla_curator.schemas.embodied_cot import ECoTEpisode
from vla_curator.schemas.interleaved import InterleavedEpisode


# ---------------------------------------------------------------------------
# Minimal mock readers
# ---------------------------------------------------------------------------


class MockECoTReader(DatasetReader[ECoTEpisode]):
    dataset_name = "mock_ecot"

    def __init__(self, episodes):
        self._episodes = episodes

    def __iter__(self) -> Iterator[ECoTEpisode]:
        return iter(self._episodes)

    def load_episode(self, episode_id):
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def episode_ids(self):
        return [ep.episode_id for ep in self._episodes]


class MockBridgeReader(DatasetReader[BridgeEpisode]):
    dataset_name = "mock_bridge"

    def __init__(self, episodes):
        self._episodes = episodes

    def __iter__(self) -> Iterator[BridgeEpisode]:
        return iter(self._episodes)

    def load_episode(self, episode_id):
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def episode_ids(self):
        return [ep.episode_id for ep in self._episodes]


# ---------------------------------------------------------------------------
# Interleaver
# ---------------------------------------------------------------------------


class TestEpisodeInterleaver:
    def _make_config(self, strategy="nearest"):
        return CurationConfig(
            ecot=ECoTDatasetConfig(),
            bridge=BridgeV2DatasetConfig(source="hdf5", local_path=Path("/fake")),
            alignment_strategy=strategy,
            output_dir=Path("/tmp/test_output"),
            validate_output=False,
        )

    def test_interleave_produces_correct_length(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config()
        ecot_reader = MockECoTReader([sample_ecot_episode])
        bridge_reader = MockBridgeReader([sample_bridge_episode])
        interleaver = EpisodeInterleaver(cfg, ecot_reader, bridge_reader)

        merged = interleaver.interleave(sample_ecot_episode, sample_bridge_episode)
        assert len(merged) == len(sample_bridge_episode)

    def test_interleave_uses_bridge_actions(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg, MockECoTReader([sample_ecot_episode]), MockBridgeReader([sample_bridge_episode])
        )
        merged = interleaver.interleave(sample_ecot_episode, sample_bridge_episode)

        for merged_step, bridge_step in zip(merged.steps, sample_bridge_episode.steps):
            np.testing.assert_array_equal(merged_step.action, bridge_step.action)

    def test_interleave_nearest_full_coverage(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config(strategy="nearest")
        interleaver = EpisodeInterleaver(
            cfg, MockECoTReader([sample_ecot_episode]), MockBridgeReader([sample_bridge_episode])
        )
        merged = interleaver.interleave(sample_ecot_episode, sample_bridge_episode)
        assert merged.reasoning_coverage() == pytest.approx(1.0)

    def test_interleave_exact_sparse_coverage(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config(strategy="exact")
        interleaver = EpisodeInterleaver(
            cfg, MockECoTReader([sample_ecot_episode]), MockBridgeReader([sample_bridge_episode])
        )
        merged = interleaver.interleave(sample_ecot_episode, sample_bridge_episode)
        # ECoT fixture has reasoning on steps 0, 3, 6, 9 → 4/10
        assert merged.reasoning_coverage() == pytest.approx(0.4)

    def test_iter_episodes_with_matching(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg, MockECoTReader([sample_ecot_episode]), MockBridgeReader([sample_bridge_episode])
        )
        episodes = list(interleaver.iter_episodes())
        assert len(episodes) == 1

    def test_iter_episodes_unmatched_skipped(self, sample_ecot_episode):
        """If Bridge v2 reader has no matching episode, it should be skipped."""
        cfg = self._make_config()
        from vla_curator.schemas.bridge_v2 import BridgeEpisode
        unrelated_bridge = BridgeEpisode(episode_id="completely_different_id")
        interleaver = EpisodeInterleaver(
            cfg,
            MockECoTReader([sample_ecot_episode]),
            MockBridgeReader([unrelated_bridge]),
        )
        episodes = list(interleaver.iter_episodes())
        assert len(episodes) == 0

    def test_alignment_metadata(self, sample_ecot_episode, sample_bridge_episode):
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg, MockECoTReader([sample_ecot_episode]), MockBridgeReader([sample_bridge_episode])
        )
        merged = interleaver.interleave(sample_ecot_episode, sample_bridge_episode)
        meta = merged.alignment_metadata
        assert meta.num_steps_bridge == 10
        assert meta.num_steps_ecot == 10
        assert meta.strategy == "nearest"


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


class TestNormalizeEpisodeId:
    def test_leading_slash_stripped(self):
        path = "/nfs/s3_bucket/username/numpy_256/bridge_data_v2/env/task/ep/out.npy"
        result = _normalize_episode_id(path)
        assert result == "nfs/s3_bucket/username/numpy_256/bridge_data_v2/env/task/ep/out.npy"

    def test_path_without_leading_slash_unchanged(self):
        path = "nfs/s3_bucket/numpy_256/bridge_data_v2/env/task/ep/out.npy"
        assert _normalize_episode_id(path) == path

    def test_same_absolute_path_matches(self):
        abs_path = "/nfs/mount/numpy_256/bridge_data_v2/env/task/ep/out.npy"
        assert _normalize_episode_id(abs_path) == _normalize_episode_id(abs_path)

    def test_source_file_takes_priority(self):
        ep_id = "some_other_id"
        source = "/nfs/foo/numpy_256/bridge_data_v2/env/task/ep/out.npy"
        result = _normalize_episode_id(ep_id, source_file=source)
        assert result == "nfs/foo/numpy_256/bridge_data_v2/env/task/ep/out.npy"

    def test_backslashes_normalised(self):
        path = "nfs\\mount\\bridge_data_v2\\env\\task\\ep\\out.npy"
        result = _normalize_episode_id(path)
        assert "\\" not in result

    def test_composite_key_format(self):
        key = _make_composite_key("/nfs/data/out.npy", 42)
        assert key == "nfs/data/out.npy_42"

    def test_composite_key_matches_ecot_format(self):
        """Composite key from Bridge v2 should match the normalised ECoT episode_id."""
        file_path = "/nfs/data/bridge_data_v2/env/task/ep/out.npy"
        episode_id = 99
        bridge_key = _make_composite_key(file_path, episode_id)
        ecot_composite = f"{file_path}_{episode_id}"
        ecot_key = _normalize_path(ecot_composite)
        assert bridge_key == ecot_key


# ---------------------------------------------------------------------------
# iter_matched_episodes
# ---------------------------------------------------------------------------


class TestIterMatchedEpisodes:
    def _make_config(self):
        return CurationConfig(
            ecot=ECoTDatasetConfig(),
            bridge=BridgeV2DatasetConfig(source="hdf5", local_path=Path("/fake")),
            alignment_strategy="nearest",
            output_dir=Path("/tmp/test_output"),
            validate_output=False,
        )

    def test_matched_episode_has_reasoning(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg,
            MockECoTReader([sample_ecot_episode]),
            MockBridgeReader([sample_bridge_episode]),
        )
        episodes = list(interleaver.iter_matched_episodes())
        assert len(episodes) == 1
        # sample episodes share the same episode_id so they should match
        assert episodes[0].reasoning_coverage() > 0

    def test_unmatched_bridge_episode_skipped(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        """iter_matched_episodes skips Bridge episodes with no ECoT match."""
        cfg = self._make_config()
        unrelated_bridge = BridgeEpisode(
            episode_id="bridge_data_v2/unrelated/path/out.npy"
        )
        interleaver = EpisodeInterleaver(
            cfg,
            MockECoTReader([sample_ecot_episode]),
            MockBridgeReader([unrelated_bridge]),
        )
        episodes = list(interleaver.iter_matched_episodes())
        assert len(episodes) == 0

    def test_episode_id_preserved(self, sample_ecot_episode, sample_bridge_episode):
        """Episode ID must be kept exactly as-is from the Bridge v2 episode."""
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg,
            MockECoTReader([sample_ecot_episode]),
            MockBridgeReader([sample_bridge_episode]),
        )
        episodes = list(interleaver.iter_matched_episodes())
        assert len(episodes) == 1
        assert episodes[0].episode_id == sample_bridge_episode.episode_id

    def test_only_matched_episodes_yielded(
        self, sample_ecot_episode, sample_bridge_episode
    ):
        """Only Bridge v2 episodes with an ECoT match appear in output."""
        unrelated = BridgeEpisode(episode_id="bridge_data_v2/other/out.npy")
        cfg = self._make_config()
        interleaver = EpisodeInterleaver(
            cfg,
            MockECoTReader([sample_ecot_episode]),
            MockBridgeReader([sample_bridge_episode, unrelated]),
        )
        episodes = list(interleaver.iter_matched_episodes())
        assert len(episodes) == 1
        assert episodes[0].episode_id == sample_bridge_episode.episode_id


# ---------------------------------------------------------------------------
# RLDS helpers (no TensorFlow required)
# ---------------------------------------------------------------------------


class TestRLDSHelpers:
    def test_has_reasoning_true(self, sample_interleaved_episode):
        from vla_curator.curation.rlds_export import _has_reasoning
        assert _has_reasoning(sample_interleaved_episode)

    def test_has_reasoning_false_when_no_traces(self):
        from vla_curator.curation.rlds_export import _has_reasoning
        from vla_curator.schemas.interleaved import InterleavedEpisode, AlignedStep
        ep = InterleavedEpisode(
            episode_id="test",
            steps=[AlignedStep(step_index=0, reasoning=None)],
        )
        assert not _has_reasoning(ep)

    def test_pad7_pads_short(self):
        from vla_curator.curation.rlds_export import _pad7
        arr = np.array([1.0, 2.0, 3.0])
        out = _pad7(arr)
        assert out.shape == (7,)
        assert out[2] == pytest.approx(3.0)
        assert out[3] == pytest.approx(0.0)

    def test_pad7_truncates_long(self):
        from vla_curator.curation.rlds_export import _pad7
        arr = np.arange(10, dtype=np.float32)
        out = _pad7(arr)
        assert out.shape == (7,)

    def test_ensure_image_returns_blank_for_none(self):
        from vla_curator.curation.rlds_export import _ensure_image
        img = _ensure_image(None)
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8

    def test_ensure_image_resizes(self):
        from vla_curator.curation.rlds_export import _ensure_image
        small = np.zeros((64, 64, 3), dtype=np.uint8)
        out = _ensure_image(small)
        assert out.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class TestDatasetValidator:
    def setup_method(self):
        self.validator = DatasetValidator()

    def test_valid_episode_passes(self, sample_interleaved_episode):
        result = self.validator.validate_episode(sample_interleaved_episode)
        assert result.passed, f"Errors: {result.errors}"

    def test_empty_episode_fails(self):
        ep = InterleavedEpisode(episode_id="empty", task_description="test")
        result = self.validator.validate_episode(ep)
        assert not result.passed
        assert any("step" in e.lower() for e in result.errors)

    def test_wrong_action_shape_fails(self, sample_interleaved_episode):
        # Corrupt one action
        sample_interleaved_episode.steps[0].action = np.zeros(3)
        result = self.validator.validate_episode(sample_interleaved_episode)
        assert not result.passed

    def test_wrong_step_index_fails(self, sample_interleaved_episode):
        sample_interleaved_episode.steps[2].step_index = 99
        result = self.validator.validate_episode(sample_interleaved_episode)
        assert not result.passed

    def test_missing_is_first_fails(self, sample_interleaved_episode):
        sample_interleaved_episode.steps[0].is_first = False
        result = self.validator.validate_episode(sample_interleaved_episode)
        assert not result.passed

    def test_validate_dataset(self, sample_interleaved_episode):
        report = self.validator.validate_dataset([sample_interleaved_episode])
        assert report.total == 1
        assert report.passed == 1

    def test_reasoning_coverage_warning(self, sample_interleaved_episode):
        validator = DatasetValidator(min_reasoning_coverage=0.99)
        result = validator.validate_episode(sample_interleaved_episode)
        # Coverage is 0.5, below threshold → warning but not error
        assert any("coverage" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class TestJSONLExporter:
    def test_export_writes_file(self, sample_interleaved_episode):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONLExporter(output_dir=Path(tmpdir), save_images=False)
            exporter.export_episode(sample_interleaved_episode)

            out = Path(tmpdir) / "episodes.jsonl"
            assert out.exists()

            with open(out) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 1

            d = json.loads(lines[0])
            assert d["episode_id"] == "test_episode_001"
            assert len(d["steps"]) == 10

    def test_export_saves_images_to_disk(self, sample_interleaved_episode):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONLExporter(
                output_dir=Path(tmpdir),
                save_images=False,
                image_dir=Path(tmpdir) / "imgs",
            )
            exporter.export_episode(sample_interleaved_episode)

            # Image dir should exist and have PNG files
            img_dir = Path(tmpdir) / "imgs"
            pngs = list(img_dir.rglob("*.png"))
            assert len(pngs) == 10  # 10 steps, each with an image

    def test_export_multiple_episodes(self, sample_interleaved_episode):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONLExporter(output_dir=Path(tmpdir), save_images=False)

            # Write 3 copies with different IDs
            for i in range(3):
                ep = InterleavedEpisode(
                    episode_id=f"ep_{i}",
                    task_description="test",
                    steps=sample_interleaved_episode.steps,
                    alignment_metadata=sample_interleaved_episode.alignment_metadata,
                    provenance=sample_interleaved_episode.provenance,
                )
                exporter.export_episode(ep)

            out = Path(tmpdir) / "episodes.jsonl"
            with open(out) as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) == 3
