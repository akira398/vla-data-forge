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
from vla_curator.curation.interleaver import EpisodeInterleaver, _normalize_episode_id
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
    def test_absolute_nfs_path_stripped(self):
        ecot_path = (
            "/nfs/s3_bucket/username/numpy_256/"
            "bridge_data_v2/datacol2_tabletop_manipulations/put_knife_on_cutting_board/"
            "2023-01-21_13-46-24/0/out.npy"
        )
        result = _normalize_episode_id(ecot_path)
        assert result == (
            "bridge_data_v2/datacol2_tabletop_manipulations/put_knife_on_cutting_board/"
            "2023-01-21_13-46-24/0/out.npy"
        )

    def test_relative_bridge_path_unchanged(self):
        rel_path = "bridge_data_v2/env/task/ep/split/out.npy"
        assert _normalize_episode_id(rel_path) == rel_path

    def test_leading_slash_stripped_fallback(self):
        path = "/bridge_data_v2/env/task/ep/out.npy"
        assert _normalize_episode_id(path) == "bridge_data_v2/env/task/ep/out.npy"

    def test_source_file_takes_priority(self):
        ep_id = "some_other_id"
        source = "/nfs/foo/numpy_256/bridge_data_v2/env/task/ep/out.npy"
        result = _normalize_episode_id(ep_id, source_file=source)
        assert result == "bridge_data_v2/env/task/ep/out.npy"

    def test_backslashes_normalised(self):
        path = "bridge_data_v2\\env\\task\\ep\\out.npy"
        result = _normalize_episode_id(path)
        assert "\\" not in result


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
