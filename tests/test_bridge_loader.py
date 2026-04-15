"""
Tests for the Bridge v2 dataset reader.

All tests use a fake on-disk directory structure — no TF, no network.
The TFDS loading path is mocked at the ``builder_from_directory`` level.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vla_curator.config import BridgeV2DatasetConfig
from vla_curator.datasets.bridge_v2 import (
    BridgeV2DatasetReader,
    _decode_bytes,
    _parse_tfds_episode,
    _parse_tfds_step,
    find_tfds_version_dir,
)


# ---------------------------------------------------------------------------
# find_tfds_version_dir
# ---------------------------------------------------------------------------


class TestFindTfdsVersionDir:
    def test_finds_version_subdir(self, tmp_path):
        """Standard bridge_orig/1.0.0/ layout."""
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        (ver_dir / "dataset_info.json").write_text("{}")

        result = find_tfds_version_dir(tmp_path)
        assert result == ver_dir

    def test_base_path_is_version_dir(self, tmp_path):
        """local_path points directly to the versioned directory."""
        (tmp_path / "dataset_info.json").write_text("{}")
        result = find_tfds_version_dir(tmp_path)
        assert result == tmp_path

    def test_picks_latest_version(self, tmp_path):
        """When multiple versions exist, the latest is returned."""
        for ver in ("1.0.0", "2.0.0", "1.5.0"):
            d = tmp_path / ver
            d.mkdir()
            (d / "dataset_info.json").write_text("{}")

        result = find_tfds_version_dir(tmp_path)
        assert result.name == "2.0.0"

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            find_tfds_version_dir(tmp_path / "nonexistent")

    def test_no_version_dir_raises(self, tmp_path):
        """Folder exists but has no dataset_info.json anywhere."""
        (tmp_path / "some_folder").mkdir()
        with pytest.raises(FileNotFoundError, match="No TFDS version"):
            find_tfds_version_dir(tmp_path)

    def test_ignores_dirs_without_dataset_info(self, tmp_path):
        """A version-named dir without dataset_info.json is not counted."""
        bad = tmp_path / "1.0.0"
        bad.mkdir()
        # No dataset_info.json here

        good = tmp_path / "1.1.0"
        good.mkdir()
        (good / "dataset_info.json").write_text("{}")

        result = find_tfds_version_dir(tmp_path)
        assert result == good


# ---------------------------------------------------------------------------
# _decode_bytes
# ---------------------------------------------------------------------------


class TestDecodeBytes:
    def test_plain_bytes(self):
        assert _decode_bytes(b"hello") == "hello"

    def test_plain_str(self):
        assert _decode_bytes("hello") == "hello"

    def test_tf_tensor_bytes(self):
        """Simulate a TF bytes tensor with a .numpy() method."""
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = b"pick up the cup"
        assert _decode_bytes(mock_tensor) == "pick up the cup"

    def test_none(self):
        assert _decode_bytes(None) == ""


# ---------------------------------------------------------------------------
# _parse_tfds_step
# ---------------------------------------------------------------------------


def _make_mock_step(
    instruction: bytes = b"pick up the cup",
    action: list = None,
    is_first: bool = False,
    is_last: bool = False,
) -> dict:
    """Build a fake TFDS step dict with plain numpy values (no TF required)."""
    h, w = 64, 64
    rng = np.random.default_rng(0)

    return {
        "observation": {
            "image_0": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            "image_1": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
        },
        "action": np.array(action or [0.0] * 7, dtype=np.float32),
        "language_instruction": instruction,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_last,
        "reward": np.float32(0.0),
        "discount": np.float32(1.0),
    }


class TestParseTfdsStep:
    def test_basic_parsing(self):
        raw = _make_mock_step(instruction=b"test task", is_first=True)
        step = _parse_tfds_step(raw, step_index=0, image_size=None, include_secondary=True)

        assert step.step_index == 0
        assert step.language_instruction == "test task"
        assert step.action.shape == (7,)
        assert step.observation.image_0 is not None
        assert step.observation.image_1 is not None
        assert step.is_first is True

    def test_image_resize(self):
        raw = _make_mock_step()
        step = _parse_tfds_step(raw, step_index=0, image_size=(32, 32), include_secondary=False)
        assert step.observation.image_0.shape == (32, 32, 3)

    def test_no_secondary_camera(self):
        raw = _make_mock_step()
        step = _parse_tfds_step(raw, step_index=0, image_size=None, include_secondary=False)
        assert step.observation.image_1 is None

    def test_action_shape(self):
        raw = _make_mock_step(action=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        step = _parse_tfds_step(raw, step_index=0, image_size=None, include_secondary=False)
        assert step.action.shape == (7,)
        assert step.action[6] == pytest.approx(1.0)

    def test_bytes_instruction_decoded(self):
        raw = _make_mock_step(instruction=b"put the apple in the bowl")
        step = _parse_tfds_step(raw, step_index=0, image_size=None, include_secondary=False)
        assert step.language_instruction == "put the apple in the bowl"


# ---------------------------------------------------------------------------
# _parse_tfds_episode
# ---------------------------------------------------------------------------


def _make_mock_episode(
    n_steps: int = 5,
    file_path: bytes = b"bridge_v2/path/to/traj",
    instruction: bytes = b"stack the blocks",
) -> dict:
    steps = [
        _make_mock_step(
            instruction=instruction,
            is_first=(i == 0),
            is_last=(i == n_steps - 1),
        )
        for i in range(n_steps)
    ]
    return {
        "steps": steps,
        "episode_metadata": {"file_path": file_path},
    }


class TestParseTfdsEpisode:
    def test_episode_length(self):
        raw = _make_mock_episode(n_steps=8)
        ep = _parse_tfds_episode(raw, ep_index=0, image_size=None, include_secondary=False)
        assert len(ep) == 8

    def test_instruction_extracted(self):
        raw = _make_mock_episode(instruction=b"stack the blocks")
        ep = _parse_tfds_episode(raw, ep_index=0, image_size=None, include_secondary=False)
        assert ep.language_instruction == "stack the blocks"

    def test_source_file_decoded(self):
        raw = _make_mock_episode(file_path=b"bridge_v2/train/traj_001")
        ep = _parse_tfds_episode(raw, ep_index=0, image_size=None, include_secondary=False)
        assert ep.source_file == "bridge_v2/train/traj_001"
        assert ep.episode_id == "bridge_v2/train/traj_001"

    def test_fallback_episode_id(self):
        raw = _make_mock_episode()
        raw["episode_metadata"]["file_path"] = b""   # empty path
        ep = _parse_tfds_episode(raw, ep_index=42, image_size=None, include_secondary=False)
        assert "042" in ep.episode_id

    def test_is_first_is_last_set(self):
        raw = _make_mock_episode(n_steps=5)
        ep = _parse_tfds_episode(raw, ep_index=0, image_size=None, include_secondary=False)
        assert ep.steps[0].is_first
        assert ep.steps[-1].is_last
        assert not ep.steps[2].is_first
        assert not ep.steps[2].is_last


# ---------------------------------------------------------------------------
# BridgeV2DatasetReader (TFDS path — mocked builder)
# ---------------------------------------------------------------------------


class TestBridgeV2DatasetReaderTFDS:
    """
    Tests for the TFDS reader path.

    ``tensorflow_datasets`` may not be installed in the test environment, so
    we inject a fully-mocked ``tensorflow_datasets`` module into ``sys.modules``
    for the duration of each test.  This also means the tests run on any machine
    without TF, which is the standard CI setup for this project.
    """

    # ---- helpers ----

    @staticmethod
    def _fake_tfds(mock_builder: MagicMock) -> MagicMock:
        """Return a MagicMock that acts as the tensorflow_datasets module."""
        fake = MagicMock()
        fake.builder_from_directory.return_value = mock_builder
        return fake

    @staticmethod
    def _make_config(tmp_path: Path, max_episodes: int = 3) -> BridgeV2DatasetConfig:
        ver_dir = tmp_path / "bridge_orig" / "1.0.0"
        ver_dir.mkdir(parents=True)
        (ver_dir / "dataset_info.json").write_text("{}")
        return BridgeV2DatasetConfig(
            source="tfds",
            local_path=tmp_path / "bridge_orig",
            max_episodes=max_episodes,
        )

    @staticmethod
    def _make_mock_builder(n_episodes: int = 3) -> MagicMock:
        mock_builder = MagicMock()
        mock_builder.info.splits = {"train": MagicMock(num_examples=n_episodes)}
        mock_builder.as_dataset.return_value = iter(
            [_make_mock_episode(n_steps=5) for _ in range(n_episodes)]
        )
        return mock_builder

    # ---- tests ----

    def test_iter_produces_episodes(self, tmp_path):
        cfg = self._make_config(tmp_path)
        mock_builder = self._make_mock_builder(3)
        fake_tfds = self._fake_tfds(mock_builder)

        import sys
        with patch.dict(sys.modules, {"tensorflow_datasets": fake_tfds}):
            reader = BridgeV2DatasetReader(cfg)
            episodes = list(reader)

        assert len(episodes) == 3
        assert all(len(ep) == 5 for ep in episodes)

    def test_max_episodes_respected(self, tmp_path):
        cfg = self._make_config(tmp_path, max_episodes=2)
        mock_builder = self._make_mock_builder(5)
        # Return 5 episodes from the dataset iterator
        mock_builder.as_dataset.return_value = iter(
            [_make_mock_episode(n_steps=5) for _ in range(5)]
        )
        fake_tfds = self._fake_tfds(mock_builder)

        import sys
        with patch.dict(sys.modules, {"tensorflow_datasets": fake_tfds}):
            reader = BridgeV2DatasetReader(cfg)
            episodes = list(reader)

        assert len(episodes) == 2

    def test_len_from_builder_info(self, tmp_path):
        cfg = self._make_config(tmp_path, max_episodes=3)
        mock_builder = self._make_mock_builder(300)
        fake_tfds = self._fake_tfds(mock_builder)

        import sys
        with patch.dict(sys.modules, {"tensorflow_datasets": fake_tfds}):
            reader = BridgeV2DatasetReader(cfg)
            n = len(reader)

        assert n == 3  # capped by max_episodes

    def test_version_dir_passed_to_builder_from_directory(self, tmp_path):
        """Reader must call builder_from_directory with the 1.0.0/ path."""
        cfg = self._make_config(tmp_path)
        mock_builder = self._make_mock_builder(0)
        mock_builder.as_dataset.return_value = iter([])
        fake_tfds = self._fake_tfds(mock_builder)

        import sys
        with patch.dict(sys.modules, {"tensorflow_datasets": fake_tfds}):
            reader = BridgeV2DatasetReader(cfg)
            list(reader)

        called_dir = Path(fake_tfds.builder_from_directory.call_args[0][0])
        assert called_dir.name == "1.0.0"
        assert called_dir.parent.name == "bridge_orig"

    def test_missing_local_path_raises(self, tmp_path):
        """FileNotFoundError is raised when local_path does not exist."""
        cfg = BridgeV2DatasetConfig(
            source="tfds",
            local_path=tmp_path / "nonexistent_bridge_orig",
        )
        fake_tfds = MagicMock()

        import sys
        with patch.dict(sys.modules, {"tensorflow_datasets": fake_tfds}):
            reader = BridgeV2DatasetReader(cfg)
            with pytest.raises(FileNotFoundError):
                reader._get_tfds_builder()
