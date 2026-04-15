"""
Tests for BridgeViewer.

All tests use synthetic BridgeEpisode / BridgeStep objects built from plain
numpy arrays — no TF, no GPU, no real dataset required.

Figures are created with a non-interactive matplotlib backend so the tests
run headlessly in CI.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vla_curator.schemas.bridge_v2 import BridgeEpisode, BridgeObservation, BridgeStep
from vla_curator.visualization.bridge_viewer import BridgeViewer, _img, _img1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_obs(
    h: int = 32,
    w: int = 32,
    include_secondary: bool = True,
) -> BridgeObservation:
    rng = np.random.default_rng(0)
    image_0 = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    image_1 = rng.integers(0, 255, (h, w, 3), dtype=np.uint8) if include_secondary else None
    state = np.zeros(7, dtype=np.float32)
    return BridgeObservation(image_0=image_0, image_1=image_1, state=state)


def _make_step(
    step_index: int,
    is_first: bool = False,
    is_last: bool = False,
    gripper: float = 0.0,
    include_secondary: bool = True,
) -> BridgeStep:
    action = np.array([0.1, -0.1, 0.05, 0.0, 0.0, 0.0, gripper], dtype=np.float32)
    return BridgeStep(
        step_index=step_index,
        observation=_make_obs(include_secondary=include_secondary),
        action=action,
        language_instruction="pick up the cup",
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
        reward=0.0,
        discount=1.0,
    )


def _make_episode(
    n_steps: int = 10,
    include_secondary: bool = True,
) -> BridgeEpisode:
    steps = [
        _make_step(
            i,
            is_first=(i == 0),
            is_last=(i == n_steps - 1),
            gripper=1.0 if i >= n_steps // 2 else 0.0,
            include_secondary=include_secondary,
        )
        for i in range(n_steps)
    ]
    return BridgeEpisode(
        episode_id="test/ep_000",
        language_instruction="pick up the cup",
        steps=steps,
        source_file="test/ep_000",
    )


@pytest.fixture
def episode() -> BridgeEpisode:
    return _make_episode(n_steps=10)


@pytest.fixture
def viewer() -> BridgeViewer:
    return BridgeViewer()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_img_returns_array_from_image_0(self):
        obs = _make_obs()
        result = _img(obs)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 3)

    def test_img1_returns_secondary_array(self):
        obs = _make_obs(include_secondary=True)
        result = _img1(obs)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_img1_returns_none_when_no_secondary(self):
        obs = _make_obs(include_secondary=False)
        result = _img1(obs)
        assert result is None

    def test_img_uses_load_image_0_method(self):
        """If obs has load_image_0 method, _img calls it."""
        mock_obs = MagicMock()
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_obs.load_image_0.return_value = arr
        result = _img(mock_obs)
        mock_obs.load_image_0.assert_called_once()
        assert result is arr

    def test_img_falls_back_to_primary_image_method(self):
        """Falls back to primary_image() if load_image_0 is not available."""
        mock_obs = MagicMock(spec=[])  # no load_image_0
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_obs.primary_image = MagicMock(return_value=arr)
        # _img uses getattr fallback chain
        result = _img(mock_obs)
        assert result is arr


# ---------------------------------------------------------------------------
# BridgeViewer.show_dual_camera
# ---------------------------------------------------------------------------


class TestShowDualCamera:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_dual_camera(episode, step_index=0, save_path=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "dual.png"
        viewer.show_dual_camera(episode, step_index=0, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_out_of_range_step_raises(self, viewer, episode):
        with pytest.raises(IndexError):
            viewer.show_dual_camera(episode, step_index=999)

    def test_last_step(self, viewer, episode):
        last = len(episode.steps) - 1
        fig = viewer.show_dual_camera(episode, step_index=last)
        plt.close(fig)

    def test_no_secondary_camera(self, viewer):
        ep = _make_episode(n_steps=5, include_secondary=False)
        fig = viewer.show_dual_camera(ep, step_index=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.show_episode_dual_camera
# ---------------------------------------------------------------------------


class TestShowEpisodeDualCamera:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_episode_dual_camera(episode, max_frames=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "grid.png"
        viewer.show_episode_dual_camera(episode, max_frames=4, save_path=out)
        assert out.exists()

    def test_subsampling(self, viewer):
        ep = _make_episode(n_steps=20)
        # max_frames < n_steps → subsampling must not crash
        fig = viewer.show_episode_dual_camera(ep, max_frames=4)
        plt.close(fig)

    def test_single_step_episode(self, viewer):
        ep = _make_episode(n_steps=1)
        fig = viewer.show_episode_dual_camera(ep, max_frames=8)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.show_gripper_state
# ---------------------------------------------------------------------------


class TestShowGripperState:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_gripper_state(episode)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "gripper.png"
        viewer.show_gripper_state(episode, save_path=out)
        assert out.exists()

    def test_all_open(self, viewer):
        """No transitions when gripper is always open."""
        steps = [_make_step(i, gripper=0.0) for i in range(8)]
        ep = BridgeEpisode(
            episode_id="ep_open",
            language_instruction="test",
            steps=steps,
            source_file="",
        )
        fig = viewer.show_gripper_state(ep)
        plt.close(fig)

    def test_all_closed(self, viewer):
        steps = [_make_step(i, gripper=1.0) for i in range(8)]
        ep = BridgeEpisode(
            episode_id="ep_closed",
            language_instruction="test",
            steps=steps,
            source_file="",
        )
        fig = viewer.show_gripper_state(ep)
        plt.close(fig)

    def test_multiple_transitions(self, viewer):
        """Gripper oscillates open/close across many transitions."""
        grippers = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        steps = [_make_step(i, gripper=g) for i, g in enumerate(grippers)]
        ep = BridgeEpisode(
            episode_id="ep_osc",
            language_instruction="test",
            steps=steps,
            source_file="",
        )
        fig = viewer.show_gripper_state(ep)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.show_state_trajectory
# ---------------------------------------------------------------------------


class TestShowStateTrajectory:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_state_trajectory(episode)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "state.png"
        viewer.show_state_trajectory(episode, save_path=out)
        assert out.exists()

    def test_no_state_data(self, viewer):
        """If observation.state is None, a placeholder figure is returned."""
        obs = BridgeObservation(image_0=np.zeros((32, 32, 3), dtype=np.uint8), state=None)
        step = BridgeStep(
            step_index=0,
            observation=obs,
            action=np.zeros(7, dtype=np.float32),
            language_instruction="test",
            is_first=True,
            is_last=True,
            is_terminal=True,
            reward=0.0,
            discount=1.0,
        )
        ep = BridgeEpisode(
            episode_id="no_state", language_instruction="test", steps=[step], source_file=""
        )
        fig = viewer.show_state_trajectory(ep)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.show_action_components
# ---------------------------------------------------------------------------


class TestShowActionComponents:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_action_components(episode)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "actions.png"
        viewer.show_action_components(episode, save_path=out)
        assert out.exists()

    def test_single_step_episode(self, viewer):
        ep = _make_episode(n_steps=1)
        fig = viewer.show_action_components(ep)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.show_summary
# ---------------------------------------------------------------------------


class TestShowSummary:
    def test_returns_figure(self, viewer, episode):
        fig = viewer.show_summary(episode, max_frames=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, viewer, episode, tmp_path):
        out = tmp_path / "summary.png"
        viewer.show_summary(episode, max_frames=4, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_more_steps_than_max_frames(self, viewer):
        ep = _make_episode(n_steps=30)
        fig = viewer.show_summary(ep, max_frames=5)
        plt.close(fig)

    def test_no_secondary_camera(self, viewer):
        ep = _make_episode(n_steps=8, include_secondary=False)
        fig = viewer.show_summary(ep, max_frames=4)
        plt.close(fig)


# ---------------------------------------------------------------------------
# BridgeViewer.save_episode_gif
# ---------------------------------------------------------------------------


class TestSaveEpisodeGif:
    def test_gif_saved_to_disk(self, viewer, episode, tmp_path):
        imageio = pytest.importorskip("imageio")
        out = tmp_path / "episode.gif"
        viewer.save_episode_gif(episode, out, fps=2, camera=0)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_secondary_camera_gif(self, viewer, episode, tmp_path):
        pytest.importorskip("imageio")
        out = tmp_path / "cam1.gif"
        viewer.save_episode_gif(episode, out, fps=2, camera=1)
        assert out.exists()

    def test_max_frames_subsampling(self, viewer, tmp_path):
        pytest.importorskip("imageio")
        ep = _make_episode(n_steps=20)
        out = tmp_path / "short.gif"
        viewer.save_episode_gif(ep, out, fps=4, camera=0, max_frames=5)
        assert out.exists()

    def test_no_imageio_raises_import_error(self, viewer, episode, tmp_path):
        """ImportError with helpful message when imageio is missing."""
        import sys
        with patch.dict(sys.modules, {"imageio": None}):
            with pytest.raises(ImportError, match="imageio"):
                viewer.save_episode_gif(episode, tmp_path / "fail.gif")

    def test_no_camera_images_raises(self, viewer, tmp_path):
        """ValueError when no images are available for the chosen camera."""
        ep = _make_episode(n_steps=3, include_secondary=False)
        pytest.importorskip("imageio")
        with pytest.raises(ValueError, match="camera 1"):
            viewer.save_episode_gif(ep, tmp_path / "fail.gif", camera=1)
