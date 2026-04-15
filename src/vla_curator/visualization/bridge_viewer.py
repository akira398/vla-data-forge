"""
Bridge v2 visualization utilities.

Bridge v2-specific features on top of the generic FrameViewer/TrajectoryViewer:

  show_dual_camera()         — primary and secondary views side-by-side per step
  show_episode_dual_camera() — full episode grid with both camera columns
  show_gripper_state()       — colour-coded gripper open/close transitions
  show_state_trajectory()    — 7-DoF proprioceptive state over time
  show_action_components()   — translation vs. rotation vs. gripper sub-plots
  show_summary()             — composite 4-panel dashboard for one episode

All functions follow the same convention as the rest of the visualization
package: return the Figure, show it inline unless ``save_path`` is given.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _img(obs: Any) -> Optional[np.ndarray]:
    """Load primary image from any BridgeObservation-like object."""
    fn = getattr(obs, "load_image_0", None) or getattr(obs, "primary_image", None)
    if fn:
        return fn()
    return getattr(obs, "image_0", None)


def _img1(obs: Any) -> Optional[np.ndarray]:
    """Load secondary image."""
    fn = getattr(obs, "load_image_1", None)
    if fn:
        return fn()
    return getattr(obs, "image_1", None)


def _no_image_ax(ax: Any, label: str = "No image") -> None:
    ax.set_facecolor("#d0d0d0")
    ax.text(0.5, 0.5, label, ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="#555555")
    ax.axis("off")


# ---------------------------------------------------------------------------
# BridgeViewer
# ---------------------------------------------------------------------------


class BridgeViewer:
    """
    Visualization suite for Bridge v2 episodes.

    Usage
    -----
    from vla_curator.datasets import BridgeV2DatasetReader
    from vla_curator.config import BridgeV2DatasetConfig
    from vla_curator.visualization import BridgeViewer
    from pathlib import Path

    cfg = BridgeV2DatasetConfig(
        source="tfds",
        local_path=Path("/datasets/bridge_orig"),
        max_episodes=1,
    )
    ep = BridgeV2DatasetReader(cfg).take(1)[0]

    viewer = BridgeViewer()
    viewer.show_dual_camera(ep, step_index=0)
    viewer.show_episode_dual_camera(ep, max_frames=8)
    viewer.show_gripper_state(ep)
    viewer.show_summary(ep)
    """

    def __init__(self, figsize_per_frame: tuple[int, int] = (3, 3)) -> None:
        self.fpf = figsize_per_frame

    # ------------------------------------------------------------------
    # 1. Single step — both cameras
    # ------------------------------------------------------------------

    def show_dual_camera(
        self,
        episode: Any,
        step_index: int = 0,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Show one step's primary and secondary camera views side by side.
        Includes the action vector and instruction as text.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        if step_index >= len(steps):
            raise IndexError(f"step_index {step_index} out of range ({len(steps)} steps).")
        step = steps[step_index]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        for ax, load_fn, label in zip(
            axes,
            [_img(step.observation), _img1(step.observation)],
            ["Camera 0 (primary)", "Camera 1 (secondary)"],
        ):
            if load_fn is not None:
                ax.imshow(load_fn if isinstance(load_fn, np.ndarray) else load_fn)
            else:
                _no_image_ax(ax, label)
            ax.set_title(label, fontsize=9)
            ax.axis("off")

        # Action annotation
        action = step.action
        action_str = (
            f"Δxyz  [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]\n"
            f"Δrpy  [{action[3]:+.3f}, {action[4]:+.3f}, {action[5]:+.3f}]\n"
            f"grip  {action[6]:.3f}  ({'CLOSE' if action[6] > 0.5 else 'OPEN'})"
        )
        fig.text(
            0.5, 0.01, action_str,
            ha="center", va="bottom", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9),
        )

        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(
            f'Step {step_index} / {len(steps)-1} — "{instr[:80]}"',
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 2. Full episode grid — dual camera columns
    # ------------------------------------------------------------------

    def show_episode_dual_camera(
        self,
        episode: Any,
        max_frames: int = 8,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Grid layout: one row per sampled frame, two columns (cam0 | cam1).
        Each row is labelled with the step index and gripper state.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        if len(steps) > max_frames:
            idxs = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in idxs]

        n = len(steps)
        fw, fh = self.fpf
        fig, axes = plt.subplots(n, 2, figsize=(fw * 2, fh * n))
        if n == 1:
            axes = np.array([axes])  # ensure 2-D indexing

        for row, step in enumerate(steps):
            for col, (load_img, cam_label) in enumerate([
                (_img(step.observation),  "cam0"),
                (_img1(step.observation), "cam1"),
            ]):
                ax = axes[row, col]
                if load_img is not None:
                    arr = load_img if isinstance(load_img, np.ndarray) else load_img
                    ax.imshow(arr)
                else:
                    _no_image_ax(ax, cam_label)

                gripper = step.action[6]
                grip_str = f"G={gripper:.2f}"
                flags = []
                if step.is_first:
                    flags.append("START")
                if step.is_last:
                    flags.append("END")
                flag_str = " ".join(flags)

                title = f"s{step.step_index} {cam_label}"
                if col == 1 and (grip_str or flag_str):
                    title += f"  {grip_str}"
                    if flag_str:
                        title += f"  [{flag_str}]"
                ax.set_title(title, fontsize=6, pad=2)
                ax.axis("off")

        # Column headers
        axes[0, 0].set_title(
            f"Camera 0 (primary)   s{steps[0].step_index}", fontsize=7, pad=3
        )
        axes[0, 1].set_title(
            f"Camera 1 (secondary)   G={steps[0].action[6]:.2f}", fontsize=7, pad=3
        )

        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'"{instr[:90]}"', fontsize=9, y=1.001)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 3. Gripper state timeline
    # ------------------------------------------------------------------

    def show_gripper_state(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Plot gripper value over time with colour bands marking OPEN vs CLOSE
        phases.  Transition steps are highlighted with vertical red lines.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        steps = list(episode.steps)
        T = len(steps)
        t = np.arange(T)
        gripper = np.array([s.action[6] for s in steps], dtype=np.float32)

        # Detect transitions (gripper state flips across 0.5 threshold)
        binary = (gripper > 0.5).astype(int)
        transitions = np.where(np.diff(binary) != 0)[0] + 1

        fig, ax = plt.subplots(figsize=(min(T * 0.25 + 2, 14), 3))

        # Shade open (green) / close (orange) regions
        prev = 0
        for tr in list(transitions) + [T]:
            is_open = binary[prev] == 0
            ax.axvspan(
                prev - 0.5, tr - 0.5,
                alpha=0.15,
                color="#2ca02c" if is_open else "#ff7f0e",
            )
            prev = tr

        ax.plot(t, gripper, "k-", linewidth=1.5, zorder=3)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

        for tr in transitions:
            ax.axvline(tr, color="red", linewidth=1.0, linestyle=":", alpha=0.8)

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Gripper", fontsize=9)
        ax.tick_params(labelsize=8)

        open_patch  = mpatches.Patch(color="#2ca02c", alpha=0.4, label="OPEN  (< 0.5)")
        close_patch = mpatches.Patch(color="#ff7f0e", alpha=0.4, label="CLOSE (≥ 0.5)")
        ax.legend(handles=[open_patch, close_patch], fontsize=8, loc="upper right")

        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        ax.set_title(f'Gripper timeline — "{instr[:70]}"', fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 4. Proprioceptive state trajectory
    # ------------------------------------------------------------------

    def show_state_trajectory(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Plot the 7-DoF robot proprioceptive state (joint positions / EEF pose)
        stored in each step's ``observation.state``.  Skips silently if state
        data is unavailable.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        states = []
        for s in steps:
            st = getattr(s.observation, "state", None)
            if st is None and hasattr(s.observation, "load_state"):
                st = s.observation.load_state()
            if st is not None:
                states.append(np.asarray(st, dtype=np.float32).flatten())

        if not states:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No state data available for this episode.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.axis("off")
            if save_path:
                fig.savefig(save_path, dpi=100, bbox_inches="tight")
            else:
                plt.show()
            return fig

        state_arr = np.stack(states, axis=0)   # (T, D)
        T, D = state_arr.shape
        t = np.arange(T)

        labels = (
            ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            if D == 7 else [f"s{i}" for i in range(D)]
        )
        colors = plt.cm.tab10(np.linspace(0, 0.9, D))

        fig, axes = plt.subplots(D, 1, figsize=(10, D * 1.3), sharex=True)
        if D == 1:
            axes = [axes]

        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(t, state_arr[:, i], color=color, linewidth=1.2)
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step", fontsize=9)
        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'State trajectory — "{instr[:70]}"', fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 5. Action component breakdown
    # ------------------------------------------------------------------

    def show_action_components(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Three-panel action plot:
          Top    — XYZ translation deltas
          Middle — Roll/Pitch/Yaw rotation deltas
          Bottom — Gripper value
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        actions = np.stack([s.action for s in steps], axis=0)  # (T, 7)
        T = actions.shape[0]
        t = np.arange(T)

        fig, (ax_xyz, ax_rpy, ax_grip) = plt.subplots(3, 1, figsize=(11, 6), sharex=True)

        # --- Translation ---
        for i, (lbl, col) in enumerate(zip(
            ["Δx", "Δy", "Δz"],
            ["#e41a1c", "#377eb8", "#4daf4a"]
        )):
            ax_xyz.plot(t, actions[:, i], label=lbl, color=col, linewidth=1.3)
        ax_xyz.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_xyz.set_ylabel("Translation (Δm)", fontsize=9)
        ax_xyz.legend(fontsize=8, loc="upper right", ncol=3)
        ax_xyz.grid(True, alpha=0.3)

        # --- Rotation ---
        for i, (lbl, col) in enumerate(zip(
            ["Δroll", "Δpitch", "Δyaw"],
            ["#984ea3", "#ff7f00", "#a65628"]
        )):
            ax_rpy.plot(t, actions[:, 3 + i], label=lbl, color=col, linewidth=1.3)
        ax_rpy.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_rpy.set_ylabel("Rotation (Δrad)", fontsize=9)
        ax_rpy.legend(fontsize=8, loc="upper right", ncol=3)
        ax_rpy.grid(True, alpha=0.3)

        # --- Gripper ---
        ax_grip.fill_between(t, actions[:, 6], alpha=0.4, color="#000000")
        ax_grip.plot(t, actions[:, 6], color="black", linewidth=1.2)
        ax_grip.axhline(0.5, color="red", linewidth=0.8, linestyle=":", alpha=0.7,
                        label="open/close threshold")
        ax_grip.set_ylim(-0.05, 1.1)
        ax_grip.set_ylabel("Gripper", fontsize=9)
        ax_grip.set_xlabel("Step", fontsize=9)
        ax_grip.legend(fontsize=8, loc="upper right")
        ax_grip.grid(True, alpha=0.3)

        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'Action components — "{instr[:70]}"', fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 6. Composite summary dashboard
    # ------------------------------------------------------------------

    def show_summary(
        self,
        episode: Any,
        max_frames: int = 6,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Four-panel summary dashboard:
          Row 0 — sampled primary-camera frames
          Row 1 — sampled secondary-camera frames
          Row 2 — XYZ translation + gripper on twin axes
          Row 3 — gripper timeline with open/close shading
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        steps = list(episode.steps)
        T = len(steps)

        # Subsample frames
        if T > max_frames:
            idxs = np.linspace(0, T - 1, max_frames, dtype=int).tolist()
        else:
            idxs = list(range(T))
        sampled = [steps[i] for i in idxs]
        n = len(sampled)

        fig = plt.figure(figsize=(max(n * 2.5, 12), 11))
        gs = gridspec.GridSpec(4, n, figure=fig, hspace=0.55, wspace=0.15,
                               height_ratios=[3, 3, 3, 2])

        # --- Row 0: primary camera ---
        for col, step in enumerate(sampled):
            ax = fig.add_subplot(gs[0, col])
            img = _img(step.observation)
            if img is not None:
                ax.imshow(img)
            else:
                _no_image_ax(ax)
            ax.axis("off")
            ax.set_title(f"s{step.step_index}\ncam0", fontsize=6, pad=1)

        # --- Row 1: secondary camera ---
        for col, step in enumerate(sampled):
            ax = fig.add_subplot(gs[1, col])
            img = _img1(step.observation)
            if img is not None:
                ax.imshow(img)
            else:
                _no_image_ax(ax, "cam1\n(N/A)")
            ax.axis("off")
            ax.set_title(f"s{step.step_index}\ncam1", fontsize=6, pad=1)

        # --- Row 2: XYZ + gripper on twin axes ---
        ax_traj = fig.add_subplot(gs[2, :])
        actions = np.stack([s.action for s in steps], axis=0)
        t = np.arange(T)
        colors_xyz = ["#e41a1c", "#377eb8", "#4daf4a"]
        for i, (lbl, c) in enumerate(zip(["Δx", "Δy", "Δz"], colors_xyz)):
            ax_traj.plot(t, actions[:, i], color=c, label=lbl, linewidth=1.2, alpha=0.9)
        ax_traj.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_traj.set_ylabel("Translation", fontsize=8)
        ax_traj.tick_params(labelsize=7)
        ax_traj.legend(fontsize=7, loc="upper left", ncol=3)
        ax_traj.grid(True, alpha=0.25)

        ax_grip2 = ax_traj.twinx()
        ax_grip2.plot(t, actions[:, 6], color="black", linewidth=1.2,
                      linestyle="--", alpha=0.7, label="gripper")
        ax_grip2.set_ylim(-0.1, 1.2)
        ax_grip2.set_ylabel("Gripper", fontsize=8)
        ax_grip2.tick_params(labelsize=7)
        ax_grip2.legend(fontsize=7, loc="upper right")

        # --- Row 3: gripper timeline ---
        ax_g = fig.add_subplot(gs[3, :])
        gripper = actions[:, 6]
        binary = (gripper > 0.5).astype(int)
        transitions = np.where(np.diff(binary) != 0)[0] + 1
        prev = 0
        for tr in list(transitions) + [T]:
            is_open = binary[prev] == 0
            ax_g.axvspan(prev - 0.5, tr - 0.5, alpha=0.2,
                         color="#2ca02c" if is_open else "#ff7f0e")
            prev = tr
        ax_g.plot(t, gripper, "k-", linewidth=1.3)
        ax_g.axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
        for tr in transitions:
            ax_g.axvline(tr, color="red", linewidth=0.9, linestyle=":", alpha=0.7)
        ax_g.set_xlim(-0.5, T - 0.5)
        ax_g.set_ylim(-0.05, 1.15)
        ax_g.set_ylabel("Gripper", fontsize=8)
        ax_g.set_xlabel("Step", fontsize=9)
        ax_g.tick_params(labelsize=7)

        instr = getattr(episode, "language_instruction", "") or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'Bridge v2 — "{instr[:90]}"', fontsize=11, y=1.002)

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 7. MP4 video export
    # ------------------------------------------------------------------

    def save_episode_video(
        self,
        episode: Any,
        output_path: Union[str, Path],
        fps: int = 10,
        camera: int = 0,
        max_frames: Optional[int] = 64,
    ) -> None:
        """
        Save the episode as an MP4 video from the specified camera.
        Requires: pip install imageio imageio-ffmpeg
        """
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "imageio is required for video export. "
                "pip install 'vla-data-curator[viz]'"
            ) from exc

        steps = list(episode.steps)
        if max_frames and len(steps) > max_frames:
            idxs = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in idxs]

        frames = []
        for step in steps:
            img = _img(step.observation) if camera == 0 else _img1(step.observation)
            if img is not None:
                frames.append(img.astype(np.uint8))

        if not frames:
            raise ValueError(f"No images found for camera {camera}.")

        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    # ------------------------------------------------------------------
    # 8. GIF export (primary camera)
    # ------------------------------------------------------------------

    def save_episode_gif(
        self,
        episode: Any,
        output_path: Union[str, Path],
        fps: int = 4,
        camera: int = 0,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Save a GIF of the episode from the specified camera (0 or 1).
        Requires: pip install imageio
        """
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "imageio is required for GIF export. "
                "pip install 'vla-data-curator[viz]'"
            ) from exc

        steps = list(episode.steps)
        if max_frames and len(steps) > max_frames:
            idxs = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in idxs]

        frames = []
        for step in steps:
            img = _img(step.observation) if camera == 0 else _img1(step.observation)
            if img is not None:
                frames.append(img.astype(np.uint8))

        if not frames:
            raise ValueError(f"No images found for camera {camera}.")

        imageio.mimsave(str(output_path), frames, fps=fps)
