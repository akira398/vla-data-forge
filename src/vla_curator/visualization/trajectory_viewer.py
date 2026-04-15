"""
Trajectory and action-sequence visualization.

Shows robot action trajectories (xyz position, gripper state), reasoning
coverage summaries, and alignment confidence plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


class TrajectoryViewer:
    """
    Visualise action trajectories and reasoning coverage for any episode.

    Usage
    -----
    viewer = TrajectoryViewer()
    viewer.plot_actions(episode)
    viewer.plot_reasoning_coverage(episode)
    viewer.plot_summary(episode, save_path="ep_summary.png")
    """

    # ------------------------------------------------------------------
    # Action plot
    # ------------------------------------------------------------------

    def plot_actions(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Plot the 7-DoF action sequence over time.

        Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        if not steps:
            raise ValueError("Episode has no steps.")

        actions = np.stack([s.action for s in steps], axis=0)  # (T, 7)
        T = actions.shape[0]
        t = np.arange(T)

        labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#000000"]

        fig, axes = plt.subplots(7, 1, figsize=(10, 9), sharex=True)
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(t, actions[:, i], color=color, linewidth=1.2)
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step", fontsize=9)
        instr = getattr(episode, "language_instruction", None) or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'Actions — "{instr[:70]}"', fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # XYZ trajectory (3D)
    # ------------------------------------------------------------------

    def plot_trajectory_3d(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Plot cumulative XYZ end-effector path as a 3D line."""
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        actions = np.stack([s.action for s in steps], axis=0)
        xyz = np.cumsum(actions[:, :3], axis=0)  # Approximate EEF positions

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "b-", linewidth=1.5, alpha=0.8)
        ax.scatter(*xyz[0], color="green", s=50, zorder=5, label="Start")
        ax.scatter(*xyz[-1], color="red", s=50, zorder=5, label="End")
        ax.set_xlabel("ΔX (cumsum)")
        ax.set_ylabel("ΔY (cumsum)")
        ax.set_zlabel("ΔZ (cumsum)")
        ax.legend(fontsize=8)
        ax.set_title("Approximate EEF Trajectory", fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Reasoning coverage
    # ------------------------------------------------------------------

    def plot_reasoning_coverage(
        self,
        episode: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Display a heatmap of which steps have reasoning traces and which fields
        are populated.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        steps = list(episode.steps)
        fields = [
            "task_reasoning",
            "subtask_reasoning",
            "move_reasoning",
            "gripper_reasoning",
            "attribute_reasoning",
            "spatial_reasoning",
        ]

        # Build presence matrix: (len(fields), T)
        T = len(steps)
        matrix = np.zeros((len(fields), T), dtype=float)
        confidence = np.zeros(T, dtype=float)

        for j, step in enumerate(steps):
            confidence[j] = getattr(step, "alignment_confidence", 1.0)
            reasoning = getattr(step, "reasoning", None)
            if reasoning is None:
                continue
            for i, field in enumerate(fields):
                val = getattr(reasoning, field, None)
                matrix[i, j] = 1.0 if (val and val.strip()) else 0.0

        fig, (ax_heat, ax_conf) = plt.subplots(
            2, 1, figsize=(min(T * 0.3 + 2, 16), 5),
            gridspec_kw={"height_ratios": [6, 1]},
        )

        cmap = mcolors.ListedColormap(["#f0f0f0", "#2ca02c"])
        ax_heat.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax_heat.set_yticks(range(len(fields)))
        ax_heat.set_yticklabels([f.replace("_reasoning", "") for f in fields], fontsize=8)
        ax_heat.set_xlabel("Step index", fontsize=9)
        ax_heat.set_title("Reasoning field coverage per step", fontsize=10)

        ax_conf.bar(range(T), confidence, color="#1f77b4", alpha=0.7, width=1.0)
        ax_conf.set_xlim(-0.5, T - 0.5)
        ax_conf.set_ylim(0, 1.1)
        ax_conf.set_ylabel("Conf.", fontsize=7)
        ax_conf.tick_params(labelsize=7)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        episode: Any,
        max_frames: int = 6,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Composite 3-panel summary: sampled frames | action plot | coverage.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from .frame_viewer import FrameViewer

        steps = list(episode.steps)
        T = len(steps)

        # Subsample frames
        if T > max_frames:
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            frame_steps = [steps[i] for i in indices]
        else:
            frame_steps = steps

        from .frame_viewer import _load_img

        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(3, max_frames, figure=fig, hspace=0.5, wspace=0.3)

        # Row 0: frames
        for col, step in enumerate(frame_steps):
            ax = fig.add_subplot(gs[0, col])
            img = _load_img(step)
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Step {step.step_index}", fontsize=6)

        # Row 1: action signals (xyz only)
        ax_action = fig.add_subplot(gs[1, :])
        actions = np.stack([s.action for s in steps], axis=0)
        t = np.arange(T)
        for i, (label, color) in enumerate(
            zip(["Δx", "Δy", "Δz"], ["#e41a1c", "#377eb8", "#4daf4a"])
        ):
            ax_action.plot(t, actions[:, i], label=label, color=color, linewidth=1.2)
        ax_action.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_action.legend(fontsize=7, loc="upper right")
        ax_action.set_ylabel("Δ EEF", fontsize=8)
        ax_action.set_xlabel("Step", fontsize=8)
        ax_action.tick_params(labelsize=7)

        # Row 2: reasoning coverage bar
        ax_cov = fig.add_subplot(gs[2, :])
        coverage = np.array([
            1.0 if getattr(s, "reasoning", None) is not None else 0.0
            for s in steps
        ])
        ax_cov.bar(t, coverage, width=1.0, color="#ff7f0e", alpha=0.6)
        ax_cov.set_ylim(0, 1.2)
        ax_cov.set_ylabel("Has reasoning", fontsize=8)
        ax_cov.set_xlabel("Step", fontsize=8)
        ax_cov.tick_params(labelsize=7)

        instr = getattr(episode, "language_instruction", None) or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'Episode: "{instr[:90]}"', fontsize=11, y=1.01)

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        else:
            plt.show()
        return fig
