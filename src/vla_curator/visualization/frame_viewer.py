"""
Frame and episode visualization utilities.

All functions use matplotlib and are designed to work in both:
  - Jupyter notebooks (inline display via plt.show())
  - Scripts (save to file via save_path argument)
  - CLI (save PNG + open with system viewer)

Design: functions accept any episode type (ECoTEpisode, InterleavedEpisode,
BridgeEpisode) as long as it has a ``.steps`` attribute with steps that have
an ``.observation`` and optional ``.reasoning``.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np


def _load_img(step: Any) -> Optional[np.ndarray]:
    """Extract image from any step type via duck typing."""
    obs = getattr(step, "observation", None)
    if obs is None:
        return None
    load_fn = (
        getattr(obs, "load_image", None)
        or getattr(obs, "load_image_0", None)
        or getattr(obs, "primary_image", None)
    )
    if load_fn:
        return load_fn()
    return getattr(obs, "image", None)


def _get_reasoning_text(step: Any) -> Optional[str]:
    """Return a short reasoning summary for a step."""
    reasoning = getattr(step, "reasoning", None)
    if reasoning is None:
        return None
    parts = []
    if getattr(reasoning, "subtask_reasoning", None):
        parts.append(f"Subtask: {reasoning.subtask_reasoning}")
    if getattr(reasoning, "move_reasoning", None):
        parts.append(f"Move: {reasoning.move_reasoning}")
    if getattr(reasoning, "gripper_reasoning", None):
        parts.append(f"Gripper: {reasoning.gripper_reasoning}")
    return "\n".join(parts) if parts else None


class FrameViewer:
    """
    Visualise frame sequences and reasoning traces from any episode type.

    Usage
    -----
    viewer = FrameViewer()
    viewer.show_episode(episode, max_frames=8)
    viewer.show_episode(episode, save_path="output/ep_001.png")
    """

    def __init__(self, figsize_per_frame: tuple[int, int] = (3, 3)) -> None:
        self.figsize_per_frame = figsize_per_frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_episode(
        self,
        episode: Any,
        max_frames: Optional[int] = 12,
        show_reasoning: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Display a grid of frames from an episode with optional reasoning overlays.

        Returns the matplotlib Figure object for further customisation.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        if max_frames and len(steps) > max_frames:
            # Subsample evenly
            indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in indices]

        ncols = min(4, len(steps))
        nrows = math.ceil(len(steps) / ncols)
        fw, fh = self.figsize_per_frame
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fw * ncols, fh * nrows + (1.5 if show_reasoning else 0)),
        )
        axes = np.array(axes).flatten()

        for ax in axes:
            ax.axis("off")

        instr = getattr(episode, "language_instruction", None) or getattr(
            episode, "task_description", ""
        )
        suptitle = title or f'Episode: "{instr[:80]}"'
        fig.suptitle(suptitle, fontsize=10, wrap=True)

        for i, step in enumerate(steps):
            ax = axes[i]
            img = _load_img(step)

            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "No image", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            step_idx = getattr(step, "step_index", i)
            header = f"Step {step_idx}"
            if getattr(step, "is_first", False):
                header += " [START]"
            if getattr(step, "is_last", False):
                header += " [END]"
            ax.set_title(header, fontsize=7, pad=2)

            if show_reasoning:
                reason_text = _get_reasoning_text(step)
                if reason_text:
                    wrapped = "\n".join(
                        textwrap.wrap(reason_text, width=40)
                    )
                    ax.text(
                        0.01, -0.02, wrapped,
                        transform=ax.transAxes,
                        fontsize=5, va="top", color="navy",
                        wrap=True,
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8),
                    )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return fig

    def show_reasoning_trace(
        self,
        episode: Any,
        step_index: int,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Full-page view of one step: image on the left, all reasoning fields
        on the right.
        """
        import matplotlib.pyplot as plt

        steps = list(episode.steps)
        if step_index >= len(steps):
            raise IndexError(f"step_index {step_index} out of range for {len(steps)}-step episode.")

        step = steps[step_index]
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: image
        img = _load_img(step)
        if img is not None:
            ax_img.imshow(img)
        else:
            ax_img.set_facecolor("#dddddd")
            ax_img.text(0.5, 0.5, "No image", ha="center", va="center",
                        transform=ax_img.transAxes)
        ax_img.axis("off")
        ax_img.set_title(f"Step {step.step_index}", fontsize=12)

        # Right: reasoning
        ax_text.axis("off")
        reasoning = getattr(step, "reasoning", None)
        if reasoning is None:
            ax_text.text(0.1, 0.5, "No reasoning trace.", fontsize=11)
        else:
            fields = [
                ("Task", getattr(reasoning, "task_reasoning", None)),
                ("Subtask", getattr(reasoning, "subtask_reasoning", None)),
                ("Movement", getattr(reasoning, "move_reasoning", None)),
                ("Gripper", getattr(reasoning, "gripper_reasoning", None)),
                ("Attributes", getattr(reasoning, "attribute_reasoning", None)),
                ("Spatial", getattr(reasoning, "spatial_reasoning", None)),
            ]
            lines = []
            for label, val in fields:
                if val:
                    wrapped = textwrap.fill(val, width=60)
                    lines.append(f"[{label}]\n{wrapped}")
            text = "\n\n".join(lines) if lines else "No reasoning fields populated."
            ax_text.text(
                0.02, 0.98, text,
                transform=ax_text.transAxes,
                va="top", fontsize=9,
                family="monospace",
                wrap=True,
            )

        instr = getattr(episode, "language_instruction", None) or getattr(
            episode, "task_description", ""
        )
        fig.suptitle(f'Task: "{instr}"', fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return fig

    def save_episode_video(
        self,
        episode: Any,
        output_path: Union[str, Path],
        fps: int = 10,
        max_frames: Optional[int] = 64,
    ) -> None:
        """
        Save the episode as an MP4 video.
        Requires: pip install imageio imageio-ffmpeg
        """
        try:
            import imageio
        except ImportError as e:
            raise ImportError(
                "imageio is required for video export. "
                "pip install 'vla-data-curator[viz]'"
            ) from e

        steps = list(episode.steps)
        if max_frames and len(steps) > max_frames:
            indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in indices]

        frames = []
        for step in steps:
            img = _load_img(step)
            if img is not None:
                frames.append(img.astype(np.uint8))

        if not frames:
            raise ValueError("No images found in episode for video export.")

        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    def save_episode_gif(
        self,
        episode: Any,
        output_path: Union[str, Path],
        fps: int = 4,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Save the episode as an animated GIF.
        Requires: pip install imageio imageio-ffmpeg
        """
        try:
            import imageio
        except ImportError as e:
            raise ImportError(
                "imageio is required for GIF export. "
                "pip install 'vla-data-curator[viz]'"
            ) from e

        from PIL import Image as PILImage

        steps = list(episode.steps)
        if max_frames and len(steps) > max_frames:
            indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            steps = [steps[i] for i in indices]

        frames = []
        for step in steps:
            img = _load_img(step)
            if img is not None:
                frames.append(img.astype(np.uint8))

        if not frames:
            raise ValueError("No images found in episode for GIF export.")

        imageio.mimsave(str(output_path), frames, fps=fps)
