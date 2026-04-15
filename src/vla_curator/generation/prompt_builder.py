"""
Prompt construction for Embodied-CoT reasoning-trace generation.

This module translates an episode + sampled frames into a ``Prompt`` object
that can be submitted to any ``ModelBackend``.

Design
------
The prompt structure closely follows the Embodied-CoT paper:
  1. A system prompt establishing the model's role as a robot-action analyst.
  2. The task instruction from the episode.
  3. N sampled frames presented in order.
  4. A structured question asking for the six reasoning dimensions.
  5. A JSON schema description so the model returns parseable output.

``ECoTPromptBuilder`` is the default implementation.  To experiment with
different prompting strategies (chain-of-thought preamble, few-shot examples,
etc.), subclass it and override ``build_episode_prompt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..backends.base import Prompt, PromptImage
from ..config import FrameSamplingConfig
from ..schemas.embodied_cot import ECoTEpisode, ECoTStep


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert robotics researcher analysing robot manipulation demonstrations.
Your task is to annotate each robot action with structured reasoning that explains
*why* the robot performs the observed motion.

You will be shown a sequence of images from a robot episode along with the task
description.  For each image (numbered from the first frame), provide the following
six reasoning dimensions in valid JSON.

JSON schema (one object per annotated frame):
[
  {
    "frame_index": <integer>,
    "task_reasoning": "<What is the overall task the robot is performing?>",
    "subtask_reasoning": "<What specific sub-goal is being addressed in this frame?>",
    "move_reasoning": "<What arm motion should occur and why? Describe direction, magnitude, and target.>",
    "gripper_reasoning": "<Should the gripper open or close? Why?>",
    "attribute_reasoning": "<What object attributes (colour, shape, size, material) matter for this action?>",
    "spatial_reasoning": "<Describe the spatial relationships between the end-effector and key objects.>"
  },
  ...
]

Rules:
- Output ONLY valid JSON.  Do not include markdown fences, preamble, or commentary.
- Provide one entry per frame you received.
- Keep each field concise (1–3 sentences).
- If uncertain, give your best estimate rather than leaving a field blank.
"""

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def sample_frames_uniform(steps: List[ECoTStep], n: int) -> List[int]:
    """Return n evenly-spaced step indices from the episode."""
    T = len(steps)
    if T == 0:
        return []
    if T <= n:
        return list(range(T))
    step = (T - 1) / (n - 1) if n > 1 else 0
    return [round(i * step) for i in range(n)]


def sample_frames_keyframe(
    steps: List[ECoTStep],
    threshold: float = 0.05,
    max_frames: int = 16,
) -> List[int]:
    """
    Return indices at action-transition frames (large gripper state change or
    large Euclidean delta in the xyz components).

    Always includes the first and last frame.
    """
    if not steps:
        return []
    indices = {0, len(steps) - 1}
    for i in range(1, len(steps)):
        prev = steps[i - 1].action
        curr = steps[i].action
        delta = abs(float(curr[6]) - float(prev[6]))          # gripper change
        xyz_delta = float(((curr[:3] - prev[:3]) ** 2).sum() ** 0.5)
        if delta > threshold or xyz_delta > threshold:
            indices.add(i)
    selected = sorted(indices)
    # Downsample if too many
    if len(selected) > max_frames:
        step = len(selected) / max_frames
        selected = [selected[round(i * step)] for i in range(max_frames)]
    return selected


def sample_frames_all(steps: List[ECoTStep]) -> List[int]:
    return list(range(len(steps)))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


@dataclass
class ECoTPromptBuilder:
    """
    Builds ``Prompt`` objects for reasoning-trace generation.

    Parameters
    ----------
    frame_sampling : FrameSamplingConfig
        Controls which frames are included.
    system_prompt : str
        Override the default system prompt if you want to experiment.
    jpeg_quality : int
        JPEG compression quality for frame encoding (lower = faster/smaller).
    """

    frame_sampling: FrameSamplingConfig = field(
        default_factory=FrameSamplingConfig
    )
    system_prompt: str = SYSTEM_PROMPT
    jpeg_quality: int = 85

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_episode_prompt(self, episode: ECoTEpisode) -> tuple[Prompt, List[int]]:
        """
        Build a prompt for the full episode.

        Returns
        -------
        prompt : Prompt
            Ready to submit to a ModelBackend.
        frame_indices : List[int]
            Which step indices were included (needed to align the response
            back to steps).
        """
        frame_indices = self._select_frames(episode)
        prompt_images = self._encode_frames(episode, frame_indices)

        # Build the text body
        text_parts = [
            f'Task: "{episode.language_instruction}"',
            "",
            f"Episode has {len(episode)} total steps. "
            f"You are provided {len(frame_indices)} sampled frames "
            f"(frame indices: {frame_indices}).",
            "",
            "Annotate each frame following the JSON schema in the system prompt.",
        ]

        return (
            Prompt(
                text="\n".join(text_parts),
                images=prompt_images,
                system_prompt=self.system_prompt,
                metadata={
                    "episode_id": episode.episode_id,
                    "frame_indices": frame_indices,
                    "num_total_steps": len(episode),
                },
            ),
            frame_indices,
        )

    def build_step_prompt(
        self,
        step: ECoTStep,
        task_description: str,
        context_steps: Optional[List[ECoTStep]] = None,
    ) -> Prompt:
        """
        Build a prompt for a *single* step, optionally with preceding context frames.

        Use this when you want per-step annotation rather than one batch call
        per episode (more API calls but easier alignment).
        """
        images: List[PromptImage] = []

        if context_steps:
            for ctx in context_steps[-3:]:  # last 3 context frames at most
                img = ctx.observation.load_image()
                if img is not None:
                    images.append(
                        PromptImage.from_numpy(img).with_quality(self.jpeg_quality)
                        if hasattr(PromptImage, "with_quality")
                        else PromptImage.from_numpy(img)
                    )

        # Current frame
        img = step.observation.load_image()
        if img is not None:
            images.append(PromptImage.from_numpy(img))

        text = (
            f'Task: "{task_description}"\n'
            f"Step index: {step.step_index}\n\n"
            "Annotate this single frame. Respond with a single JSON object "
            "(not an array) using the schema from the system prompt (omit frame_index)."
        )

        return Prompt(
            text=text,
            images=images,
            system_prompt=self.system_prompt,
            metadata={"episode_step": step.step_index},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_frames(self, episode: ECoTEpisode) -> List[int]:
        strategy = self.frame_sampling.strategy
        steps = episode.steps

        if strategy == "uniform":
            return sample_frames_uniform(steps, self.frame_sampling.num_frames)
        elif strategy == "keyframe":
            return sample_frames_keyframe(
                steps,
                threshold=self.frame_sampling.keyframe_threshold,
                max_frames=self.frame_sampling.num_frames,
            )
        elif strategy == "all":
            return sample_frames_all(steps)
        else:
            raise ValueError(f"Unknown frame sampling strategy: {strategy!r}")

    def _encode_frames(
        self, episode: ECoTEpisode, indices: List[int]
    ) -> List[PromptImage]:
        images = []
        for idx in indices:
            if idx >= len(episode):
                continue
            step = episode[idx]
            arr = step.observation.load_image()
            if arr is None:
                continue
            images.append(PromptImage.from_numpy(arr))
        return images
