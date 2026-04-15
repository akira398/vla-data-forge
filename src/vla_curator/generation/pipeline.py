"""
Reasoning-trace generation pipeline.

This is the top-level orchestrator for annotating episodes with structured
reasoning traces from a model backend.

Workflow
--------
For each episode in the source dataset:
  1. Sample key frames (FrameSamplingConfig).
  2. Build a prompt (ECoTPromptBuilder).
  3. Submit to the model backend (ModelBackend).
  4. Parse the response (ReasoningTraceParser).
  5. Post-process (TracePostprocessor).
  6. Write the annotated episode to output_dir (JSONLines).

Resume behaviour
----------------
The pipeline checks output_dir for already-annotated episode IDs before
processing.  Re-running with --resume=True only processes new episodes.

Dry-run mode
------------
Set ``config.dry_run = True`` to build prompts and log them without making
any API calls.  Useful for auditing the prompt structure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from tqdm import tqdm

from ..backends.base import ModelBackend
from ..config import GenerationConfig
from ..datasets.base import DatasetReader
from ..schemas.embodied_cot import ECoTEpisode, ECoTStep
from .prompt_builder import ECoTPromptBuilder
from .response_parser import ReasoningTraceParser
from .trace_postprocessor import TracePostprocessor

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Orchestrates reasoning-trace generation over a dataset.

    Parameters
    ----------
    config : GenerationConfig
    backend : ModelBackend
        Pre-instantiated backend.  Use ``create_backend(config.backend)`` to
        build one from config.
    reader : DatasetReader
        Pre-instantiated dataset reader.
    """

    def __init__(
        self,
        config: GenerationConfig,
        backend: ModelBackend,
        reader: DatasetReader,
    ) -> None:
        self.config = config
        self.backend = backend
        self.reader = reader

        self.prompt_builder = ECoTPromptBuilder(
            frame_sampling=config.frame_sampling
        )
        self.parser = ReasoningTraceParser()
        self.postprocessor = TracePostprocessor(propagation="nearest")

        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> List[ECoTEpisode]:
        """
        Process all episodes in the reader.

        Returns the list of annotated episodes (also written to disk).
        """
        done_ids = self._load_resume_set()
        episodes_iter = self.reader.__iter__()
        annotated: List[ECoTEpisode] = []

        total = len(self.reader)
        pbar = tqdm(total=total, desc="Generating traces", unit="ep")

        for episode in episodes_iter:
            if episode.episode_id in done_ids:
                logger.debug("Skipping already-processed episode: %s", episode.episode_id)
                pbar.update(1)
                continue

            try:
                annotated_ep = self.process_episode(episode)
                self._write_episode(annotated_ep)
                annotated.append(annotated_ep)
                done_ids.add(episode.episode_id)
            except Exception:
                logger.exception(
                    "Failed to process episode %s — skipping.", episode.episode_id
                )
            pbar.update(1)

        pbar.close()
        logger.info(
            "Generation complete. %d episodes annotated → %s",
            len(annotated),
            self.config.output_dir,
        )
        return annotated

    # ------------------------------------------------------------------
    # Per-episode processing
    # ------------------------------------------------------------------

    def process_episode(self, episode: ECoTEpisode) -> ECoTEpisode:
        """
        Generate reasoning traces for one episode.

        Returns a new ECoTEpisode with traces attached to each step.
        The original episode is not mutated.
        """
        prompt, frame_indices = self.prompt_builder.build_episode_prompt(episode)

        logger.debug(
            "Episode %s: %d frames selected from %d steps.",
            episode.episode_id,
            len(frame_indices),
            len(episode),
        )

        if self.config.dry_run:
            logger.info(
                "[DRY RUN] Episode %s — prompt text:\n%s",
                episode.episode_id,
                prompt.text[:500],
            )
            return episode

        # Call the model
        result = self.backend.generate(prompt)
        logger.debug(
            "Episode %s: model=%s, finish=%s, tokens=%s",
            episode.episode_id,
            result.model,
            result.finish_reason,
            result.usage.total_tokens if result.usage else "?",
        )

        # Parse
        annotated_pairs = self.parser.parse_episode_response(
            result.text, frame_indices
        )
        sparse_map = {idx: trace for idx, trace in annotated_pairs}

        # Post-process
        dense_traces = self.postprocessor.process_episode(sparse_map, len(episode))

        # Attach to steps
        new_steps: List[ECoTStep] = []
        for step, trace in zip(episode.steps, dense_traces):
            new_step = ECoTStep(
                step_index=step.step_index,
                observation=step.observation,
                action=step.action,
                reasoning=trace,
                is_first=step.is_first,
                is_last=step.is_last,
            )
            new_steps.append(new_step)

        annotated_episode = ECoTEpisode(
            episode_id=episode.episode_id,
            language_instruction=episode.language_instruction,
            steps=new_steps,
            metadata={
                **episode.metadata,
                "generation_model": result.model,
                "generation_backend": self.backend.provider,
                "frame_indices": frame_indices,
            },
            source_dataset=episode.source_dataset,
        )

        quality = self.postprocessor.validate(dense_traces)
        logger.info(
            "Episode %s annotated: coverage=%.0f%%, complete=%.0f%%",
            episode.episode_id,
            quality["coverage"] * 100,
            quality["completeness_rate"] * 100,
        )

        return annotated_episode

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _output_path(self) -> Path:
        return self.config.output_dir / "episodes.jsonl"

    def _write_episode(self, episode: ECoTEpisode) -> None:
        """Append one episode (as JSON) to the output file."""
        with open(self._output_path(), "a", encoding="utf-8") as f:
            d = episode.to_dict(include_images=False)
            if self.config.save_raw_responses:
                # Already embedded in ReasoningTrace.raw_response
                pass
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _load_resume_set(self) -> Set[str]:
        """Read already-processed episode IDs from the output file."""
        out = self._output_path()
        if not out.exists():
            return set()
        done: Set[str] = set()
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ep_id = obj.get("episode_id")
                    if ep_id:
                        done.add(ep_id)
                except json.JSONDecodeError:
                    pass
        logger.info("Resume: %d episodes already processed.", len(done))
        return done
