"""
Dataset validation — structural and semantic checks on curated episodes.

The validator is intentionally strict about structure (always checked) and
permissive about content (checked with warnings, not hard errors).

Structural checks (raise or flag error):
  - Episode has at least one step
  - All actions are 7-dimensional
  - step_index values are sequential and zero-based
  - is_first/is_last are set exactly once at the correct positions
  - schema_version is present

Content checks (flag warnings):
  - Reasoning coverage (warn if < threshold)
  - Actions that are all-zeros for many steps (possible data corruption)
  - Missing task description
  - Images present (warn if none)

Usage
-----
validator = DatasetValidator()
result = validator.validate_episode(episode)
if not result.passed:
    print(result.errors)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..schemas.interleaved import InterleavedEpisode

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating one episode."""

    episode_id: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({len(self.errors)} errors)"
        return (
            f"ValidationResult(episode={self.episode_id!r}, "
            f"status={status}, warnings={len(self.warnings)})"
        )


@dataclass
class DatasetValidationReport:
    """Aggregate validation report for a full dataset."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    results: List[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        self.total += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"Validation: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.1%}), {self.failed} failed."
        )

    def failed_episodes(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed]


class DatasetValidator:
    """
    Runs structural and content checks on ``InterleavedEpisode`` objects.

    Parameters
    ----------
    min_steps : int
        Minimum number of steps required in a valid episode.
    min_reasoning_coverage : float
        Warning threshold: warn if reasoning_coverage < this value.
    require_images : bool
        Emit a warning (not an error) if no images are present.
    """

    def __init__(
        self,
        min_steps: int = 1,
        min_reasoning_coverage: float = 0.0,
        require_images: bool = False,
    ) -> None:
        self.min_steps = min_steps
        self.min_reasoning_coverage = min_reasoning_coverage
        self.require_images = require_images

    def validate_episode(self, episode: InterleavedEpisode) -> ValidationResult:
        result = ValidationResult(episode_id=episode.episode_id)

        # --- Structural checks ---

        if not episode.episode_id:
            result.add_error("episode_id is empty.")

        if not episode.schema_version:
            result.add_warning("schema_version is not set.")

        if len(episode) < self.min_steps:
            result.add_error(
                f"Episode has {len(episode)} steps; minimum is {self.min_steps}."
            )

        self._check_steps(episode, result)
        self._check_metadata(episode, result)

        # --- Content checks ---

        if not episode.task_description:
            result.add_warning("task_description is empty.")

        coverage = episode.reasoning_coverage()
        result.stats["reasoning_coverage"] = coverage
        if coverage < self.min_reasoning_coverage:
            result.add_warning(
                f"Reasoning coverage {coverage:.0%} is below threshold "
                f"{self.min_reasoning_coverage:.0%}."
            )

        if self.require_images:
            has_image = any(
                s.observation.load_image() is not None or s.observation.image_path
                for s in episode.steps
            )
            if not has_image:
                result.add_warning("No images found in any step.")

        return result

    def validate_dataset(
        self, episodes: List[InterleavedEpisode]
    ) -> DatasetValidationReport:
        report = DatasetValidationReport()
        for ep in episodes:
            r = self.validate_episode(ep)
            report.add(r)
            if not r.passed:
                logger.warning("Episode %s failed validation: %s", ep.episode_id, r.errors)
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_steps(
        self, episode: InterleavedEpisode, result: ValidationResult
    ) -> None:
        if not episode.steps:
            return  # Already caught by min_steps check

        # step_index is sequential
        for i, step in enumerate(episode.steps):
            if step.step_index != i:
                result.add_error(
                    f"step_index mismatch at position {i}: "
                    f"expected {i}, got {step.step_index}."
                )

        # is_first / is_last
        first_steps = [s for s in episode.steps if s.is_first]
        last_steps = [s for s in episode.steps if s.is_last]

        if len(first_steps) != 1:
            result.add_error(
                f"Expected exactly 1 is_first step, found {len(first_steps)}."
            )
        elif first_steps[0].step_index != 0:
            result.add_error("is_first is not on step 0.")

        if len(last_steps) != 1:
            result.add_error(
                f"Expected exactly 1 is_last step, found {len(last_steps)}."
            )
        elif last_steps[0].step_index != len(episode) - 1:
            result.add_error(
                f"is_last is not on the last step "
                f"(step {len(episode) - 1})."
            )

        # Action dimensions
        for step in episode.steps:
            if step.action.shape != (7,):
                result.add_error(
                    f"Step {step.step_index}: action shape is "
                    f"{step.action.shape}, expected (7,)."
                )

        # All-zero action warning (common in broken datasets)
        zero_actions = sum(
            1 for s in episode.steps if np.allclose(s.action, 0.0)
        )
        frac_zero = zero_actions / len(episode)
        result.stats["zero_action_fraction"] = frac_zero
        if frac_zero > 0.5:
            result.add_warning(
                f"{frac_zero:.0%} of actions are all-zeros — possible data issue."
            )

    def _check_metadata(
        self, episode: InterleavedEpisode, result: ValidationResult
    ) -> None:
        meta = episode.alignment_metadata
        if meta.num_steps_bridge > 0 and len(episode) != meta.num_steps_bridge:
            result.add_warning(
                f"Episode step count ({len(episode)}) differs from "
                f"alignment_metadata.num_steps_bridge ({meta.num_steps_bridge})."
            )
        if meta.strategy not in ("exact", "nearest", "broadcast"):
            result.add_warning(
                f"Unknown alignment strategy: {meta.strategy!r}."
            )
