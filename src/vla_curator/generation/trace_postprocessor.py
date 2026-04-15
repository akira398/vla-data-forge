"""
Post-processing for generated reasoning traces.

After parsing, traces may have:
  - Incomplete fields (partial generation, safety blocks)
  - Placeholder text like "N/A" or "Unknown"
  - Excessive verbosity (truncation may help)
  - No coverage for intermediate steps between annotated key-frames

This module provides a ``TracePostprocessor`` that cleans and extends traces.

Design
------
- Cleaning is cheap and always applied.
- Propagation (filling unannotated steps) is strategy-driven.
- All operations are pure functions over lists of Optional[ReasoningTrace],
  making them easy to unit test.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional

from ..schemas.embodied_cot import ReasoningTrace


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

# Strings that indicate a field has no real content
_EMPTY_PATTERNS = re.compile(
    r"^(n/?a|unknown|none|not applicable|no information|"
    r"not enough information|unclear|unspecified|\.+|-)$",
    re.IGNORECASE,
)


def _clean_field(text: Optional[str]) -> Optional[str]:
    """Return None if the text is a placeholder / empty, otherwise strip."""
    if text is None:
        return None
    text = text.strip()
    if not text or _EMPTY_PATTERNS.match(text):
        return None
    return text


def clean_trace(trace: ReasoningTrace) -> ReasoningTrace:
    """
    Return a new ``ReasoningTrace`` with placeholder values removed.
    Does not modify the input (dataclasses are mutable but we create a new one).
    """
    return ReasoningTrace(
        task_reasoning=_clean_field(trace.task_reasoning),
        subtask_reasoning=_clean_field(trace.subtask_reasoning),
        move_reasoning=_clean_field(trace.move_reasoning),
        gripper_reasoning=_clean_field(trace.gripper_reasoning),
        attribute_reasoning=_clean_field(trace.attribute_reasoning),
        spatial_reasoning=_clean_field(trace.spatial_reasoning),
        raw_response=trace.raw_response,
        extra=trace.extra,
    )


# ---------------------------------------------------------------------------
# Propagation strategies
# ---------------------------------------------------------------------------


def propagate_nearest(
    traces: List[Optional[ReasoningTrace]],
) -> List[Optional[ReasoningTrace]]:
    """
    Fill None entries with the nearest non-None trace.

    First fills forward (copy from the previous annotated step), then
    backward (copy from the next annotated step for initial gaps).
    Steps between two annotated steps get the earlier one's trace.
    """
    result: List[Optional[ReasoningTrace]] = list(traces)
    n = len(result)

    # Forward pass
    last_trace: Optional[ReasoningTrace] = None
    for i in range(n):
        if result[i] is not None:
            last_trace = result[i]
        elif last_trace is not None:
            result[i] = last_trace

    # Backward pass (for leading Nones)
    last_trace = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last_trace = result[i]
        elif last_trace is not None:
            result[i] = last_trace

    return result


def propagate_broadcast(
    traces: List[Optional[ReasoningTrace]],
) -> List[Optional[ReasoningTrace]]:
    """
    Find the first non-None trace and copy it to all positions.
    Used for episode-level annotations.
    """
    first_trace = next((t for t in traces if t is not None), None)
    if first_trace is None:
        return traces
    return [first_trace] * len(traces)


# ---------------------------------------------------------------------------
# Main postprocessor class
# ---------------------------------------------------------------------------


class TracePostprocessor:
    """
    Applies cleaning and propagation to a list of per-step traces.

    Usage
    -----
    postprocessor = TracePostprocessor(propagation="nearest")

    # ``sparse_traces``: list of Optional[ReasoningTrace], one per step.
    # Annotated steps have a trace; unannotated steps have None.
    filled_traces = postprocessor.process(sparse_traces)
    """

    def __init__(
        self,
        propagation: str = "nearest",
        clean: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        propagation : str
            One of: 'none', 'nearest', 'broadcast'.
        clean : bool
            Whether to strip placeholder values from all traces.
        """
        if propagation not in ("none", "nearest", "broadcast"):
            raise ValueError(f"Unknown propagation strategy: {propagation!r}")
        self.propagation = propagation
        self.clean = clean

    def process(
        self, traces: List[Optional[ReasoningTrace]]
    ) -> List[Optional[ReasoningTrace]]:
        """
        Clean and propagate a list of sparse traces.

        Parameters
        ----------
        traces : List[Optional[ReasoningTrace]]
            One trace per step; None for steps without annotation.

        Returns
        -------
        List[Optional[ReasoningTrace]]
            Same length, with None gaps filled according to strategy.
        """
        # Step 1: clean
        if self.clean:
            traces = [clean_trace(t) if t is not None else None for t in traces]

        # Step 2: propagate
        if self.propagation == "nearest":
            traces = propagate_nearest(traces)
        elif self.propagation == "broadcast":
            traces = propagate_broadcast(traces)
        # 'none': no propagation

        return traces

    def process_episode(
        self,
        sparse_map: dict[int, ReasoningTrace],
        num_steps: int,
    ) -> List[Optional[ReasoningTrace]]:
        """
        Convert a {step_index: trace} dict to a dense list of length ``num_steps``.

        Useful when the parser returns only annotated frames rather than a full list.
        """
        traces: List[Optional[ReasoningTrace]] = [None] * num_steps
        for idx, trace in sparse_map.items():
            if 0 <= idx < num_steps:
                traces[idx] = trace
        return self.process(traces)

    def coverage(self, traces: List[Optional[ReasoningTrace]]) -> float:
        """Return the fraction of steps with a non-None trace."""
        if not traces:
            return 0.0
        return sum(1 for t in traces if t is not None) / len(traces)

    def validate(self, traces: List[Optional[ReasoningTrace]]) -> dict:
        """
        Return a brief quality report for a processed trace list.

        Returns a dict with keys: total, annotated, coverage, complete.
        """
        total = len(traces)
        annotated = sum(1 for t in traces if t is not None)
        complete = sum(1 for t in traces if t is not None and t.is_complete())
        return {
            "total_steps": total,
            "annotated_steps": annotated,
            "coverage": annotated / total if total else 0.0,
            "complete_traces": complete,
            "completeness_rate": complete / annotated if annotated else 0.0,
        }
