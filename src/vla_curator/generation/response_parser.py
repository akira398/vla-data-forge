"""
Response parser — converts raw model output to ``ReasoningTrace`` objects.

The generation pipeline expects the model to return valid JSON matching the
schema defined in the system prompt.  In practice, models sometimes:
  - Add markdown fences (```json ... ```)
  - Return a single object instead of an array
  - Partially fill fields
  - Include trailing commas or other minor JSON errors

The parser handles all of these gracefully.  It tries strict JSON parsing
first, then progressively more lenient approaches, and finally falls back to
a partial trace with ``raw_response`` preserved for manual inspection.

Key function: ``parse_episode_response(text, frame_indices) -> List[(int, ReasoningTrace)]``
Returns a list of (step_index, trace) pairs corresponding to the frames that
were annotated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.embodied_cot import ReasoningTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } (common LLM mistake)."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_block(text: str) -> Optional[str]:
    """
    Attempt to extract the first JSON array or object from ``text``.
    Returns None if no JSON structure is found.
    """
    # Try to find a JSON array first, then a JSON object
    for pattern in (
        r"(\[[\s\S]*?\])",      # array
        r"(\{[\s\S]*?\})",      # object
    ):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def _try_parse_json(text: str) -> Optional[Any]:
    """Try several parsing strategies, returning parsed object or None."""
    # Strategy 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = _strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: fix trailing commas
    fixed = _fix_trailing_commas(cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: extract first JSON block
    block = _extract_json_block(text)
    if block:
        block = _fix_trailing_commas(block)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Dict → ReasoningTrace
# ---------------------------------------------------------------------------


_FIELD_ALIASES: Dict[str, List[str]] = {
    "task_reasoning":      ["task_reasoning", "task", "overall_task", "goal"],
    "subtask_reasoning":   ["subtask_reasoning", "subtask", "current_goal", "sub_task"],
    "move_reasoning":      ["move_reasoning", "movement", "motion", "arm_movement"],
    "gripper_reasoning":   ["gripper_reasoning", "gripper", "gripper_action"],
    "attribute_reasoning": ["attribute_reasoning", "attributes", "object_attributes"],
    "spatial_reasoning":   ["spatial_reasoning", "spatial", "spatial_relationships"],
}


def _get_field(d: Dict[str, Any], field_name: str) -> Optional[str]:
    """Extract a field from a dict, trying aliases."""
    for alias in _FIELD_ALIASES.get(field_name, [field_name]):
        val = d.get(alias)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _dict_to_trace(d: Dict[str, Any], raw_response: str = "") -> ReasoningTrace:
    return ReasoningTrace(
        task_reasoning=_get_field(d, "task_reasoning"),
        subtask_reasoning=_get_field(d, "subtask_reasoning"),
        move_reasoning=_get_field(d, "move_reasoning"),
        gripper_reasoning=_get_field(d, "gripper_reasoning"),
        attribute_reasoning=_get_field(d, "attribute_reasoning"),
        spatial_reasoning=_get_field(d, "spatial_reasoning"),
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------


class ReasoningTraceParser:
    """
    Parses raw model output into structured ``ReasoningTrace`` objects.

    Usage
    -----
    parser = ReasoningTraceParser()

    # For a full-episode response (list of per-frame annotations)
    traces = parser.parse_episode_response(result.text, frame_indices=[0, 5, 10])

    # For a single-step response
    trace = parser.parse_step_response(result.text)
    """

    def parse_episode_response(
        self,
        text: str,
        frame_indices: List[int],
    ) -> List[Tuple[int, ReasoningTrace]]:
        """
        Parse a response that covers multiple frames.

        Returns a list of (step_index, ReasoningTrace) pairs in the same
        order as ``frame_indices``.  If fewer items are in the response than
        expected, the remaining entries get empty traces with the raw response.
        """
        parsed = _try_parse_json(text)
        results: List[Tuple[int, ReasoningTrace]] = []

        if parsed is None:
            logger.warning(
                "Could not parse model response as JSON. "
                "Storing raw response for manual review."
            )
            for idx in frame_indices:
                results.append((idx, ReasoningTrace(raw_response=text)))
            return results

        # Normalise to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            logger.warning("Unexpected JSON type: %s. Expected list.", type(parsed))
            for idx in frame_indices:
                results.append((idx, ReasoningTrace(raw_response=text)))
            return results

        # Map parsed items to frame_indices
        # The model may include a frame_index field; if so, use it.
        # Otherwise, assume positional correspondence.
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            # Determine which step this corresponds to
            model_frame_idx = item.get("frame_index")
            if model_frame_idx is not None and int(model_frame_idx) in frame_indices:
                step_idx = int(model_frame_idx)
            elif i < len(frame_indices):
                step_idx = frame_indices[i]
            else:
                logger.debug("Excess item at position %d — skipping.", i)
                continue

            trace = _dict_to_trace(item, raw_response=text)
            results.append((step_idx, trace))

        # Fill in any frame_indices that got no annotation
        annotated_indices = {idx for idx, _ in results}
        for idx in frame_indices:
            if idx not in annotated_indices:
                logger.debug("Frame %d was not annotated in response.", idx)
                results.append((idx, ReasoningTrace(raw_response=text)))

        # Sort by step index
        results.sort(key=lambda x: x[0])
        return results

    def parse_step_response(self, text: str) -> ReasoningTrace:
        """
        Parse a response for a single step.

        Expects a JSON object (not array).  Handles array responses by using
        the first element.
        """
        parsed = _try_parse_json(text)

        if parsed is None:
            logger.warning("Could not parse step response as JSON.")
            return ReasoningTrace(raw_response=text)

        if isinstance(parsed, list):
            if not parsed:
                return ReasoningTrace(raw_response=text)
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            logger.warning("Unexpected parsed type: %s", type(parsed))
            return ReasoningTrace(raw_response=text)

        return _dict_to_trace(parsed, raw_response=text)
