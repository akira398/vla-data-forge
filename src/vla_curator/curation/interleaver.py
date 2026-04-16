"""
Interleaving pipeline — merges ECoT reasoning traces with Bridge v2 data.

The ECoT dataset is derived from Bridge v2, so each ECoT episode has a
corresponding Bridge v2 episode.  The join key is the ``source_file`` path
stored in ``BridgeEpisode.source_file`` / ECoT episode metadata.

Step alignment
--------------
ECoT may annotate only key frames (sparse reasoning), while Bridge v2 has the
full step sequence.  The alignment strategy controls how sparse reasoning is
spread across the full Bridge v2 step sequence:

  EXACT     — Only steps with a direct trace keep it; others get reasoning=None.
  NEAREST   — Propagate the nearest annotated trace to unannotated steps.
  BROADCAST — Copy the first/only trace to all steps.

Alignment is handled by ``TracePostprocessor`` (same logic used during generation).

Output
------
Each merged pair becomes one ``InterleavedEpisode``.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import CurationConfig
from ..datasets.base import DatasetReader
from ..generation.trace_postprocessor import TracePostprocessor
from ..schemas.bridge_v2 import BridgeEpisode, BridgeObservation, BridgeStep
from ..schemas.embodied_cot import ECoTEpisode, ReasoningTrace
from ..schemas.interleaved import (
    AlignedStep,
    AlignmentMetadata,
    AlignmentStrategy,
    DataProvenance,
    EnrichedObservation,
    InterleavedEpisode,
)
from ..schemas.modalities import DepthMap, SceneGraph

# Type alias used in index dicts
_ECoTIndex = Dict[str, ECoTEpisode]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bridge_obs_to_enriched(obs: BridgeObservation) -> EnrichedObservation:
    """Convert a BridgeObservation to an EnrichedObservation."""
    return EnrichedObservation(
        step_index=obs.step_index,
        image=obs.image_0,
        image_path=obs.image_0_path,
        image_secondary=obs.image_1,
        image_secondary_path=obs.image_1_path,
        state=obs.state,
        depth_map=DepthMap(valid=False),
        scene_graph=SceneGraph(valid=False),
    )


def _normalize_path(path: str) -> str:
    """Strip leading slash and normalise backslashes for cross-platform safety."""
    return path.replace("\\", "/").lstrip("/")


def _make_composite_key(file_path: str, episode_num: int) -> str:
    """
    Build the composite join key for ECoT ↔ Bridge v2 matching.

    Format matches the original ECoT codebase (MichalZawalski/embodied-CoT):
        file_path + "_" + str(episode_id)

    Both sides (ECoT JSON and Bridge v2 TFDS) store the same absolute NFS
    path in ``file_path`` and a per-file integer ``episode_id``.
    """
    return f"{_normalize_path(file_path)}_{episode_num}"


def _normalize_episode_id(ep_id: str, source_file: Optional[str] = None) -> str:
    """
    Produce a normalised identifier for episode matching.

    This is the legacy path-only normalisation, kept as a fallback for
    episodes that lack an integer episode_id.
    """
    candidate = source_file or ep_id
    return _normalize_path(candidate)


# ---------------------------------------------------------------------------
# Episode index building
# ---------------------------------------------------------------------------


def _build_bridge_index(
    bridge_reader: DatasetReader[BridgeEpisode],
) -> Dict[str, BridgeEpisode]:
    """
    Load all Bridge v2 episodes into a dict keyed by composite key.

    Primary key: ``file_path + "_" + episode_id`` (composite).
    Fallback key: normalised ``file_path`` only (for legacy compatibility).
    """
    index: Dict[str, BridgeEpisode] = {}
    for ep in bridge_reader:
        # Primary: composite key
        if ep.source_file and ep.episode_num is not None:
            composite = _make_composite_key(ep.source_file, ep.episode_num)
            index[composite] = ep
        # Fallback: path-only key
        path_key = _normalize_episode_id(ep.episode_id, ep.source_file)
        index[path_key] = ep
    logger.info("Bridge index: %d episodes loaded.", len({id(v) for v in index.values()}))
    return index


# ---------------------------------------------------------------------------
# Main interleaver class
# ---------------------------------------------------------------------------


class EpisodeInterleaver:
    """
    Merges ECoT reasoning traces with Bridge v2 observations/actions to
    produce ``InterleavedEpisode`` objects.

    Usage
    -----
    from vla_curator.curation import EpisodeInterleaver
    from vla_curator.config import CurationConfig

    cfg = CurationConfig(...)
    interleaver = EpisodeInterleaver(cfg, ecot_reader, bridge_reader)

    for merged_ep in interleaver.iter_episodes():
        validator.validate_episode(merged_ep)
        exporter.export_episode(merged_ep)
    """

    def __init__(
        self,
        config: CurationConfig,
        ecot_reader: DatasetReader[ECoTEpisode],
        bridge_reader: DatasetReader[BridgeEpisode],
    ) -> None:
        self.config = config
        self.ecot_reader = ecot_reader
        self.bridge_reader = bridge_reader

        strategy = AlignmentStrategy(config.alignment_strategy)
        # Map strategy to postprocessor propagation arg
        propagation_map = {
            AlignmentStrategy.EXACT: "none",
            AlignmentStrategy.NEAREST: "nearest",
            AlignmentStrategy.BROADCAST: "broadcast",
        }
        self.postprocessor = TracePostprocessor(
            propagation=propagation_map[strategy], clean=True
        )
        self._bridge_index: Optional[Dict[str, BridgeEpisode]] = None

    @property
    def bridge_index(self) -> Dict[str, BridgeEpisode]:
        if self._bridge_index is None:
            logger.info("Building Bridge v2 episode index…")
            self._bridge_index = _build_bridge_index(self.bridge_reader)
        return self._bridge_index

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_episodes(self) -> Iterator[InterleavedEpisode]:
        """Yield merged episodes one at a time."""
        stats = {"matched": 0, "unmatched": 0, "errors": 0}

        for ecot_ep in self.ecot_reader:
            bridge_ep = self._find_bridge_episode(ecot_ep)

            if bridge_ep is None:
                logger.warning(
                    "No Bridge v2 match for ECoT episode %s — skipping.",
                    ecot_ep.episode_id,
                )
                stats["unmatched"] += 1
                continue

            try:
                merged = self.interleave(ecot_ep, bridge_ep)
                stats["matched"] += 1
                yield merged
            except Exception:
                logger.exception(
                    "Failed to interleave episode %s.", ecot_ep.episode_id
                )
                stats["errors"] += 1

        logger.info(
            "Interleaving complete: matched=%d, unmatched=%d, errors=%d",
            stats["matched"],
            stats["unmatched"],
            stats["errors"],
        )

    def interleave(
        self,
        ecot_ep: ECoTEpisode,
        bridge_ep: BridgeEpisode,
    ) -> InterleavedEpisode:
        """
        Merge one ECoT episode with its corresponding Bridge v2 episode.

        ECoT provides the reasoning traces; Bridge v2 provides observations
        and actions.  The Bridge v2 step count is authoritative for alignment.
        """
        # Build sparse reasoning map from ECoT (step_index → trace)
        sparse_reasoning: Dict[int, ReasoningTrace] = {}
        for step in ecot_ep.steps:
            if step.reasoning is not None:
                sparse_reasoning[step.step_index] = step.reasoning

        num_bridge_steps = len(bridge_ep)

        # Produce dense trace list aligned to Bridge v2 steps
        dense_traces = self.postprocessor.process_episode(
            sparse_reasoning, num_bridge_steps
        )

        # Build AlignedStep list
        aligned_steps: List[AlignedStep] = []
        for bridge_step, trace in zip(bridge_ep.steps, dense_traces):
            enriched_obs = _bridge_obs_to_enriched(bridge_step.observation)
            enriched_obs.step_index = bridge_step.step_index

            # Confidence: 1.0 if directly annotated, 0.7 if propagated
            directly_annotated = bridge_step.step_index in sparse_reasoning
            confidence = 1.0 if directly_annotated else (
                0.7 if trace is not None else 0.0
            )

            aligned = AlignedStep(
                step_index=bridge_step.step_index,
                observation=enriched_obs,
                action=bridge_step.action,
                reasoning=trace,
                is_first=bridge_step.is_first,
                is_last=bridge_step.is_last,
                source_dataset="bridge_v2",
                alignment_confidence=confidence,
            )
            aligned_steps.append(aligned)

        # Alignment metadata
        num_annotated = len(sparse_reasoning)
        num_aligned_with_trace = sum(1 for t in dense_traces if t is not None)
        coverage = num_aligned_with_trace / num_bridge_steps if num_bridge_steps else 0.0

        alignment_meta = AlignmentMetadata(
            strategy=self.config.alignment_strategy,
            ecot_episode_id=ecot_ep.episode_id,
            bridge_episode_id=bridge_ep.episode_id,
            num_steps_ecot=len(ecot_ep),
            num_steps_bridge=num_bridge_steps,
            num_aligned_steps=num_bridge_steps,
            num_annotated_steps=num_annotated,
            reasoning_coverage=coverage,
        )

        # Provenance
        gen_backend = ecot_ep.metadata.get("generation_backend")
        gen_model = ecot_ep.metadata.get("generation_model")
        provenance = DataProvenance(
            ecot_source=ecot_ep.source_dataset,
            bridge_source=bridge_ep.episode_id,
            generation_backend=gen_backend,
            generation_model=gen_model,
            curation_version=self.config.schema_version,
        )

        return InterleavedEpisode(
            episode_id=bridge_ep.episode_id,
            episode_num=bridge_ep.episode_num,
            task_description=(
                bridge_ep.language_instruction or ecot_ep.language_instruction
            ),
            steps=aligned_steps,
            alignment_metadata=alignment_meta,
            provenance=provenance,
            schema_version=self.config.schema_version,
        )

    # ------------------------------------------------------------------
    # Full-dataset iteration (all Bridge v2 episodes)
    # ------------------------------------------------------------------

    def iter_all_episodes(self) -> Iterator[InterleavedEpisode]:
        """
        Yield a merged episode for every Bridge v2 episode.

        Episodes that have an ECoT match are merged with reasoning traces
        (same as ``iter_episodes``).  Episodes with no ECoT match are
        wrapped with empty reasoning strings so every Bridge v2 trajectory
        ends up in the output.

        Use this to produce the *full* RLDS dataset variant.
        Use ``iter_episodes()`` when you only want matched episodes.
        """
        ecot_index = self._build_ecot_index()
        logger.info("ECoT index built: %d entries.", len(ecot_index))

        stats = {"matched": 0, "unmatched": 0, "errors": 0}

        for bridge_ep in self.bridge_reader:
            ecot_ep = None

            # Primary: composite key  file_path + "_" + episode_id
            if bridge_ep.source_file and bridge_ep.episode_num is not None:
                key = _make_composite_key(bridge_ep.source_file, bridge_ep.episode_num)
                ecot_ep = ecot_index.get(key)

            if ecot_ep is not None:
                try:
                    yield self.interleave(ecot_ep, bridge_ep)
                    stats["matched"] += 1
                except Exception:
                    logger.exception(
                        "Failed to interleave %s — falling back to empty reasoning.",
                        bridge_ep.episode_id,
                    )
                    yield self._bridge_ep_empty(bridge_ep)
                    stats["errors"] += 1
            else:
                yield self._bridge_ep_empty(bridge_ep)
                stats["unmatched"] += 1

        logger.info(
            "iter_all_episodes complete: matched=%d, unmatched=%d, errors=%d",
            stats["matched"],
            stats["unmatched"],
            stats["errors"],
        )

    def _build_ecot_index(self) -> _ECoTIndex:
        """
        Build an in-memory lookup table of ECoT episodes keyed by the
        normalised composite key ``file_path + "_" + episode_id``.

        This matches the original ECoT codebase key format used during
        both annotation generation and training-time lookup.

        The composite key is built in :func:`_parse_entry` and stored as
        ``ECoTEpisode.episode_id``; here we just normalise the path
        component (strip leading slash, normalise backslashes).
        """
        index: _ECoTIndex = {}
        for ep in self.ecot_reader:
            # ep.episode_id is already "file_path_episode_id" from _parse_entry
            norm_key = _normalize_path(ep.episode_id)
            index[norm_key] = ep
        return index

    def _bridge_ep_empty(self, bridge_ep: BridgeEpisode) -> InterleavedEpisode:
        """
        Wrap a Bridge v2 episode as an InterleavedEpisode with empty reasoning.

        The episode ID is preserved exactly as-is so loading code can always
        trace back to the original Bridge v2 trajectory.
        """
        aligned_steps: List[AlignedStep] = []
        for step in bridge_ep.steps:
            enriched_obs = _bridge_obs_to_enriched(step.observation)
            enriched_obs.step_index = step.step_index
            aligned_steps.append(AlignedStep(
                step_index=step.step_index,
                observation=enriched_obs,
                action=step.action,
                reasoning=None,
                is_first=step.is_first,
                is_last=step.is_last,
                source_dataset="bridge_v2",
                alignment_confidence=0.0,
            ))

        alignment_meta = AlignmentMetadata(
            strategy="none",
            ecot_episode_id="",
            bridge_episode_id=bridge_ep.episode_id,
            num_steps_ecot=0,
            num_steps_bridge=len(bridge_ep),
            num_aligned_steps=len(bridge_ep),
            num_annotated_steps=0,
            reasoning_coverage=0.0,
        )

        return InterleavedEpisode(
            episode_id=bridge_ep.episode_id,   # original path preserved
            episode_num=bridge_ep.episode_num,
            task_description=bridge_ep.language_instruction or "",
            steps=aligned_steps,
            alignment_metadata=alignment_meta,
            provenance=DataProvenance(
                ecot_source="",
                bridge_source=bridge_ep.episode_id,
                curation_version=self.config.schema_version,
            ),
            schema_version=self.config.schema_version,
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _find_bridge_episode(
        self, ecot_ep: ECoTEpisode
    ) -> Optional[BridgeEpisode]:
        """
        Look up the Bridge v2 episode corresponding to an ECoT episode.

        Primary lookup: composite key ``file_path + "_" + episode_id``
        (normalised).  Falls back to path-only matching for legacy data.
        """
        idx = self.bridge_index

        # Primary: composite key (ecot_ep.episode_id is "file_path_episode_id")
        composite_key = _normalize_path(ecot_ep.episode_id)
        if composite_key in idx:
            return idx[composite_key]

        # Fallback: path-only match via metadata file_path
        source_file = ecot_ep.metadata.get("file_path")
        if source_file:
            path_key = _normalize_path(str(source_file))
            if path_key in idx:
                return idx[path_key]

        return None
