"""
RLDS/TFDS exporter — writes Bridge v2 compatible TFRecord datasets.

Produces two variants under output_dir:

  output_dir/vla_curated_dataset/matched/1.0.0/
      — All ECoT-matched Bridge v2 episodes; episodes where ECoT has no
        reasoning annotations get empty reasoning strings.

  output_dir/vla_curated_dataset/reasoning_only/1.0.0/
      — Only episodes that have at least one non-empty reasoning annotation.

Both directories are loadable with ``tfds.builder_from_directory()``:

    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(
        "output_dir/vla_curated_dataset/matched/1.0.0/"
    )
    ds = builder.as_dataset(split="train")
    for ep in ds:
        for step in ep["steps"]:
            print(step["observation"]["task_reasoning"])

Feature spec (Bridge v2 field names preserved, reasoning fields added):

  episode_metadata/file_path          bytes   original Bridge v2 path
  episode_metadata/episode_id         int32   Bridge v2 per-file episode ID
  steps/observation/image_0           uint8   (480, 640, 3) JPEG
  steps/observation/image_1           uint8   (480, 640, 3) JPEG
  steps/observation/state             float32 (7,)
  steps/observation/task_reasoning    bytes   empty string when no ECoT match
  steps/observation/subtask_reasoning bytes
  steps/observation/move_reasoning    bytes
  steps/language_instruction          bytes   (at step level, same as Bridge v2)
  steps/action                        float32 (7,)
  steps/is_first                      bool
  steps/is_last                       bool
  steps/is_terminal                   bool
  steps/reward                        float32
  steps/discount                      float32
  steps/alignment_confidence          float32 1.0=direct, 0.7=propagated, 0.0=none
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..schemas.interleaved import InterleavedEpisode
from .export import BaseExporter

logger = logging.getLogger(__name__)

# Blank arrays used when an episode has no image / state data
_BLANK_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)
_BLANK_STATE = np.zeros(7, dtype=np.float32)
_BLANK_ACTION = np.zeros(7, dtype=np.float32)

# ---------------------------------------------------------------------------
# Module-level episode buffer — populated by RLDSExporter before the TFDS
# builder's download_and_prepare() runs.
# ---------------------------------------------------------------------------
_EPISODE_BUFFER: Dict[str, List[InterleavedEpisode]] = {}
_BUFFER_SPLITS: List[str] = []


# ---------------------------------------------------------------------------
# Feature spec
# ---------------------------------------------------------------------------


def _make_feature_spec():
    """Build the TFDS feature spec (Bridge v2 compatible + reasoning fields)."""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow and tensorflow-datasets are required for RLDS export.\n"
            "Install with:  pip install 'vla-data-curator[bridge]'"
        ) from exc

    return tfds.features.FeaturesDict({
        "episode_metadata": tfds.features.FeaturesDict({
            "file_path": tfds.features.Text(),
            "episode_id": tfds.features.Scalar(dtype=tf.int32),
        }),
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "image_0": tfds.features.Image(
                    shape=(480, 640, 3), dtype=tf.uint8, encoding_format="jpeg"
                ),
                "image_1": tfds.features.Image(
                    shape=(480, 640, 3), dtype=tf.uint8, encoding_format="jpeg"
                ),
                "state":                tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                "task_reasoning":       tfds.features.Text(),
                "subtask_reasoning":    tfds.features.Text(),
                "move_reasoning":       tfds.features.Text(),
            }),
            "language_instruction": tfds.features.Text(),
            "action":               tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "is_first":             tfds.features.Scalar(dtype=tf.bool),
            "is_last":              tfds.features.Scalar(dtype=tf.bool),
            "is_terminal":          tfds.features.Scalar(dtype=tf.bool),
            "reward":               tfds.features.Scalar(dtype=tf.float32),
            "discount":             tfds.features.Scalar(dtype=tf.float32),
            "alignment_confidence": tfds.features.Scalar(dtype=tf.float32),
        }),
    })


# ---------------------------------------------------------------------------
# Episode → TFDS dict conversion
# ---------------------------------------------------------------------------


def _pad7(arr: np.ndarray) -> np.ndarray:
    """Ensure a 1-D float32 array has exactly 7 elements (pad or truncate)."""
    arr = np.asarray(arr, dtype=np.float32).flatten()
    if len(arr) >= 7:
        return arr[:7]
    return np.pad(arr, (0, 7 - len(arr)))


def _ensure_image(img: Optional[np.ndarray]) -> np.ndarray:
    """
    Return a (480, 640, 3) uint8 image.
    Resizes with PIL if needed; returns blank array if img is None.
    """
    if img is None:
        return _BLANK_IMAGE
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:                       # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape == (480, 640, 3):
        return arr
    # Resize
    from PIL import Image as PILImage
    pil = PILImage.fromarray(arr).resize((640, 480), PILImage.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def _episode_to_dict(ep: InterleavedEpisode) -> dict:
    """Convert one InterleavedEpisode to a TFDS-serialisable dict."""
    task_desc = ep.task_description or ""

    steps = []
    for step in ep.steps:
        obs = step.observation
        r = step.reasoning

        steps.append({
            "observation": {
                "image_0": _ensure_image(obs.load_image()),
                "image_1": _ensure_image(obs.load_secondary_image()),
                "state":                _pad7(obs.state if obs.state is not None else _BLANK_STATE),
                "task_reasoning":       (r.task_reasoning    or "") if r else "",
                "subtask_reasoning":    (r.subtask_reasoning or "") if r else "",
                "move_reasoning":       (r.move_reasoning    or "") if r else "",
            },
            "language_instruction": task_desc,
            "action":               _pad7(step.action),
            "is_first":             bool(step.is_first),
            "is_last":              bool(step.is_last),
            "is_terminal":          bool(step.is_last),   # Bridge v2 uses is_last as terminal
            "reward":               0.0,
            "discount":             1.0,
            "alignment_confidence": float(step.alignment_confidence),
        })

    return {
        "episode_metadata": {
            "file_path": ep.episode_id,                    # original Bridge v2 path
            "episode_id": ep.episode_num if ep.episode_num is not None else -1,
        },
        "steps": steps,
    }


def _has_reasoning(ep: InterleavedEpisode) -> bool:
    """Return True if any step has a non-empty reasoning trace."""
    return any(
        step.reasoning is not None and (
            step.reasoning.task_reasoning
            or step.reasoning.subtask_reasoning
            or step.reasoning.move_reasoning
        )
        for step in ep.steps
    )


# ---------------------------------------------------------------------------
# TFDS builder
# ---------------------------------------------------------------------------


def _make_builder_class():
    """
    Dynamically create the TFDS GeneratorBasedBuilder class.

    Deferred so tensorflow_datasets is only imported when actually needed.
    """
    import tensorflow_datasets as tfds

    class VlaCuratedDataset(tfds.core.GeneratorBasedBuilder):
        """
        VLA curated dataset: ECoT reasoning + Bridge v2 demonstrations.

        Two builder configs:
          matched        — all ECoT-matched Bridge v2 episodes (empty reasoning
                           strings if the ECoT entry has no annotations)
          reasoning_only — only episodes with at least one non-empty reasoning
        """

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {
            "1.0.0": "ECoT reasoning annotations fused with Bridge v2 demonstrations."
        }
        BUILDER_CONFIGS = [
            tfds.core.BuilderConfig(
                name="matched",
                version=tfds.core.Version("1.0.0"),
                description=(
                    "All ECoT-matched Bridge v2 episodes. Episodes where ECoT "
                    "has no reasoning annotations get empty reasoning strings."
                ),
            ),
            tfds.core.BuilderConfig(
                name="reasoning_only",
                version=tfds.core.Version("1.0.0"),
                description=(
                    "Only Bridge v2 episodes that have at least one ECoT "
                    "reasoning annotation."
                ),
            ),
        ]

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                description=self.builder_config.description,
                features=_make_feature_spec(),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager):
            return {
                split: self._generate_examples(split)
                for split in _BUFFER_SPLITS
            }

        def _generate_examples(self, split):
            reasoning_only = (self.builder_config.name == "reasoning_only")
            episodes = _EPISODE_BUFFER.get(split, [])
            count = 0
            for ep in episodes:
                if reasoning_only and not _has_reasoning(ep):
                    continue
                try:
                    yield count, _episode_to_dict(ep)
                    count += 1
                except Exception:
                    logger.exception("Failed to serialise episode %s", ep.episode_id)
            logger.info(
                "Config=%s, split=%s: serialised %d episodes.",
                self.builder_config.name,
                split,
                count,
            )

    return VlaCuratedDataset


# ---------------------------------------------------------------------------
# Public exporter
# ---------------------------------------------------------------------------


class RLDSExporter(BaseExporter):
    """
    Export curated episodes to RLDS/TFDS TFRecord format.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    variants : list of str
        Which dataset variants to write.  Any combination of:
          ``"matched"``        — all ECoT-matched episodes (default)
          ``"reasoning_only"`` — only episodes with ECoT reasoning
        Default: ``["matched", "reasoning_only"]``

    Output layout:
        output_dir/
          vla_curated_dataset/
            matched/1.0.0/           (if "matched" in variants)
            reasoning_only/1.0.0/    (if "reasoning_only" in variants)
    """

    VALID_VARIANTS = ("matched", "reasoning_only")

    def __init__(
        self,
        output_dir: Path,
        variants: Optional[List[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if variants is None:
            self.variants = list(self.VALID_VARIANTS)
        else:
            unknown = [v for v in variants if v not in self.VALID_VARIANTS]
            if unknown:
                raise ValueError(
                    f"Unknown variant(s): {unknown}. "
                    f"Choose from: {self.VALID_VARIANTS}"
                )
            self.variants = list(variants)

        self._episodes: Dict[str, List[InterleavedEpisode]] = {}

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------

    def export_episode(self, episode: InterleavedEpisode, split: str = "train") -> None:
        """Buffer one episode. Episodes are written in bulk on write_metadata()."""
        if split not in self._episodes:
            self._episodes[split] = []
        self._episodes[split].append(episode)

    def write_metadata(self, stats: Dict[str, Any]) -> None:
        """
        Write all buffered episodes to RLDS TFRecord files.

        Called once after all episodes have been collected via export_episode().
        Prints a stats summary and merges episode counts into the stats dict.
        """
        if not self._episodes:
            logger.warning("RLDSExporter: no episodes to write.")
            return

        episode_stats = self._write_datasets()
        stats.update(episode_stats)

        import json
        stats_path = self.output_dir / "curation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Curation stats written to %s", stats_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_datasets(self) -> Dict[str, Any]:
        """
        Write TFDS dataset variants using the buffered episodes.

        Episodes are organized by split (train, val, etc.) matching the
        original Bridge v2 splits.  Each variant (matched, reasoning_only)
        is written as a separate TFDS dataset with all splits preserved.
        """
        global _EPISODE_BUFFER, _BUFFER_SPLITS
        _EPISODE_BUFFER = self._episodes
        _BUFFER_SPLITS = sorted(_EPISODE_BUFFER.keys())

        all_episodes = [ep for eps in _EPISODE_BUFFER.values() for ep in eps]
        total = len(all_episodes)
        n_with_reasoning = sum(1 for ep in all_episodes if _has_reasoning(ep))
        n_without_reasoning = total - n_with_reasoning

        # ── Print stats ──────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("RLDS export stats")
        logger.info("  Splits: %s", _BUFFER_SPLITS)
        for split in _BUFFER_SPLITS:
            split_eps = _EPISODE_BUFFER[split]
            split_reasoning = sum(1 for ep in split_eps if _has_reasoning(ep))
            logger.info(
                "  %s: %d episodes (%d with reasoning)",
                split, len(split_eps), split_reasoning,
            )
        logger.info("  Total: %d episodes (%d with reasoning)", total, n_with_reasoning)
        logger.info("=" * 60)

        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow-datasets required. "
                "pip install 'vla-data-curator[bridge]'"
            ) from exc

        builder_cls = _make_builder_class()
        dl_config = tfds.download.DownloadConfig(
            manual_dir=str(self.output_dir),
            register_checksums=False,
        )

        written: Dict[str, int] = {}

        # ── matched ──────────────────────────────────────────────────────────
        if "matched" in self.variants:
            logger.info("Writing matched dataset (%d episodes, splits=%s)…",
                        total, _BUFFER_SPLITS)
            builder_matched = builder_cls(config="matched", data_dir=str(self.output_dir))
            builder_matched.download_and_prepare(download_config=dl_config)
            written["matched"] = total
            logger.info(
                "matched → %s/vla_curated_dataset/matched/1.0.0/  (%d episodes)",
                self.output_dir, total,
            )
        else:
            logger.info("Skipping matched dataset (not requested).")
            written["matched"] = 0

        # ── reasoning_only ───────────────────────────────────────────────────
        if "reasoning_only" not in self.variants:
            logger.info("Skipping reasoning_only dataset (not requested).")
            written["reasoning_only"] = 0
        elif n_with_reasoning == 0:
            logger.warning(
                "reasoning_only dataset skipped — no episodes had reasoning.\n"
                "  Check that --ecot-path points to the correct directory and that\n"
                "  Bridge v2 source_file paths overlap with ECoT file_path keys."
            )
            written["reasoning_only"] = 0
        else:
            logger.info(
                "Writing reasoning_only dataset (%d episodes)…", n_with_reasoning
            )
            builder_ro = builder_cls(
                config="reasoning_only", data_dir=str(self.output_dir)
            )
            builder_ro.download_and_prepare(download_config=dl_config)
            written["reasoning_only"] = n_with_reasoning
            logger.info(
                "reasoning_only → %s/vla_curated_dataset/reasoning_only/1.0.0/"
                "  (%d episodes)",
                self.output_dir, n_with_reasoning,
            )

        return {
            "episodes_total":           total,
            "episodes_with_reasoning":  n_with_reasoning,
            "episodes_without_reasoning": n_without_reasoning,
            "reasoning_match_rate":     round(n_with_reasoning / max(total, 1), 4),
            "written_matched":          written["matched"],
            "written_reasoning_only":   written["reasoning_only"],
            "splits":                   _BUFFER_SPLITS,
        }
