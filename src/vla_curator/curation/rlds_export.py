"""
RLDS/TFDS exporter — writes Bridge v2 superset TFRecord datasets.

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
            print(step["reasoning"]["task"])

Feature spec — Bridge v2 superset (all Bridge v2 fields + ECoT annotations):

  episode_metadata/
    file_path              text     original Bridge v2 path
    episode_id             int32    Bridge v2 per-file episode ID
    has_image_0            bool     Bridge v2 metadata
    has_image_1            bool
    has_image_2            bool
    has_image_3            bool
    has_language           bool
  steps/
    observation/
      image_0              uint8   (256, 256, 3) JPEG
      image_1              uint8   (256, 256, 3) JPEG
      image_2              uint8   (256, 256, 3) JPEG
      image_3              uint8   (256, 256, 3) JPEG
      state                float32 (7,)
    action                 float32 (7,)
    language_instruction   text
    language_embedding     float32 (512,)
    is_first               bool
    is_last                bool
    is_terminal            bool
    reward                 float32
    discount               float32
    reasoning/
      task                 text    high-level task description
      plan                 text    multi-step plan
      subtask              text    current subtask label
      subtask_reason       text    why this subtask
      move                 text    motion description
      move_reason          text    why this motion
    ecot_features/
      caption              text    scene description (episode-level, repeated)
      move_primitive       text    per-step motion label
      gripper_position     float32 (2,) pixel [x,y]
      bboxes               text    JSON string, variable shape
      state_3d             float32 (3,) end-effector [x,y,z]
    alignment_confidence   float32 1.0=direct, 0.7=propagated, 0.0=none
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..schemas.interleaved import InterleavedEpisode
from .export import BaseExporter

logger = logging.getLogger(__name__)

# Blank arrays used when an episode has no image / state data
_IMG_SHAPE = (256, 256, 3)
_BLANK_IMAGE = np.zeros(_IMG_SHAPE, dtype=np.uint8)
_BLANK_STATE = np.zeros(7, dtype=np.float32)
_BLANK_ACTION = np.zeros(7, dtype=np.float32)
_BLANK_EMBEDDING = np.zeros(512, dtype=np.float32)
_BLANK_GRIPPER_POS = np.zeros(2, dtype=np.float32)
_BLANK_STATE_3D = np.zeros(3, dtype=np.float32)

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
    """Build the TFDS feature spec (Bridge v2 superset + ECoT annotations)."""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow and tensorflow-datasets are required for RLDS export.\n"
            "Install with:  pip install 'vla-data-curator[bridge]'"
        ) from exc

    img_shape = _IMG_SHAPE

    return tfds.features.FeaturesDict({
        "episode_metadata": tfds.features.FeaturesDict({
            "file_path":    tfds.features.Text(),
            "episode_id":   tfds.features.Scalar(dtype=tf.int32),
            "has_image_0":  tfds.features.Scalar(dtype=tf.bool),
            "has_image_1":  tfds.features.Scalar(dtype=tf.bool),
            "has_image_2":  tfds.features.Scalar(dtype=tf.bool),
            "has_image_3":  tfds.features.Scalar(dtype=tf.bool),
            "has_language":  tfds.features.Scalar(dtype=tf.bool),
        }),
        "steps": tfds.features.Dataset({
            # ── Bridge v2 observation ────────────────────────────────────
            "observation": tfds.features.FeaturesDict({
                "image_0": tfds.features.Image(
                    shape=img_shape, dtype=tf.uint8, encoding_format="jpeg",
                ),
                "image_1": tfds.features.Image(
                    shape=img_shape, dtype=tf.uint8, encoding_format="jpeg",
                ),
                "image_2": tfds.features.Image(
                    shape=img_shape, dtype=tf.uint8, encoding_format="jpeg",
                ),
                "image_3": tfds.features.Image(
                    shape=img_shape, dtype=tf.uint8, encoding_format="jpeg",
                ),
                "state": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            }),
            # ── Bridge v2 step-level fields ──────────────────────────────
            "action":               tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "language_instruction": tfds.features.Text(),
            "language_embedding":   tfds.features.Tensor(shape=(512,), dtype=tf.float32),
            "is_first":             tfds.features.Scalar(dtype=tf.bool),
            "is_last":              tfds.features.Scalar(dtype=tf.bool),
            "is_terminal":          tfds.features.Scalar(dtype=tf.bool),
            "reward":               tfds.features.Scalar(dtype=tf.float32),
            "discount":             tfds.features.Scalar(dtype=tf.float32),
            # ── ECoT reasoning (6 fields) ────────────────────────────────
            "reasoning": tfds.features.FeaturesDict({
                "task":            tfds.features.Text(),
                "plan":            tfds.features.Text(),
                "subtask":         tfds.features.Text(),
                "subtask_reason":  tfds.features.Text(),
                "move":            tfds.features.Text(),
                "move_reason":     tfds.features.Text(),
            }),
            # ── ECoT per-step features ───────────────────────────────────
            "ecot_features": tfds.features.FeaturesDict({
                "caption":           tfds.features.Text(),
                "move_primitive":    tfds.features.Text(),
                "gripper_position":  tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                "bboxes":            tfds.features.Text(),
                "state_3d":          tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            }),
            # ── Curation metadata ────────────────────────────────────────
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
    Return a (256, 256, 3) uint8 image matching Bridge v2 TFDS format.
    Resizes with PIL if needed; returns blank array if img is None.
    """
    if img is None:
        return _BLANK_IMAGE
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:                       # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape == _IMG_SHAPE:
        return arr
    # Resize to match Bridge v2
    from PIL import Image as PILImage
    h, w = _IMG_SHAPE[:2]
    pil = PILImage.fromarray(arr).resize((w, h), PILImage.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def _ensure_embedding(emb: Optional[np.ndarray]) -> np.ndarray:
    """Return a (512,) float32 embedding, or zeros if None."""
    if emb is None:
        return _BLANK_EMBEDDING
    arr = np.asarray(emb, dtype=np.float32).flatten()
    if len(arr) >= 512:
        return arr[:512]
    return np.pad(arr, (0, 512 - len(arr)))


def _ensure_gripper_pos(pos: Optional[np.ndarray]) -> np.ndarray:
    """Return a (2,) float32 gripper position, or zeros if None."""
    if pos is None:
        return _BLANK_GRIPPER_POS
    return np.asarray(pos, dtype=np.float32).flatten()[:2]


def _ensure_state_3d(s3d: Optional[np.ndarray]) -> np.ndarray:
    """Return a (3,) float32 state_3d, or zeros if None."""
    if s3d is None:
        return _BLANK_STATE_3D
    arr = np.asarray(s3d, dtype=np.float32).flatten()
    if len(arr) >= 3:
        return arr[:3]
    return np.pad(arr, (0, 3 - len(arr)))


def _load_tertiary_image(obs) -> Optional[np.ndarray]:
    """Load image_tertiary from EnrichedObservation."""
    if obs.image_tertiary is not None:
        return obs.image_tertiary
    if obs.image_tertiary_path is not None:
        from PIL import Image as PILImage
        return np.array(PILImage.open(obs.image_tertiary_path).convert("RGB"))
    return None


def _load_quaternary_image(obs) -> Optional[np.ndarray]:
    """Load image_quaternary from EnrichedObservation."""
    if obs.image_quaternary is not None:
        return obs.image_quaternary
    if obs.image_quaternary_path is not None:
        from PIL import Image as PILImage
        return np.array(PILImage.open(obs.image_quaternary_path).convert("RGB"))
    return None


def _episode_to_dict(ep: InterleavedEpisode) -> dict:
    """Convert one InterleavedEpisode to a TFDS-serialisable dict."""
    task_desc = ep.task_description or ""
    caption = ep.caption or ""

    steps = []
    for step in ep.steps:
        obs = step.observation
        r = step.reasoning

        # Serialize bboxes as JSON string
        bboxes_str = ""
        if step.bboxes is not None:
            try:
                bboxes_str = _json.dumps(step.bboxes)
            except (TypeError, ValueError):
                bboxes_str = str(step.bboxes)

        steps.append({
            "observation": {
                "image_0": _ensure_image(obs.load_image()),
                "image_1": _ensure_image(obs.load_secondary_image()),
                "image_2": _ensure_image(_load_tertiary_image(obs)),
                "image_3": _ensure_image(_load_quaternary_image(obs)),
                "state":   _pad7(obs.state if obs.state is not None else _BLANK_STATE),
            },
            "action":               _pad7(step.action),
            "language_instruction": task_desc,
            "language_embedding":   _ensure_embedding(step.language_embedding),
            "is_first":             bool(step.is_first),
            "is_last":              bool(step.is_last),
            "is_terminal":          bool(step.is_terminal),
            "reward":               float(step.reward),
            "discount":             float(step.discount),
            "reasoning": {
                "task":            (r.task_reasoning    or "") if r else "",
                "plan":            (r.plan              or "") if r else "",
                "subtask":         (r.subtask_reasoning or "") if r else "",
                "subtask_reason":  (r.subtask_reason    or "") if r else "",
                "move":            (r.move_reasoning    or "") if r else "",
                "move_reason":     (r.move_reason       or "") if r else "",
            },
            "ecot_features": {
                "caption":          caption,
                "move_primitive":   step.move_primitive or "",
                "gripper_position": _ensure_gripper_pos(step.gripper_position),
                "bboxes":           bboxes_str,
                "state_3d":         _ensure_state_3d(step.state_3d),
            },
            "alignment_confidence": float(step.alignment_confidence),
        })

    return {
        "episode_metadata": {
            "file_path":    ep.episode_id,
            "episode_id":   ep.episode_num if ep.episode_num is not None else -1,
            "has_image_0":  bool(ep.has_image_0),
            "has_image_1":  bool(ep.has_image_1),
            "has_image_2":  bool(ep.has_image_2),
            "has_image_3":  bool(ep.has_image_3),
            "has_language": bool(ep.has_language),
        },
        "steps": steps,
    }


def _has_reasoning(ep: InterleavedEpisode) -> bool:
    """Return True if any step has a non-empty reasoning trace."""
    return any(
        step.reasoning is not None and (
            step.reasoning.task_reasoning
            or step.reasoning.plan
            or step.reasoning.subtask_reasoning
            or step.reasoning.subtask_reason
            or step.reasoning.move_reasoning
            or step.reasoning.move_reason
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

        # Skip TFDS's shuffle-bucket stage. Records are episode-level
        # trajectories (no benefit to cross-example randomization), and the
        # shuffle stage's bucket write/read/unlink cycle trips EINVAL on some
        # shared HPC filesystems (e.g. /groups/AIC-MV Lustre-like storage).
        DISABLE_SHUFFLING = True

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
