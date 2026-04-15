"""
RLDS/TFDS exporter — writes Bridge v2 compatible TFRecord datasets.

Produces two variants under output_dir:

  output_dir/vla_curated_dataset/full/1.0.0/
      — All Bridge v2 episodes; episodes without an ECoT match get empty
        reasoning strings.

  output_dir/vla_curated_dataset/reasoning_only/1.0.0/
      — Only episodes that have at least one non-empty reasoning annotation.

Both directories are loadable with ``tfds.builder_from_directory()``:

    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(
        "output_dir/vla_curated_dataset/full/1.0.0/"
    )
    ds = builder.as_dataset(split="train")
    for ep in ds:
        for step in ep["steps"]:
            print(step["observation"]["task_reasoning"])

Feature spec (Bridge v2 field names preserved, reasoning fields added):

  episode_metadata/file_path          bytes   original Bridge v2 path
  steps/observation/image_0           uint8   (480, 640, 3) JPEG
  steps/observation/image_1           uint8   (480, 640, 3) JPEG
  steps/observation/state             float32 (7,)
  steps/observation/language_instruction  bytes
  steps/observation/task_reasoning    bytes   empty string when no ECoT match
  steps/observation/subtask_reasoning bytes
  steps/observation/move_reasoning    bytes
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
_EPISODE_BUFFER: List[InterleavedEpisode] = []


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
                "language_instruction": tfds.features.Text(),
                "task_reasoning":       tfds.features.Text(),
                "subtask_reasoning":    tfds.features.Text(),
                "move_reasoning":       tfds.features.Text(),
            }),
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
                "language_instruction": task_desc,
                "task_reasoning":       r.task_reasoning    if r else "",
                "subtask_reasoning":    r.subtask_reasoning if r else "",
                "move_reasoning":       r.move_reasoning    if r else "",
            },
            "action":               _pad7(step.action),
            "is_first":             bool(step.is_first),
            "is_last":              bool(step.is_last),
            "is_terminal":          bool(step.is_last),   # Bridge v2 uses is_last as terminal
            "reward":               0.0,
            "discount":             1.0,
            "alignment_confidence": float(step.alignment_confidence),
        })

    return {
        "episode_metadata": {"file_path": ep.episode_id},   # original path, unchanged
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
          full           — all Bridge v2 episodes (empty reasoning if no ECoT match)
          reasoning_only — only episodes with at least one reasoning annotation
        """

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {
            "1.0.0": "ECoT reasoning annotations fused with Bridge v2 demonstrations."
        }
        BUILDER_CONFIGS = [
            tfds.core.BuilderConfig(
                name="full",
                version=tfds.core.Version("1.0.0"),
                description=(
                    "All Bridge v2 episodes. Episodes without an ECoT match "
                    "have empty reasoning strings."
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
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            reasoning_only = (self.builder_config.name == "reasoning_only")
            count = 0
            for ep in _EPISODE_BUFFER:
                if reasoning_only and not _has_reasoning(ep):
                    continue
                try:
                    yield ep.episode_id, _episode_to_dict(ep)
                    count += 1
                except Exception:
                    logger.exception("Failed to serialise episode %s", ep.episode_id)
            logger.info(
                "Config=%s: serialised %d episodes.",
                self.builder_config.name,
                count,
            )

    return VlaCuratedDataset


# ---------------------------------------------------------------------------
# Public exporter
# ---------------------------------------------------------------------------


class RLDSExporter(BaseExporter):
    """
    Export curated episodes to RLDS/TFDS TFRecord format.

    Collects all episodes in memory via ``export_episode()``, then writes
    both the ``full`` and ``reasoning_only`` TFDS datasets when
    ``write_metadata()`` is called.

    Output layout:
        output_dir/
          vla_curated_dataset/
            full/1.0.0/
              dataset_info.json
              features.json
              vla_curated_dataset-train.tfrecord-00000-of-NNNNN
              ...
            reasoning_only/1.0.0/
              dataset_info.json
              features.json
              vla_curated_dataset-train.tfrecord-00000-of-NNNNN
              ...
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: List[InterleavedEpisode] = []

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------

    def export_episode(self, episode: InterleavedEpisode) -> None:
        """Buffer one episode. Episodes are written in bulk on write_metadata()."""
        self._episodes.append(episode)

    def write_metadata(self, stats: Dict[str, Any]) -> None:
        """
        Write all buffered episodes to RLDS TFRecord files.

        Called once after all episodes have been collected via export_episode().
        """
        if not self._episodes:
            logger.warning("RLDSExporter: no episodes to write.")
            return

        self._write_datasets()

        # Also write a human-readable stats file alongside
        import json
        stats_path = self.output_dir / "curation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Curation stats written to %s", stats_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_datasets(self) -> None:
        """Write both TFDS dataset variants using the buffered episodes."""
        global _EPISODE_BUFFER
        _EPISODE_BUFFER = self._episodes

        logger.info(
            "Writing RLDS datasets (%d episodes total)…", len(_EPISODE_BUFFER)
        )

        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow-datasets required. "
                "pip install 'vla-data-curator[bridge]'"
            ) from exc

        builder_cls = _make_builder_class()

        for config_name in ("full", "reasoning_only"):
            logger.info("Writing config=%s…", config_name)
            builder = builder_cls(
                config=config_name,
                data_dir=str(self.output_dir),
            )
            builder.download_and_prepare(
                download_config=tfds.download.DownloadConfig(
                    manual_dir=str(self.output_dir),
                    register_checksums=False,
                )
            )
            version_dir = (
                self.output_dir
                / "vla_curated_dataset"
                / config_name
                / "1.0.0"
            )
            n = sum(
                1 for ep in _EPISODE_BUFFER
                if config_name == "full" or _has_reasoning(ep)
            )
            logger.info(
                "Config=%s written → %s  (%d episodes)",
                config_name,
                version_dir,
                n,
            )
