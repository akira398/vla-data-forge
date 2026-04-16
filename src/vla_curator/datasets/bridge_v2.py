"""
Bridge v2 dataset reader — compatible with the OpenVLA download layout.

Obtaining the data (OpenVLA instructions)
-----------------------------------------
    cd <BASE_DIR>
    wget -r -nH --cut-dirs=4 --reject="index.html*" \\
        https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
    mv bridge_dataset bridge_orig          # required rename

This produces:
    <BASE_DIR>/bridge_orig/
        1.0.0/
            dataset_info.json
            features.json
            bridge_dataset-train.tfrecord-00000-of-00300
            bridge_dataset-train.tfrecord-00001-of-00300
            ...
            bridge_dataset-val.tfrecord-00000-of-00003

Set in config / YAML:
    source: tfds
    local_path: <BASE_DIR>/bridge_orig

Loading strategy
----------------
When ``local_path`` is set, the reader calls
``tfds.builder_from_directory(version_dir)`` which reads the local TFRecord
files directly — no network access, no dependency on the dataset being
registered under a particular name (which would break after the mv rename).

``version_dir`` is auto-discovered as the first subdirectory of ``local_path``
that contains a ``dataset_info.json`` file, e.g. ``bridge_orig/1.0.0/``.

Why not ``tfds.builder("bridge_dataset", data_dir=local_path)``?
  TFDS would look for ``local_path/bridge_dataset/`` which no longer exists
  after the rename to ``bridge_orig``.  ``builder_from_directory`` bypasses
  the name-based lookup entirely.

TFDS step field layout (bridge_dataset 1.0.0)
----------------------------------------------
Each episode:
    steps                               tf.data.Dataset
    episode_metadata/file_path          bytes (trajectory path in archive)

Each step:
    observation/image_0                 (480, 640, 3) uint8
    observation/image_1                 (480, 640, 3) uint8
    observation/image_2                 (480, 640, 3) uint8   (may be zeros)
    observation/image_3                 (480, 640, 3) uint8   (may be zeros)
    observation/state                   (7,) float32
    observation/language_instruction    bytes
    action                              (7,) float32
    is_first / is_last / is_terminal    bool
    reward / discount                   float32

HDF5 format (older distribution)
---------------------------------
    obs/images0/   (T, H, W, 3) uint8
    obs/images1/   (T, H, W, 3) uint8
    obs/state/     (T, 7) float32
    actions/       (T, 7) float32
    lang           str attribute
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..config import BridgeV2DatasetConfig
from ..schemas.bridge_v2 import BridgeEpisode, BridgeObservation, BridgeStep
from .base import DatasetReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TFDS version-directory discovery
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def find_tfds_version_dir(base_path: Path) -> Path:
    """
    Return the TFDS version directory inside ``base_path``.

    Looks for a subdirectory whose name matches ``X.Y.Z`` and that contains
    a ``dataset_info.json`` file.  If multiple versions exist the latest is
    returned.  If ``base_path`` itself is a version directory it is returned
    directly.

    Raises FileNotFoundError with a clear message if nothing is found.
    """
    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Bridge v2 local_path does not exist: {base_path}\n"
            "Check the local_path in your BridgeV2DatasetConfig."
        )

    # Is base_path itself the version dir?
    if (base_path / "dataset_info.json").exists():
        return base_path

    # Look one level down
    version_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir()
        and _VERSION_RE.match(d.name)
        and (d / "dataset_info.json").exists()
    ]

    if not version_dirs:
        raise FileNotFoundError(
            f"No TFDS version directory found under: {base_path}\n\n"
            "Expected structure:\n"
            f"  {base_path}/\n"
            "    1.0.0/\n"
            "      dataset_info.json\n"
            "      *.tfrecord.gz\n\n"
            "Make sure you followed the OpenVLA download instructions:\n"
            "  wget ... https://.../bridge_dataset/\n"
            "  mv bridge_dataset bridge_orig\n"
            "Then set  local_path: <BASE_DIR>/bridge_orig  in your config."
        )

    # Return latest version
    version_dirs.sort(key=lambda d: tuple(int(x) for x in d.name.split(".")))
    chosen = version_dirs[-1]
    logger.debug("Auto-discovered TFDS version directory: %s", chosen)
    return chosen


# ---------------------------------------------------------------------------
# TF tensor decoding helpers
# ---------------------------------------------------------------------------


def _tensor_to_numpy(val: Any) -> Any:
    """Call .numpy() on a TF tensor if needed; pass through everything else."""
    if hasattr(val, "numpy"):
        return val.numpy()
    return val


def _decode_bytes(val: Any) -> str:
    """Decode a bytes / TF bytes-tensor to a Python str."""
    val = _tensor_to_numpy(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return str(val) if val is not None else ""


def _decode_tf_image(
    img_raw: Any,
    image_size: Optional[tuple[int, int]],
) -> Optional[np.ndarray]:
    """
    Convert a TF tensor or numpy array to a resized uint8 numpy image.

    Returns None if the input is None or has zero elements.
    """
    if img_raw is None:
        return None

    arr = _tensor_to_numpy(img_raw)
    arr = np.asarray(arr, dtype=np.uint8)

    if arr.size == 0:
        return None

    if image_size is not None:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(arr)
        pil = pil.resize((image_size[1], image_size[0]), PILImage.BILINEAR)
        arr = np.array(pil, dtype=np.uint8)

    return arr


def _get_obs_image(obs: dict, key: str) -> Any:
    """
    Safely retrieve an image field from the observation dict.

    Uses explicit None check instead of ``or`` to avoid the
    ``ValueError: The truth value of a Tensor is ambiguous`` error that
    occurs when using boolean short-circuit on TF tensors.
    """
    val = obs.get(key)
    return val if val is not None else None


# ---------------------------------------------------------------------------
# TFDS step / episode parsers
# ---------------------------------------------------------------------------


def _parse_tfds_step(
    step: Any,
    step_index: int,
    image_size: Optional[tuple[int, int]],
    include_secondary: bool,
) -> BridgeStep:
    """Parse one RLDS step dict (values may be TF tensors) into a BridgeStep."""
    obs = step.get("observation", step)

    # --- Images ---
    # Use explicit None checks, never Python `or` on TF tensors.
    img0_raw = _get_obs_image(obs, "image_0")
    if img0_raw is None:
        img0_raw = _get_obs_image(obs, "images0")
    img0 = _decode_tf_image(img0_raw, image_size)

    img1 = None
    if include_secondary:
        img1_raw = _get_obs_image(obs, "image_1")
        if img1_raw is None:
            img1_raw = _get_obs_image(obs, "images1")
        img1 = _decode_tf_image(img1_raw, image_size)

    # --- State ---
    state_raw = obs.get("state")
    if state_raw is None:
        state_raw = obs.get("proprio")
    state: Optional[np.ndarray] = None
    if state_raw is not None:
        state = np.asarray(_tensor_to_numpy(state_raw), dtype=np.float32).flatten()

    obs_obj = BridgeObservation(
        step_index=step_index,
        image_0=img0,
        image_1=img1,
        state=state,
    )

    # --- Action ---
    action_raw = step.get("action")
    if action_raw is not None:
        action = np.asarray(_tensor_to_numpy(action_raw), dtype=np.float32).flatten()
    else:
        action = np.zeros(7, dtype=np.float32)
        logger.warning("Step %d has no action — using zeros.", step_index)

    # --- Language instruction ---
    # Bridge v2 TFDS stores language_instruction inside observation.
    # Fall back to step level for other RLDS datasets that put it there.
    raw_instruction = obs.get("language_instruction")
    if raw_instruction is None:
        raw_instruction = step.get("language_instruction", b"")
    instruction = _decode_bytes(raw_instruction)

    # --- Flags (TF bool tensors — bool() works fine on scalars in eager mode) ---
    def _to_bool(val: Any, default: bool) -> bool:
        if val is None:
            return default
        try:
            return bool(_tensor_to_numpy(val))
        except Exception:
            return default

    return BridgeStep(
        step_index=step_index,
        observation=obs_obj,
        action=action,
        language_instruction=instruction,
        is_first=_to_bool(step.get("is_first"), step_index == 0),
        is_last=_to_bool(step.get("is_last"), False),
        is_terminal=_to_bool(step.get("is_terminal"), False),
        reward=float(_tensor_to_numpy(step.get("reward", 0.0))),
        discount=float(_tensor_to_numpy(step.get("discount", 1.0))),
    )


def _parse_tfds_episode(
    ep: Any,
    ep_index: int,
    image_size: Optional[tuple[int, int]],
    include_secondary: bool,
) -> BridgeEpisode:
    """Convert one RLDS episode to a BridgeEpisode."""
    # --- Episode metadata ---
    source_file: Optional[str] = None
    if "episode_metadata" in ep:
        fp_raw = ep["episode_metadata"].get("file_path")
        if fp_raw is not None:
            source_file = _decode_bytes(fp_raw) or None

    episode_id = source_file or f"bridge_ep_{ep_index:06d}"

    # --- Steps ---
    steps: List[BridgeStep] = []
    instruction = ""

    for i, raw_step in enumerate(ep["steps"]):
        s = _parse_tfds_step(raw_step, i, image_size, include_secondary)
        if not instruction and s.language_instruction:
            instruction = s.language_instruction
        steps.append(s)

    if steps:
        steps[0].is_first = True
        steps[-1].is_last = True

    return BridgeEpisode(
        episode_id=episode_id,
        language_instruction=instruction,
        steps=steps,
        source_file=source_file,
    )


# ---------------------------------------------------------------------------
# HDF5 loader (older Bridge v2 distribution)
# ---------------------------------------------------------------------------


def _load_hdf5_episode(path: Path, image_size: Optional[tuple[int, int]]) -> BridgeEpisode:
    """Load a single per-episode HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load HDF5 episodes. pip install h5py"
        ) from exc

    with h5py.File(path, "r") as f:
        obs_grp = f.get("obs", f)
        imgs0  = np.asarray(obs_grp["images0"]) if "images0" in obs_grp else None
        imgs1  = np.asarray(obs_grp["images1"]) if "images1" in obs_grp else None
        states = np.asarray(obs_grp["state"])   if "state"   in obs_grp else None
        actions = np.asarray(f["actions"])       if "actions" in f       else None
        instruction = ""
        for attr_key in ("lang", "language_instruction"):
            if attr_key in f.attrs:
                instruction = f.attrs[attr_key]
                break

    T = (
        len(actions)  if actions is not None else
        len(imgs0)    if imgs0   is not None else 0
    )
    steps: List[BridgeStep] = []
    for i in range(T):
        img0 = imgs0[i] if imgs0 is not None else None
        img1 = imgs1[i] if imgs1 is not None else None

        if image_size is not None and img0 is not None:
            from PIL import Image as PILImage
            img0 = np.array(
                PILImage.fromarray(img0).resize(
                    (image_size[1], image_size[0]), PILImage.BILINEAR
                ),
                dtype=np.uint8,
            )

        obs_obj = BridgeObservation(
            step_index=i,
            image_0=img0,
            image_1=img1,
            state=(
                np.asarray(states[i], dtype=np.float32).flatten()
                if states is not None else None
            ),
        )
        steps.append(BridgeStep(
            step_index=i,
            observation=obs_obj,
            action=(
                np.asarray(actions[i], dtype=np.float32).flatten()
                if actions is not None else np.zeros(7, dtype=np.float32)
            ),
            language_instruction=instruction,
            is_first=(i == 0),
            is_last=(i == T - 1),
        ))

    return BridgeEpisode(
        episode_id=path.stem,
        language_instruction=instruction,
        steps=steps,
        source_file=str(path),
    )


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------


class BridgeV2DatasetReader(DatasetReader[BridgeEpisode]):
    """
    Reader for Bridge v2 trajectories (OpenVLA ``bridge_orig`` layout).

    Quick start
    -----------
    from vla_curator.datasets import BridgeV2DatasetReader
    from vla_curator.config import BridgeV2DatasetConfig
    from pathlib import Path

    cfg = BridgeV2DatasetConfig(
        source="tfds",
        local_path=Path("/datasets/bridge_orig"),  # the renamed folder
        split="train",
        max_episodes=10,
    )
    reader = BridgeV2DatasetReader(cfg)
    for episode in reader:
        print(episode)

    HDF5 mode
    ---------
    cfg = BridgeV2DatasetConfig(
        source="hdf5",
        local_path=Path("/data/bridge_v2_raw"),
    )
    """

    dataset_name = "bridge_v2"

    def __init__(self, config: BridgeV2DatasetConfig) -> None:
        self.config = config
        self._hdf5_paths: Optional[List[Path]] = None
        self._tfds_dataset: Any = None
        self._tfds_builder: Any = None

    # ------------------------------------------------------------------
    # TFDS / RLDS loading
    # ------------------------------------------------------------------

    def _get_tfds_builder(self) -> Any:
        """
        Return a TFDS builder for the local bridge_orig dataset.

        Uses ``tfds.builder_from_directory`` when ``local_path`` is set
        (the standard case for the OpenVLA download).  Falls back to the
        registered dataset name + default cache otherwise.
        """
        if self._tfds_builder is not None:
            return self._tfds_builder

        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow and tensorflow-datasets are required. "
                "Install with: pip install 'vla-data-curator[bridge]'"
            ) from exc

        if self.config.local_path is not None:
            # Discover version dir, e.g. bridge_orig/1.0.0/
            version_dir = find_tfds_version_dir(Path(self.config.local_path))
            logger.info(
                "Loading Bridge v2 from local TFDS dir: %s (split=%s)",
                version_dir,
                self.config.split,
            )
            self._tfds_builder = tfds.builder_from_directory(str(version_dir))
        else:
            # Fallback: registered name + TFDS default cache
            logger.info(
                "Loading Bridge v2 via TFDS registry: %s (split=%s)",
                self.config.tfds_name,
                self.config.split,
            )
            self._tfds_builder = tfds.builder(self.config.tfds_name)

        return self._tfds_builder

    def _get_tfds_dataset(self) -> Any:
        if self._tfds_dataset is None:
            builder = self._get_tfds_builder()
            self._tfds_dataset = builder.as_dataset(
                split=self.config.split,
                shuffle_files=self.config.shuffle,
            )
        return self._tfds_dataset

    def _get_hdf5_paths(self) -> List[Path]:
        if self._hdf5_paths is None:
            base = Path(self.config.local_path)
            paths = sorted(base.rglob("*.hdf5")) + sorted(base.rglob("*.h5"))
            if not paths:
                raise FileNotFoundError(
                    f"No HDF5 files found under {base}. "
                    "Check local_path in your BridgeV2DatasetConfig."
                )
            self._hdf5_paths = paths
            logger.info("Found %d HDF5 files under %s.", len(paths), base)
        return self._hdf5_paths

    # ------------------------------------------------------------------
    # DatasetReader interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[BridgeEpisode]:
        limit = self.config.max_episodes

        if self.config.source in ("tfds", "rlds"):
            ds = self._get_tfds_dataset()
            count = 0
            for i, ep in enumerate(ds):
                if limit is not None and count >= limit:
                    break
                try:
                    yield _parse_tfds_episode(
                        ep, i,
                        self.config.image_size,
                        self.config.include_secondary_camera,
                    )
                    count += 1
                except Exception:
                    logger.exception("Failed to parse episode %d — skipping.", i)

        elif self.config.source == "hdf5":
            paths = self._get_hdf5_paths()
            if limit is not None:
                paths = paths[:limit]
            for path in paths:
                try:
                    yield _load_hdf5_episode(path, self.config.image_size)
                except Exception:
                    logger.exception("Failed to load %s — skipping.", path)

        else:
            raise ValueError(
                f"Unknown source: {self.config.source!r}. "
                "Choose from: 'tfds', 'rlds', 'hdf5'."
            )

    def load_episode(self, episode_id: str) -> Optional[BridgeEpisode]:
        if self.config.source == "hdf5":
            for path in self._get_hdf5_paths():
                if path.stem == episode_id or str(path) == episode_id:
                    return _load_hdf5_episode(path, self.config.image_size)
        # TFDS: linear scan (build an episode_id→index cache for repeated use)
        for ep in self:
            if ep.episode_id == episode_id or ep.source_file == episode_id:
                return ep
        return None

    def episode_ids(self) -> List[str]:
        if self.config.source == "hdf5":
            return [p.stem for p in self._get_hdf5_paths()]
        ids: List[str] = []
        for ep in self:
            ids.append(ep.episode_id)
        return ids

    def __len__(self) -> int:
        if self.config.source == "hdf5":
            n = len(self._get_hdf5_paths())
        else:
            try:
                builder = self._get_tfds_builder()
                split_info = builder.info.splits.get(self.config.split)
                n = split_info.num_examples if split_info else 0
            except Exception:
                logger.debug("Could not determine dataset length without iterating.")
                return 0
        if self.config.max_episodes is not None:
            return min(n, self.config.max_episodes)
        return n

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "source": self.config.source,
            "local_path": str(self.config.local_path),
            "split": self.config.split,
            "image_size": self.config.image_size,
        })
        return base
