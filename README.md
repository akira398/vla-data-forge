# VLA Data Forge

A research-grade Python framework for curating and preprocessing datasets for
**Vision-Language-Action (VLA)** model training, with a focus on
reasoning-aware embodied datasets.

The primary workflow: load reasoning annotations from
[Embodied-CoT](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge)
and robot demonstrations from [Bridge v2](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/),
align them, and produce merged interleaved episodes ready for downstream VLA training.

---

## Datasets

### Embodied-CoT (`embodied_features_bridge`)

A single JSON file (~1.4 GB) structured as a dict:

```
{
  "/path/to/bridge_episode/out.npy": {
      "task_reasoning":      "pick up the orange from the table",
      "subtask_reasoning":   "move arm above the orange",
      "move_reasoning":      "...",
      "gripper_reasoning":   "...",
      "attribute_reasoning": "...",
      "spatial_reasoning":   "..."
  },
  ...
}
```

Each key is a Bridge v2 episode path (the join key). Values are the 6 reasoning
fields. **No images are stored here** — images come from Bridge v2.

### Bridge v2

TFDS-format robot manipulation dataset with dual RGB cameras and 7-DoF actions.
Downloaded via the OpenVLA script into `bridge_orig/1.0.0/`.

---

## Repository Structure

```
vla-data-forge/
├── configs/
│   ├── datasets/          # embodied_cot.yaml, bridge_v2.yaml
│   ├── backends/          # gemini.yaml, openai.yaml, qwen.yaml
│   ├── generation/        # default.yaml
│   └── curation/          # interleaved.yaml
│
├── src/vla_curator/
│   ├── schemas/           # Canonical data types
│   │   ├── base.py            RobotAction
│   │   ├── embodied_cot.py    ReasoningTrace, ECoTStep, ECoTEpisode
│   │   ├── bridge_v2.py       BridgeObservation, BridgeStep, BridgeEpisode
│   │   ├── interleaved.py     EnrichedObservation, AlignedStep, InterleavedEpisode
│   │   └── modalities.py      DepthMap, SceneGraph, ModalityRegistry
│   │
│   ├── datasets/
│   │   ├── base.py            DatasetReader ABC
│   │   ├── embodied_cot.py    ECoTDatasetReader  (local JSON or HF Hub)
│   │   └── bridge_v2.py       BridgeV2DatasetReader  (TFDS)
│   │
│   ├── backends/          # Model provider abstraction
│   │   ├── gemini.py, openai_backend.py, qwen.py
│   │   └── registry.py        BackendRegistry, create_backend()
│   │
│   ├── generation/        # Reasoning-trace generation pipeline
│   ├── curation/          # Interleaving, validation, export
│   ├── visualization/
│   │   ├── bridge_viewer.py   BridgeViewer — dual-camera, gripper, actions, video
│   │   ├── frame_viewer.py    FrameViewer — frame grids, reasoning overlays
│   │   └── trajectory_viewer.py  TrajectoryViewer — action plots, coverage heatmaps
│   └── config.py
│
├── scripts/
│   ├── visualize_ecot.py      Inspect ECoT reasoning annotations
│   ├── visualize_bridge.py    Visualize Bridge v2 episodes
│   ├── generate_traces.py     Run reasoning-trace generation pipeline
│   ├── curate_interleaved.py  Merge ECoT + Bridge v2
│   └── validate_dataset.py    Validate curated JSONL output
│
└── tests/
```

---

## Installation

### 1. Create a conda environment

```bash
conda create -n vla-forge python=3.11 -y
conda activate vla-forge
```

### 2. Clone and install

```bash
git clone https://github.com/akira398/vla-data-forge
cd vla-data-forge
pip install -e ".[viz]"
```

For Bridge v2 loading (requires TensorFlow):

```bash
pip install -e ".[viz,bridge]"
```

For model backends:

```bash
pip install -e ".[gemini]"    && export GOOGLE_API_KEY="your-key"
pip install -e ".[openai]"    && export OPENAI_API_KEY="your-key"
pip install -e ".[qwen-api]"  && export DASHSCOPE_API_KEY="your-key"
```

Everything:

```bash
pip install -e ".[all]"
```

---

## Quick Start

### 1. Download the datasets

**Embodied-CoT:**
```bash
huggingface-cli download Embodied-CoT/embodied_features_bridge \
    --repo-type dataset --local-dir /datasets/embodied_features_bridge
# produces: /datasets/embodied_features_bridge/embodied_features_bridge.json
```

**Bridge v2:**
```bash
wget -r -nH --cut-dirs=4 --reject="index.html*" \
    https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
mv bridge_dataset bridge_orig
# produces: bridge_orig/1.0.0/dataset_info.json + *.tfrecord.gz
```

---

### 2. Inspect Embodied-CoT

Print top-level structure and a summary table of the first N file-path entries:

```bash
python scripts/visualize_ecot.py \
    --local-path /datasets/embodied_features_bridge
```

Show more entries:

```bash
python scripts/visualize_ecot.py \
    --local-path /datasets/embodied_features_bridge \
    --n 20
```

Show all reasoning steps of the first episode (features + annotated steps table):

```bash
python scripts/visualize_ecot.py \
    --local-path /datasets/embodied_features_bridge \
    --steps
```

---

### 3. Visualize Bridge v2

All figures are saved to disk — safe to run on headless servers.

```bash
# Dual-camera frame grid:
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode dual

# Full summary dashboard (frames + actions + gripper timeline):
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode summary --max-episodes 10

# Gripper open/close timeline:
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode gripper

# 7-DoF action component panels:
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode actions

# Proprioceptive state trajectory:
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode state

# Save MP4 videos (primary camera, max 16 frames):
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode video --camera 0 --fps 10

# Save GIFs (secondary camera):
python scripts/visualize_bridge.py \
    --local-path /datasets/bridge_orig --mode gif --camera 1
```

Available modes: `dual` | `grid` | `gripper` | `actions` | `state` | `summary` | `video` | `gif`

Default output: `outputs/viz/bridge/`. Override with `--save-dir`.
Default episode cap: `--max-episodes 5`. Default frame cap: `--max-frames 16`.

---

### 4. Curate interleaved dataset

Merges ECoT reasoning annotations with Bridge v2 observations and actions.
The default output format is **RLDS** (Bridge v2 compatible TFRecord files).

```bash
python scripts/curate_interleaved.py \
    --bridge-path /datasets/bridge_orig \
    --ecot-path /datasets/embodied_features_bridge

# Quick test with 100 episodes:
python scripts/curate_interleaved.py \
    --bridge-path /datasets/bridge_orig \
    --ecot-path /datasets/embodied_features_bridge \
    --max-episodes 100

# Legacy JSONL output (only matched episodes):
python scripts/curate_interleaved.py \
    --bridge-path /datasets/bridge_orig \
    --ecot-path /datasets/embodied_features_bridge \
    --format jsonl
```

Alignment strategies (used for ECoT → Bridge v2 step matching):

| Strategy | Description |
|----------|-------------|
| `nearest` | Each Bridge step gets reasoning from the closest annotated ECoT step (default) |
| `exact` | Only steps with a direct annotation keep reasoning; others get empty strings |
| `broadcast` | Copy a single episode-level trace to every step |

### 5. Validate output

```bash
python scripts/validate_dataset.py outputs/curated/episodes.jsonl
python scripts/validate_dataset.py outputs/curated/episodes.jsonl \
    --min-reasoning 0.5 --report outputs/validation_report.json
```

---

## Output Format (RLDS / TFDS TFRecords)

The default output is two TFDS-compatible datasets written under `outputs/curated/`:

```
outputs/curated/
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
```

| Variant | Contents |
|---------|----------|
| `full` | All Bridge v2 episodes. Episodes without an ECoT match have empty reasoning strings. |
| `reasoning_only` | Only episodes that have at least one ECoT reasoning annotation. |

### Feature spec (Bridge v2 compatible)

Field names and types match Bridge v2 exactly. Reasoning fields are added
inside `observation` so existing loaders that don't know about them will
silently ignore them.

```
episode_metadata/file_path           bytes   original Bridge v2 path (unchanged)

steps/observation/image_0            uint8   (480, 640, 3)  primary camera, JPEG
steps/observation/image_1            uint8   (480, 640, 3)  wrist camera, JPEG
steps/observation/state              float32 (7,)           EEF position + gripper
steps/observation/language_instruction  bytes
steps/observation/task_reasoning     bytes   empty string if no ECoT match
steps/observation/subtask_reasoning  bytes
steps/observation/move_reasoning     bytes

steps/action                         float32 (7,)           EEF delta + gripper cmd
steps/is_first                       bool
steps/is_last                        bool
steps/is_terminal                    bool
steps/reward                         float32
steps/discount                       float32
steps/alignment_confidence           float32  1.0=direct, 0.7=propagated, 0.0=none
```

### Loading

```python
import tensorflow_datasets as tfds

# All Bridge v2 episodes (empty reasoning for unmatched):
ds_full = tfds.builder_from_directory(
    "outputs/curated/vla_curated_dataset/full/1.0.0/"
).as_dataset(split="train")

# Only episodes with reasoning:
ds_reasoning = tfds.builder_from_directory(
    "outputs/curated/vla_curated_dataset/reasoning_only/1.0.0/"
).as_dataset(split="train")

for episode in ds_full:
    for step in episode["steps"]:
        image   = step["observation"]["image_0"]        # (480, 640, 3) uint8
        action  = step["action"]                        # (7,) float32
        task_r  = step["observation"]["task_reasoning"] # bytes
        conf    = step["alignment_confidence"]          # float32
```

`alignment_confidence`: `1.0` = directly annotated step, `0.7` = propagated from nearest, `0.0` = no ECoT match.

---

## Python API

```python
from pathlib import Path
from vla_curator.config import ECoTDatasetConfig, BridgeV2DatasetConfig, CurationConfig
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.datasets.bridge_v2 import BridgeV2DatasetReader
from vla_curator.curation.interleaver import EpisodeInterleaver
from vla_curator.curation.export import ExportFormat, create_exporter
from vla_curator.visualization import BridgeViewer

# Load ECoT reasoning annotations
ecot_cfg = ECoTDatasetConfig(
    local_path=Path("/datasets/embodied_features_bridge"),
    max_episodes=10,
)
for ep in ECoTDatasetReader(ecot_cfg):
    print(ep.episode_id)
    annotated = [s for s in ep.steps if s.reasoning is not None]
    print(f"{len(annotated)}/{len(ep.steps)} steps annotated")

# Load Bridge v2 episodes
bridge_cfg = BridgeV2DatasetConfig(
    source="tfds",
    local_path=Path("/datasets/bridge_orig"),
    max_episodes=5,
)
viewer = BridgeViewer()
for ep in BridgeV2DatasetReader(bridge_cfg):
    viewer.show_summary(ep, save_path=f"outputs/{ep.episode_id}.png")

# Curate and write RLDS datasets
cfg = CurationConfig(
    ecot=ECoTDatasetConfig(local_path=Path("/datasets/embodied_features_bridge")),
    bridge=BridgeV2DatasetConfig(source="tfds", local_path=Path("/datasets/bridge_orig")),
    alignment_strategy="nearest",
    output_dir=Path("outputs/curated"),
)
interleaver = EpisodeInterleaver(cfg, ECoTDatasetReader(cfg.ecot), BridgeV2DatasetReader(cfg.bridge))
exporter = create_exporter(ExportFormat.RLDS, cfg.output_dir)

for ep in interleaver.iter_all_episodes():   # all Bridge v2 eps, empty reasoning if no ECoT match
    exporter.export_episode(ep)

exporter.write_metadata({"alignment_strategy": "nearest"})
# → outputs/curated/vla_curated_dataset/full/1.0.0/
# → outputs/curated/vla_curated_dataset/reasoning_only/1.0.0/
```

---

## Running Tests

```bash
# All tests (no API keys, no GPU needed)
pytest

# With coverage
pytest --cov=vla_curator --cov-report=html
```

---

## License

MIT
