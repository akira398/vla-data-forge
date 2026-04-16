"""
CLI: Count episodes in Bridge v2 and/or ECoT datasets.

Usage
-----
# Count ECoT episodes (from local JSON):
python scripts/count_episodes.py ecot --path /path/to/embodied_features_bridge.json

# Count Bridge v2 episodes (from local TFDS):
python scripts/count_episodes.py bridge --path /path/to/bridge_dataset

# Count both:
python scripts/count_episodes.py both \
    --ecot-path /path/to/embodied_features_bridge.json \
    --bridge-path /path/to/bridge_dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Count episodes in Bridge v2 and ECoT datasets.")
console = Console()


# ---------------------------------------------------------------------------
# ECoT counting
# ---------------------------------------------------------------------------


def count_ecot(json_path: Path) -> dict:
    """Count episodes in an ECoT JSON file.

    Returns a dict with:
      - num_file_paths: number of top-level file_path keys
      - num_episodes: total (file_path, episode_id) pairs
      - episodes_per_path: distribution stats
    """
    if json_path.is_dir():
        candidates = sorted(
            f for f in json_path.rglob("*.json")
            if not f.name.startswith(".")
            and f.name not in ("dataset_info.json", "dataset_dict.json")
        )
        if not candidates:
            raise FileNotFoundError(f"No .json files found under {json_path}")
        json_path = candidates[0]

    size_mb = json_path.stat().st_size / 1e6
    console.print(f"Loading [bold]{json_path.name}[/bold] ({size_mb:.0f} MB)...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_file_paths = 0
    num_episodes = 0
    episodes_per_path = []

    for file_path, episodes in data.items():
        if not isinstance(episodes, dict):
            continue
        num_file_paths += 1
        ep_count = len(episodes)
        num_episodes += ep_count
        episodes_per_path.append(ep_count)

    # Distribution stats
    if episodes_per_path:
        avg = sum(episodes_per_path) / len(episodes_per_path)
        mn = min(episodes_per_path)
        mx = max(episodes_per_path)
    else:
        avg = mn = mx = 0

    return {
        "json_file": str(json_path),
        "num_file_paths": num_file_paths,
        "num_episodes": num_episodes,
        "avg_episodes_per_path": round(avg, 2),
        "min_episodes_per_path": mn,
        "max_episodes_per_path": mx,
    }


# ---------------------------------------------------------------------------
# Bridge v2 counting
# ---------------------------------------------------------------------------


def count_bridge(dataset_path: Path) -> dict:
    """Count episodes in a Bridge v2 TFDS dataset.

    Loads the dataset via tensorflow_datasets and counts episodes.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets required. pip install 'vla-data-curator[bridge]'"
        ) from exc

    from vla_curator.datasets.bridge_v2 import find_tfds_version_dir

    version_dir = find_tfds_version_dir(dataset_path)
    console.print(f"Loading Bridge v2 from [bold]{version_dir}[/bold]...")

    builder = tfds.builder_from_directory(str(version_dir))
    split_info = builder.info.splits.get("train")

    if split_info is not None:
        num_episodes = split_info.num_examples
        console.print(f"Episodes from split info: [bold]{num_episodes}[/bold]")
    else:
        console.print("No split info found, counting by iteration...")
        num_episodes = 0
        for _ in builder.as_dataset(split="train"):
            num_episodes += 1

    return {
        "dataset_path": str(version_dir),
        "num_episodes": num_episodes,
    }


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def ecot(
    path: Path = typer.Option(
        ..., "--path", "-p",
        help="Path to ECoT JSON file or directory containing it.",
    ),
) -> None:
    """Count episodes in ECoT dataset."""
    stats = count_ecot(path)

    t = Table(title="ECoT Episode Count", show_lines=True)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("JSON file", stats["json_file"])
    t.add_row("File paths (top-level keys)", str(stats["num_file_paths"]))
    t.add_row("Total episodes", str(stats["num_episodes"]))
    t.add_row("Avg episodes per path", str(stats["avg_episodes_per_path"]))
    t.add_row("Min episodes per path", str(stats["min_episodes_per_path"]))
    t.add_row("Max episodes per path", str(stats["max_episodes_per_path"]))
    console.print(t)


@app.command()
def bridge(
    path: Path = typer.Option(
        ..., "--path", "-p",
        help="Path to Bridge v2 TFDS dataset directory.",
    ),
) -> None:
    """Count episodes in Bridge v2 dataset."""
    stats = count_bridge(path)

    t = Table(title="Bridge v2 Episode Count", show_lines=True)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Dataset path", stats["dataset_path"])
    t.add_row("Total episodes", str(stats["num_episodes"]))
    console.print(t)


@app.command()
def both(
    ecot_path: Path = typer.Option(
        ..., "--ecot-path",
        help="Path to ECoT JSON file or directory.",
    ),
    bridge_path: Path = typer.Option(
        ..., "--bridge-path",
        help="Path to Bridge v2 TFDS dataset directory.",
    ),
) -> None:
    """Count episodes in both datasets and show overlap potential."""
    ecot_stats = count_ecot(ecot_path)
    bridge_stats = count_bridge(bridge_path)

    t = Table(title="Dataset Episode Counts", show_lines=True)
    t.add_column("Dataset", style="bold")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("ECoT", "File paths", str(ecot_stats["num_file_paths"]))
    t.add_row("ECoT", "Total episodes", str(ecot_stats["num_episodes"]))
    t.add_row("ECoT", "Avg episodes/path", str(ecot_stats["avg_episodes_per_path"]))
    t.add_row("Bridge v2", "Total episodes", str(bridge_stats["num_episodes"]))
    console.print(t)

    ecot_n = ecot_stats["num_episodes"]
    bridge_n = bridge_stats["num_episodes"]
    console.print(
        f"\nMax possible matches: [bold]{min(ecot_n, bridge_n)}[/bold] "
        f"(limited by {'ECoT' if ecot_n < bridge_n else 'Bridge v2'})"
    )


if __name__ == "__main__":
    app()
