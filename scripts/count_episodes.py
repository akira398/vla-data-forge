"""
CLI: Count episodes in Bridge v2 (train/test) and ECoT, show overlap.

Usage
-----
python scripts/count_episodes.py \
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


def _resolve_json(path: Path) -> Path:
    if path.is_file() and path.suffix == ".json":
        return path
    candidates = sorted(
        f for f in path.rglob("*.json")
        if not f.name.startswith(".")
        and f.name not in ("dataset_info.json", "dataset_dict.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No .json files found under {path}")
    return candidates[0]


def _count_ecot(json_path: Path) -> tuple[int, set[str]]:
    """Return (total_episodes, set_of_composite_keys)."""
    json_path = _resolve_json(json_path)
    size_mb = json_path.stat().st_size / 1e6
    console.print(f"Loading ECoT from [bold]{json_path.name}[/bold] ({size_mb:.0f} MB)...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from vla_curator.curation.interleaver import _normalize_path

    keys: set[str] = set()
    for file_path, episodes in data.items():
        if not isinstance(episodes, dict):
            continue
        for ep_id_str in episodes:
            composite = _normalize_path(f"{file_path}_{ep_id_str}")
            keys.add(composite)

    return len(keys), keys


def _count_bridge_split(builder, split: str) -> tuple[int, set[str]]:
    """Count episodes and collect composite keys for one split."""
    from vla_curator.curation.interleaver import _normalize_path, _make_composite_key

    split_info = builder.info.splits.get(split)
    if split_info is None:
        return 0, set()

    count = 0
    keys: set[str] = set()
    ds = builder.as_dataset(split=split)

    for ep in ds:
        meta = ep.get("episode_metadata", {})
        fp_raw = meta.get("file_path")
        eid_raw = meta.get("episode_id")

        file_path = ""
        if fp_raw is not None:
            v = fp_raw.numpy() if hasattr(fp_raw, "numpy") else fp_raw
            file_path = v.decode("utf-8") if isinstance(v, bytes) else str(v)

        episode_id = None
        if eid_raw is not None:
            v = eid_raw.numpy() if hasattr(eid_raw, "numpy") else eid_raw
            episode_id = int(v)

        if file_path and episode_id is not None:
            keys.add(_make_composite_key(file_path, episode_id))

        count += 1
        if count % 10000 == 0:
            console.print(f"  [{split}] scanned {count}...")

    return count, keys


@app.callback(invoke_without_command=True)
def main(
    ecot_path: Path = typer.Option(
        ..., "--ecot-path",
        help="Path to ECoT JSON file or directory.",
    ),
    bridge_path: Path = typer.Option(
        ..., "--bridge-path",
        help="Path to Bridge v2 TFDS dataset directory.",
    ),
) -> None:
    """Count episodes in Bridge v2 (train/test), ECoT, and their overlap."""
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets required. pip install 'vla-data-curator[bridge]'"
        ) from exc

    from vla_curator.datasets.bridge_v2 import find_tfds_version_dir

    # --- ECoT ---
    ecot_total, ecot_keys = _count_ecot(ecot_path)

    # --- Bridge v2 ---
    version_dir = find_tfds_version_dir(bridge_path)
    console.print(f"Loading Bridge v2 from [bold]{version_dir}[/bold]...")
    builder = tfds.builder_from_directory(str(version_dir))

    available_splits = list(builder.info.splits.keys())
    console.print(f"Available splits: {available_splits}")

    train_count, train_keys = 0, set()
    test_count, test_keys = 0, set()

    if "train" in available_splits:
        console.print("Scanning [bold]train[/bold] split...")
        train_count, train_keys = _count_bridge_split(builder, "train")

    for split_name in available_splits:
        if split_name != "train":
            console.print(f"Scanning [bold]{split_name}[/bold] split...")
            c, k = _count_bridge_split(builder, split_name)
            test_count += c
            test_keys |= k

    bridge_all_keys = train_keys | test_keys
    bridge_total = train_count + test_count

    # --- Overlap ---
    overlap_train = len(ecot_keys & train_keys)
    overlap_test = len(ecot_keys & test_keys)
    overlap_all = len(ecot_keys & bridge_all_keys)
    ecot_only = len(ecot_keys - bridge_all_keys)

    # --- Table ---
    t = Table(title="Episode Counts", show_lines=True, title_style="bold")
    t.add_column("Dataset", style="bold")
    t.add_column("Count", justify="right")

    t.add_row("Bridge v2 — train", f"{train_count:,}")
    non_train = [s for s in available_splits if s != "train"]
    split_label = ", ".join(non_train) if non_train else "test"
    t.add_row(f"Bridge v2 — {split_label}", f"{test_count:,}")
    t.add_row("Bridge v2 — total", f"[bold]{bridge_total:,}[/bold]")
    t.add_row("", "")
    t.add_row("ECoT — total", f"[bold]{ecot_total:,}[/bold]")
    t.add_row("", "")
    t.add_row("Overlap (ECoT in train)", f"[green]{overlap_train:,}[/green]")
    if test_count > 0:
        t.add_row(f"Overlap (ECoT in {split_label})", f"[green]{overlap_test:,}[/green]")
    t.add_row("Overlap (ECoT in all Bridge)", f"[bold green]{overlap_all:,}[/bold green]")
    t.add_row("ECoT with no Bridge match", f"[red]{ecot_only:,}[/red]")

    console.print()
    console.print(t)

    if ecot_only > 0:
        pct = 100 * ecot_only / max(ecot_total, 1)
        console.print(
            f"\n[yellow]{ecot_only:,} ECoT episodes ({pct:.1f}%) have no match in Bridge v2. "
            f"These may reference a different Bridge v2 version or split.[/yellow]"
        )


if __name__ == "__main__":
    app()
