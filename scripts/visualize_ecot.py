"""
CLI: Visualize / inspect Embodied-CoT reasoning annotations.

The dataset (embodied_features_bridge.json) contains reasoning traces keyed
by Bridge episode paths.  There are no images in this dataset — visualization
shows reasoning text and episode summaries.

Usage
-----
# Inspect raw JSON structure of the first few entries:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --mode inspect

# Print a table of loaded episodes:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --mode table

# Save reasoning trace figures (one PNG per episode):
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --mode reasoning

Modes: inspect | table | reasoning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.config import ECoTDatasetConfig
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.utils.logging import setup_logging

app = typer.Typer(help="Inspect Embodied-CoT reasoning annotations.")
console = Console()

_DEFAULT_SAVE_DIR = Path("outputs/viz/ecot")


@app.command()
def main(
    local_path: Path = typer.Option(
        None, "--local-path", "-p",
        help="Path to local download directory (contains embodied_features_bridge.json).",
    ),
    hf_repo: str = typer.Option(
        "Embodied-CoT/embodied_features_bridge", "--repo",
        help="HF repo (used only when --local-path is not set).",
    ),
    max_episodes: int = typer.Option(10, "--max-episodes", "-n", help="Number of episodes to load."),
    mode: str = typer.Option(
        "inspect", "--mode", "-m",
        help="inspect | table | reasoning",
    ),
    save_dir: Path = typer.Option(_DEFAULT_SAVE_DIR, "--save-dir", help="Output directory for figures."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    source = str(local_path) if local_path else hf_repo
    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Mode:[/bold]   {mode}  |  max_episodes={max_episodes}")

    # ── inspect: show raw JSON structure before any parsing ──────────────────
    if mode == "inspect":
        _raw_inspect(local_path, max_entries=3)
        return

    # ── load episodes ─────────────────────────────────────────────────────────
    cfg = ECoTDatasetConfig(
        hf_repo=hf_repo,
        local_path=local_path,
        max_episodes=max_episodes,
    )
    reader = ECoTDatasetReader(cfg)

    console.print(f"\nLoading up to {max_episodes} episodes…")
    episodes = list(reader)

    if not episodes:
        console.print("[red]No episodes loaded.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(episodes)} episodes.[/green]\n")

    # ── table: summary of all loaded episodes ─────────────────────────────────
    if mode in ("inspect", "table"):
        t = Table(title="Embodied-CoT episodes")
        t.add_column("#", style="cyan", justify="right")
        t.add_column("Episode ID (path)")
        t.add_column("Steps", justify="right")
        t.add_column("Has reasoning", justify="center")
        t.add_column("Task reasoning (preview)")
        for i, ep in enumerate(episodes):
            has_r = "✓" if ep.has_any_reasoning() else "✗"
            task_r = ""
            if ep.steps and ep.steps[0].reasoning:
                task_r = (ep.steps[0].reasoning.task_reasoning or "")[:60]
            t.add_row(str(i), ep.episode_id[-60:], str(len(ep.steps)), has_r, task_r)
        console.print(t)

    # ── reasoning: save one PNG per episode with all 6 fields ─────────────────
    elif mode == "reasoning":
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Saving figures to [green]{save_dir}[/green]")

        for i, ep in enumerate(episodes):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.axis("off")

            step = ep.steps[0] if ep.steps else None
            r = step.reasoning if step else None

            lines = [f"Episode: {ep.episode_id}", ""]
            if r:
                for field in [
                    "task_reasoning", "subtask_reasoning", "move_reasoning",
                    "gripper_reasoning", "attribute_reasoning", "spatial_reasoning",
                ]:
                    val = getattr(r, field, "") or "(empty)"
                    lines.append(f"[{field}]")
                    lines.append(f"  {val[:120]}")
                    lines.append("")
            else:
                lines.append("(no reasoning data)")

            ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes,
                    va="top", fontsize=8, family="monospace")

            out = save_dir / f"ep_{i:04d}_reasoning.png"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            console.print(f"  saved {out.name}")

    else:
        console.print(f"[red]Unknown mode: {mode!r}. Choose: inspect | table | reasoning[/red]")
        raise typer.Exit(1)

    console.print("\n[bold green]Done.[/bold green]")


def _raw_inspect(local_path: Path | None, max_entries: int = 3) -> None:
    """Print the raw JSON structure of the first few entries without any parsing."""
    if local_path is None:
        console.print("[yellow]--local-path not set; cannot inspect raw file.[/yellow]")
        return

    json_files = sorted(
        f for f in Path(local_path).rglob("*.json")
        if not f.name.startswith(".")
        and f.name not in ("dataset_info.json", "dataset_dict.json")
    )
    if not json_files:
        console.print(f"[red]No .json files found under {local_path}[/red]")
        console.print(f"Contents: {[f.name for f in Path(local_path).iterdir()]}")
        return

    json_file = json_files[0]
    console.print(f"\nFile: [cyan]{json_file}[/cyan]  ({json_file.stat().st_size / 1e6:.0f} MB)")
    console.print("Loading first few bytes to detect structure…\n")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"Top-level type: [bold]{type(data).__name__}[/bold]")

    if isinstance(data, dict):
        keys = list(data.keys())
        console.print(f"Number of entries: [bold]{len(keys)}[/bold]")
        console.print(f"\nFirst {max_entries} keys:")
        for k in keys[:max_entries]:
            console.print(f"  [cyan]{k}[/cyan]")
        console.print(f"\nValue type for first key: [bold]{type(data[keys[0]]).__name__}[/bold]")
        console.print("\nFirst entry value:")
        console.print(Panel(json.dumps(data[keys[0]], indent=2)[:1000], expand=False))

    elif isinstance(data, list):
        console.print(f"Number of entries: [bold]{len(data)}[/bold]")
        console.print(f"\nFirst entry:")
        console.print(Panel(json.dumps(data[0], indent=2)[:1000], expand=False))

    else:
        console.print(f"[yellow]Unexpected type: {type(data)}[/yellow]")


if __name__ == "__main__":
    app()
