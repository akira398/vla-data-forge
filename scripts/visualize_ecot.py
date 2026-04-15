"""
CLI: Visualize Embodied-CoT episodes.

Usage
-----
# Show first 3 episodes from the HF dataset (8 frames each):
python scripts/visualize_ecot.py --max-episodes 3

# Save episode grids to PNG files:
python scripts/visualize_ecot.py --max-episodes 5 --save-dir outputs/viz

# Show a single step's full reasoning trace:
python scripts/visualize_ecot.py --episode-idx 0 --step-idx 4 --mode step

# Show trajectory action plots:
python scripts/visualize_ecot.py --max-episodes 2 --mode trajectory
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.config import ECoTDatasetConfig
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.utils.logging import setup_logging
from vla_curator.visualization.frame_viewer import FrameViewer
from vla_curator.visualization.trajectory_viewer import TrajectoryViewer

app = typer.Typer(help="Visualize Embodied-CoT dataset episodes.")
console = Console()


@app.command()
def main(
    hf_repo: str = typer.Option(
        "Embodied-CoT/embodied_features_bridge",
        "--repo",
        help="HuggingFace dataset repo.",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split."),
    max_episodes: int = typer.Option(3, "--max-episodes", "-n", help="Episodes to load."),
    episode_idx: int = typer.Option(0, "--episode-idx", "-e", help="Episode index to show."),
    step_idx: int = typer.Option(0, "--step-idx", "-s", help="Step index for step mode."),
    max_frames: int = typer.Option(8, "--max-frames", help="Max frames per episode grid."),
    mode: str = typer.Option(
        "grid",
        "--mode",
        "-m",
        help="Visualization mode: grid | step | trajectory | summary",
    ),
    save_dir: Path = typer.Option(
        None, "--save-dir", help="Save figures to this directory."
    ),
    no_reasoning: bool = typer.Option(
        False, "--no-reasoning", help="Hide reasoning overlays."
    ),
    image_size: str = typer.Option("256x256", "--image-size", help="HxW."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    h, w = [int(x) for x in image_size.split("x")]
    cfg = ECoTDatasetConfig(
        hf_repo=hf_repo,
        split=split,
        max_episodes=max_episodes,
        image_size=(h, w),
    )

    console.print(f"[bold]Loading[/bold] {hf_repo} (split={split}, max={max_episodes})…")
    reader = ECoTDatasetReader(cfg)
    episodes = reader.take(max_episodes)

    if not episodes:
        console.print("[red]No episodes loaded.[/red]")
        raise typer.Exit(1)

    # Print summary table
    table = Table(title="Episodes loaded")
    table.add_column("Index", style="cyan")
    table.add_column("Episode ID")
    table.add_column("Steps", justify="right")
    table.add_column("Reasoning", justify="right")
    table.add_column("Instruction")

    for i, ep in enumerate(episodes):
        table.add_row(
            str(i),
            ep.episode_id[:40],
            str(len(ep)),
            f"{ep.reasoning_coverage():.0%}",
            ep.language_instruction[:50],
        )
    console.print(table)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Saving figures to [green]{save_dir}[/green]")

    frame_viewer = FrameViewer()
    traj_viewer = TrajectoryViewer()

    for i, ep in enumerate(episodes):
        if mode == "grid":
            sp = (save_dir / f"ep_{i:04d}_grid.png") if save_dir else None
            console.print(f"\n[bold]Episode {i}[/bold]: {ep.episode_id}")
            frame_viewer.show_episode(
                ep,
                max_frames=max_frames,
                show_reasoning=not no_reasoning,
                save_path=sp,
            )

        elif mode == "step":
            idx = step_idx if i == episode_idx else 0
            sp = (save_dir / f"ep_{i:04d}_step_{idx}.png") if save_dir else None
            frame_viewer.show_reasoning_trace(ep, step_index=idx, save_path=sp)

        elif mode == "trajectory":
            sp = (save_dir / f"ep_{i:04d}_trajectory.png") if save_dir else None
            traj_viewer.plot_actions(ep, save_path=sp)

        elif mode == "summary":
            sp = (save_dir / f"ep_{i:04d}_summary.png") if save_dir else None
            traj_viewer.plot_summary(ep, max_frames=max_frames, save_path=sp)

        else:
            console.print(f"[red]Unknown mode: {mode!r}[/red]")
            raise typer.Exit(1)

    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()
