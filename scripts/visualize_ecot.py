"""
CLI: Visualize Embodied-CoT episodes.

All figures are saved to disk — no display window is opened, making this
safe to run on headless servers.

Usage
-----
# Save frame grids for first 5 episodes (default):
python scripts/visualize_ecot.py

# Custom repo / limit:
python scripts/visualize_ecot.py --max-episodes 10 --save-dir outputs/my_run

# Full reasoning trace for step 4 of episode 0:
python scripts/visualize_ecot.py --episode-idx 0 --step-idx 4 --mode step

# Trajectory action plots:
python scripts/visualize_ecot.py --max-episodes 5 --mode trajectory

# Save MP4 videos:
python scripts/visualize_ecot.py --max-episodes 5 --mode video

Modes: grid | step | trajectory | summary | video | gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must come before any other matplotlib import

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.config import ECoTDatasetConfig
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.utils.logging import setup_logging
from vla_curator.visualization.frame_viewer import FrameViewer
from vla_curator.visualization.trajectory_viewer import TrajectoryViewer

app = typer.Typer(help="Visualize Embodied-CoT dataset episodes (headless / server-safe).")
console = Console()

_DEFAULT_SAVE_DIR = Path("outputs/viz/ecot")


@app.command()
def main(
    hf_repo: str = typer.Option(
        "Embodied-CoT/embodied_features_bridge",
        "--repo",
        help="HuggingFace dataset repo.",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split."),
    max_episodes: int = typer.Option(5, "--max-episodes", "-n", help="Episodes to load."),
    episode_idx: int = typer.Option(0, "--episode-idx", "-e", help="Episode index for step mode."),
    step_idx: int = typer.Option(0, "--step-idx", "-s", help="Step index for step mode."),
    max_frames: int = typer.Option(
        16, "--max-frames", "-f",
        help="Max frames sampled per episode for grid/summary/video/gif modes.",
    ),
    fps: int = typer.Option(10, "--fps", help="Frames per second for video/gif output."),
    mode: str = typer.Option(
        "grid",
        "--mode",
        "-m",
        help="grid | step | trajectory | summary | video | gif",
    ),
    save_dir: Path = typer.Option(
        _DEFAULT_SAVE_DIR, "--save-dir", help="Directory to save all outputs."
    ),
    no_reasoning: bool = typer.Option(
        False, "--no-reasoning", help="Hide reasoning overlays in grid mode."
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

    table = Table(title="Episodes loaded")
    table.add_column("Index", style="cyan")
    table.add_column("Episode ID")
    table.add_column("Steps", justify="right")
    table.add_column("Reasoning", justify="right")
    table.add_column("Instruction")
    for i, ep in enumerate(episodes):
        table.add_row(
            str(i), ep.episode_id[:40], str(len(ep)),
            f"{ep.reasoning_coverage():.0%}", ep.language_instruction[:50],
        )
    console.print(table)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving outputs to [green]{save_dir}[/green]")

    frame_viewer = FrameViewer()
    traj_viewer = TrajectoryViewer()

    for i, ep in enumerate(episodes):
        prefix = save_dir / f"ep_{i:04d}"

        if mode == "grid":
            frame_viewer.show_episode(
                ep,
                max_frames=max_frames,
                show_reasoning=not no_reasoning,
                save_path=Path(f"{prefix}_grid.png"),
            )

        elif mode == "step":
            idx = step_idx if i == episode_idx else 0
            frame_viewer.show_reasoning_trace(
                ep, step_index=idx,
                save_path=Path(f"{prefix}_step{idx}.png"),
            )

        elif mode == "trajectory":
            traj_viewer.plot_actions(ep, save_path=Path(f"{prefix}_trajectory.png"))

        elif mode == "summary":
            traj_viewer.plot_summary(
                ep, max_frames=max_frames,
                save_path=Path(f"{prefix}_summary.png"),
            )

        elif mode == "video":
            out = Path(f"{prefix}.mp4")
            console.print(f"  Writing {out.name}…")
            frame_viewer.save_episode_video(ep, out, fps=fps, max_frames=max_frames)

        elif mode == "gif":
            out = Path(f"{prefix}.gif")
            console.print(f"  Writing {out.name}…")
            frame_viewer.save_episode_gif(ep, out, fps=fps, max_frames=max_frames)

        else:
            console.print(f"[red]Unknown mode: {mode!r}[/red]")
            raise typer.Exit(1)

        console.print(f"  [green]ep {i:04d}[/green] done ({len(ep)} steps)")

    console.print(f"\n[bold green]Done.[/bold green] All outputs in {save_dir}")


if __name__ == "__main__":
    app()
