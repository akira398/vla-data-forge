"""
CLI: Visualize Bridge v2 episodes.

All figures are saved to disk — no display window is opened, making this
safe to run on headless servers.

Usage
-----
# Dual-camera grid (saves PNGs to outputs/viz/bridge):
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig

# Custom output dir and episode limit:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig \\
    --max-episodes 10 --save-dir outputs/my_run

# Full summary dashboard per episode:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode summary

# Save MP4 videos (primary camera):
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode video

# Save GIFs instead:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode gif

Modes: dual | grid | gripper | actions | state | summary | video | gif
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

from vla_curator.config import BridgeV2DatasetConfig
from vla_curator.datasets.bridge_v2 import BridgeV2DatasetReader
from vla_curator.utils.logging import setup_logging
from vla_curator.visualization.bridge_viewer import BridgeViewer

app = typer.Typer(help="Visualize Bridge v2 dataset episodes (headless / server-safe).")
console = Console()

_DEFAULT_SAVE_DIR = Path("outputs/viz/bridge")


@app.command()
def main(
    local_path: Path = typer.Option(
        ...,
        "--local-path",
        "-p",
        help="Path to bridge_orig directory (contains 1.0.0/ subdirectory).",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split."),
    max_episodes: int = typer.Option(5, "--max-episodes", "-n", help="Episodes to load."),
    episode_idx: int = typer.Option(0, "--episode-idx", "-e", help="Episode index for single-ep modes."),
    step_idx: int = typer.Option(0, "--step-idx", "-s", help="Step index for dual-camera single-step mode."),
    max_frames: int = typer.Option(
        16, "--max-frames", "-f",
        help="Max frames sampled per episode for grid/summary/video/gif modes.",
    ),
    mode: str = typer.Option(
        "dual",
        "--mode",
        "-m",
        help="dual | grid | gripper | actions | state | summary | video | gif",
    ),
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index for video/gif (0 or 1)."),
    fps: int = typer.Option(10, "--fps", help="Frames per second for video/gif output."),
    save_dir: Path = typer.Option(
        _DEFAULT_SAVE_DIR, "--save-dir", help="Directory to save all outputs."
    ),
    image_size: str = typer.Option("256x256", "--image-size", help="HxW resize target."),
    include_secondary: bool = typer.Option(
        True, "--include-secondary/--no-secondary", help="Load secondary camera."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    h, w = [int(x) for x in image_size.split("x")]
    cfg = BridgeV2DatasetConfig(
        source="tfds",
        local_path=local_path,
        split=split,
        max_episodes=max_episodes,
        image_size=(h, w),
        include_secondary_camera=include_secondary,
    )

    console.print(
        f"[bold]Loading[/bold] Bridge v2 from [cyan]{local_path}[/cyan] "
        f"(split={split}, max={max_episodes})…"
    )
    reader = BridgeV2DatasetReader(cfg)
    episodes = list(reader)

    if not episodes:
        console.print("[red]No episodes loaded.[/red]")
        raise typer.Exit(1)

    table = Table(title="Bridge v2 episodes loaded")
    table.add_column("Index", style="cyan")
    table.add_column("Episode ID")
    table.add_column("Steps", justify="right")
    table.add_column("Instruction")
    for i, ep in enumerate(episodes):
        table.add_row(str(i), ep.episode_id[:50], str(len(ep)), ep.language_instruction[:60])
    console.print(table)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving outputs to [green]{save_dir}[/green]")

    viewer = BridgeViewer()

    for i, ep in enumerate(episodes):
        prefix = save_dir / f"ep_{i:04d}"

        if mode == "dual":
            viewer.show_dual_camera(ep, step_index=step_idx,
                                    save_path=Path(f"{prefix}_dual_step{step_idx}.png"))

        elif mode == "grid":
            viewer.show_episode_dual_camera(ep, max_frames=max_frames,
                                            save_path=Path(f"{prefix}_grid.png"))

        elif mode == "gripper":
            viewer.show_gripper_state(ep, save_path=Path(f"{prefix}_gripper.png"))

        elif mode == "actions":
            viewer.show_action_components(ep, save_path=Path(f"{prefix}_actions.png"))

        elif mode == "state":
            viewer.show_state_trajectory(ep, save_path=Path(f"{prefix}_state.png"))

        elif mode == "summary":
            viewer.show_summary(ep, max_frames=max_frames,
                                save_path=Path(f"{prefix}_summary.png"))

        elif mode == "video":
            out = Path(f"{prefix}_cam{camera}.mp4")
            console.print(f"  Writing {out.name}…")
            viewer.save_episode_video(ep, out, fps=fps, camera=camera, max_frames=max_frames)

        elif mode == "gif":
            out = Path(f"{prefix}_cam{camera}.gif")
            console.print(f"  Writing {out.name}…")
            viewer.save_episode_gif(ep, out, fps=fps, camera=camera, max_frames=max_frames)

        else:
            console.print(f"[red]Unknown mode: {mode!r}[/red]")
            raise typer.Exit(1)

        console.print(f"  [green]ep {i:04d}[/green] done ({len(ep)} steps)")

    console.print(f"\n[bold green]Done.[/bold green] All outputs in {save_dir}")


if __name__ == "__main__":
    app()
