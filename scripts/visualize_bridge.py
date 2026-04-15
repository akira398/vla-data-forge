"""
CLI: Visualize Bridge v2 episodes.

Usage
-----
# Dual-camera grid for first 3 episodes:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --max-episodes 3

# Save figures to a directory:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --save-dir outputs/viz

# Gripper-state timeline for a single episode:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode gripper

# 7-DoF action component panels:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode actions

# Full summary dashboard:
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode summary

# Save episode GIFs (primary camera):
python scripts/visualize_bridge.py --local-path /datasets/bridge_orig --mode gif --save-dir outputs/gifs
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.config import BridgeV2DatasetConfig
from vla_curator.datasets.bridge_v2 import BridgeV2DatasetReader
from vla_curator.utils.logging import setup_logging
from vla_curator.visualization.bridge_viewer import BridgeViewer

app = typer.Typer(help="Visualize Bridge v2 dataset episodes.")
console = Console()


@app.command()
def main(
    local_path: Path = typer.Option(
        ...,
        "--local-path",
        "-p",
        help="Path to bridge_orig directory (contains 1.0.0/ subdirectory).",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split."),
    max_episodes: int = typer.Option(3, "--max-episodes", "-n", help="Episodes to load."),
    episode_idx: int = typer.Option(0, "--episode-idx", "-e", help="Episode index for single-ep modes."),
    step_idx: int = typer.Option(0, "--step-idx", "-s", help="Step index for dual-camera single-step mode."),
    mode: str = typer.Option(
        "dual",
        "--mode",
        "-m",
        help="Visualization mode: dual | grid | gripper | actions | state | summary | gif",
    ),
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index for GIF mode (0 or 1)."),
    save_dir: Path = typer.Option(
        None, "--save-dir", help="Save figures / GIFs to this directory."
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

    # Summary table
    table = Table(title="Bridge v2 episodes loaded")
    table.add_column("Index", style="cyan")
    table.add_column("Episode ID")
    table.add_column("Steps", justify="right")
    table.add_column("Instruction")

    for i, ep in enumerate(episodes):
        table.add_row(
            str(i),
            ep.episode_id[:50],
            str(len(ep)),
            ep.language_instruction[:60],
        )
    console.print(table)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Saving outputs to [green]{save_dir}[/green]")

    viewer = BridgeViewer()

    for i, ep in enumerate(episodes):
        save_prefix = save_dir / f"ep_{i:04d}" if save_dir else None

        if mode == "dual":
            # Single-step dual-camera view
            sp = Path(f"{save_prefix}_dual_step{step_idx}.png") if save_prefix else None
            console.print(f"\n[bold]Episode {i}[/bold]: {ep.episode_id}")
            viewer.show_dual_camera(ep, step_index=step_idx, save_path=sp)

        elif mode == "grid":
            # Full episode dual-camera grid
            sp = Path(f"{save_prefix}_grid.png") if save_prefix else None
            console.print(f"\n[bold]Episode {i}[/bold]: {ep.episode_id}")
            viewer.show_episode_dual_camera(ep, save_path=sp)

        elif mode == "gripper":
            sp = Path(f"{save_prefix}_gripper.png") if save_prefix else None
            viewer.show_gripper_state(ep, save_path=sp)

        elif mode == "actions":
            sp = Path(f"{save_prefix}_actions.png") if save_prefix else None
            viewer.show_action_components(ep, save_path=sp)

        elif mode == "state":
            sp = Path(f"{save_prefix}_state.png") if save_prefix else None
            viewer.show_state_trajectory(ep, save_path=sp)

        elif mode == "summary":
            sp = Path(f"{save_prefix}_summary.png") if save_prefix else None
            viewer.show_summary(ep, save_path=sp)

        elif mode == "gif":
            if save_dir is None:
                console.print("[red]--save-dir is required for GIF mode.[/red]")
                raise typer.Exit(1)
            gif_path = save_dir / f"ep_{i:04d}_cam{camera}.gif"
            console.print(f"Saving GIF → {gif_path}")
            viewer.save_episode_gif(ep, gif_path, camera=camera)

        else:
            console.print(f"[red]Unknown mode: {mode!r}[/red]")
            raise typer.Exit(1)

    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()
