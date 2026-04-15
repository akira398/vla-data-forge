"""
CLI: Inspect the Embodied-CoT JSON file (embodied_features_bridge.json).

JSON structure:
  {
    "/path/to/bridge_episode/out.npy": [       ← episode key (= Bridge v2 path)
      {                                         ← one dict per step
        "frame_index":          0,
        "language_instruction": "pick up ...",
        "task":                 "...",          ← high-level task
        "plan":                 "...",          ← multi-step plan
        "subtask_reason":       "...",          ← why this subtask
        "subtask":              "...",          ← current subtask label
        "move_reason":          "...",          ← why this motion
        "move":                 "...",          ← motion description
        "gripper":              "...",          ← future gripper positions (5-step lookahead)
        "bboxes":               "...",          ← visible objects + bounding boxes
        "action":               [x,y,z,r,p,y,g] ← 7-DoF action
      },
      ...
    ],
    ...
  }

Usage
-----
# Inspect structure + print first N episodes:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge

# Print more episodes:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --n 10

# Show all steps for the first episode:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --steps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Inspect the Embodied-CoT JSON file.")
console = Console()


def _find_json(path: Path) -> Path:
    if path.is_file() and path.suffix == ".json":
        return path
    candidates = sorted(
        f for f in path.rglob("*.json")
        if not f.name.startswith(".")
        and f.name not in ("dataset_info.json", "dataset_dict.json")
    )
    if not candidates:
        console.print(f"[red]No .json files found under {path}[/red]")
        console.print(f"Contents: {[x.name for x in path.iterdir()]}")
        raise typer.Exit(1)
    return candidates[0]


@app.command()
def main(
    local_path: Path = typer.Option(
        ..., "--local-path", "-p",
        help="Directory containing embodied_features_bridge.json (or path to the file itself).",
    ),
    n: int = typer.Option(5, "--n", help="Number of episodes to show."),
    steps: bool = typer.Option(False, "--steps", help="Print all steps of the first episode."),
) -> None:

    json_file = _find_json(Path(local_path))
    size_mb = json_file.stat().st_size / 1e6
    console.print(f"\n[bold]File:[/bold] {json_file}  ({size_mb:.0f} MB)")
    console.print("Loading… (may take ~30s for the full 1.4 GB file)\n")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── top-level stats ───────────────────────────────────────────────────────
    console.print(f"[bold]Top-level type:[/bold]     {type(data).__name__}")
    console.print(f"[bold]Total episodes:[/bold]     {len(data):,}")

    ep_keys = list(data.keys())
    first_val = data[ep_keys[0]]
    console.print(f"[bold]Value type:[/bold]         {type(first_val).__name__}")

    if isinstance(first_val, list) and first_val:
        console.print(f"[bold]Steps per episode:[/bold]  e.g. {len(first_val)} (first episode)")
        console.print(f"[bold]Step fields:[/bold]        {list(first_val[0].keys())}")
    elif isinstance(first_val, dict):
        console.print(f"[bold]Episode fields:[/bold]     {list(first_val.keys())}")

    # ── episode summary table ─────────────────────────────────────────────────
    console.print()
    t = Table(title=f"First {min(n, len(ep_keys))} episodes", show_lines=True)
    t.add_column("#",          style="cyan",  justify="right", no_wrap=True)
    t.add_column("Episode path (key)",         overflow="fold", max_width=50)
    t.add_column("Steps",      justify="right", no_wrap=True)
    t.add_column("Instruction",                overflow="fold", max_width=40)
    t.add_column("Task",                       overflow="fold", max_width=40)
    t.add_column("Action[0] (7-DoF)",          overflow="fold", max_width=35)

    for i, key in enumerate(ep_keys[:n]):
        val = data[key]
        if isinstance(val, list) and val:
            first_step = val[0]
            instr   = first_step.get("language_instruction", "")[:40]
            task    = first_step.get("task", "")[:40]
            action  = first_step.get("action")
            act_str = str([round(x, 3) for x in action]) if action else "—"
            n_steps = str(len(val))
        elif isinstance(val, dict):
            instr   = val.get("language_instruction", "")[:40]
            task    = val.get("task", "")[:40]
            action  = val.get("action")
            act_str = str([round(x, 3) for x in action]) if action else "—"
            n_steps = "1"
        else:
            instr, task, act_str, n_steps = "", "", "—", "?"
        t.add_row(str(i), key[-50:], n_steps, instr, task, act_str)

    console.print(t)

    # ── all steps for first episode ───────────────────────────────────────────
    if steps:
        key = ep_keys[0]
        val = data[key]
        step_list = val if isinstance(val, list) else [val]

        console.print(f"\n[bold]All steps for episode:[/bold] {key}\n")

        st = Table(title=f"{len(step_list)} steps", show_lines=True)
        st.add_column("frame", justify="right", no_wrap=True)
        st.add_column("action (7-DoF)",    overflow="fold", max_width=40)
        st.add_column("subtask",           overflow="fold", max_width=30)
        st.add_column("move",              overflow="fold", max_width=30)
        st.add_column("task_reason",       overflow="fold", max_width=40)
        st.add_column("bboxes",            overflow="fold", max_width=35)

        for s in step_list:
            action = s.get("action")
            act_str = str([round(x, 3) for x in action]) if action else "—"
            st.add_row(
                str(s.get("frame_index", "?")),
                act_str,
                (s.get("subtask") or "")[:30],
                (s.get("move") or "")[:30],
                (s.get("task") or "")[:40],
                (s.get("bboxes") or "")[:35],
            )
        console.print(st)

        # Full detail of first step
        console.print("\n[bold]Full first step (raw JSON):[/bold]")
        console.print(Panel(json.dumps(step_list[0], indent=2), expand=False))

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
