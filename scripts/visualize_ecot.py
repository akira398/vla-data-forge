"""
CLI: Inspect the Embodied-CoT JSON file (embodied_features_bridge.json).

JSON structure (actual on-disk format):
  {
    "/nfs/.../numpy_256/bridge_data_v2/env/task/ep/split/out.npy": {  ← file_path key
      "43": {                                                            ← episode_id (string)
        "metadata": {
          "episode_id":           "43",
          "file_path":            "...",
          "n_steps":              47,
          "language_instruction": "pick up the block",
          "caption":              "scene description"
        },
        "features": {
          "state_3d":          [[x,y,z], ...],       per-step 3D end-effector
          "move_primitive":    ["move forward", ...], per-step motion label
          "gripper_positions": [[x,y], ...]           per-step gripper pixel coords
        },
        "reasoning": {
          "0": {
            "task":           "...",   high-level task description
            "plan":           "...",   multi-step plan
            "subtask":        "...",   current subtask label
            "subtask_reason": "...",   why this subtask
            "move":           "...",   motion description
            "move_reason":    "..."    why this motion
          },
          ...                          sparse — not every step is annotated
        }
      }
    },
    ...
  }

Usage
-----
# Show top-level structure + first N file-path entries:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge

# Show more entries:
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge --n 10

# Show all steps of the first episode:
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
    n: int = typer.Option(5, "--n", help="Number of file-path entries to show."),
    steps: bool = typer.Option(False, "--steps", help="Print all reasoning steps of the first episode."),
) -> None:

    json_file = _find_json(Path(local_path))
    size_mb = json_file.stat().st_size / 1e6
    console.print(f"\n[bold]File:[/bold] {json_file}  ({size_mb:.0f} MB)")
    console.print("Loading… (may take ~30s for the full 1.4 GB file)\n")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── top-level stats ───────────────────────────────────────────────────────
    console.print(f"[bold]Top-level type:[/bold]       {type(data).__name__}")
    console.print(f"[bold]Total file-path keys:[/bold] {len(data):,}")

    fp_keys = list(data.keys())
    first_fp_val = data[fp_keys[0]]

    console.print(f"[bold]Value type:[/bold]           {type(first_fp_val).__name__}")

    if isinstance(first_fp_val, dict):
        ep_ids = list(first_fp_val.keys())
        console.print(f"[bold]Episodes per file:[/bold]    e.g. {len(ep_ids)} (first file)")

        first_ep = first_fp_val[ep_ids[0]]
        if isinstance(first_ep, dict):
            console.print(f"[bold]Episode top keys:[/bold]     {list(first_ep.keys())}")
            if "metadata" in first_ep:
                console.print(f"[bold]Metadata keys:[/bold]        {list(first_ep['metadata'].keys())}")
            if "features" in first_ep:
                console.print(f"[bold]Feature keys:[/bold]         {list(first_ep['features'].keys())}")
            if "reasoning" in first_ep:
                r = first_ep["reasoning"]
                console.print(f"[bold]Reasoning steps:[/bold]      {len(r)} annotated (sparse)")
                if r:
                    sample_r = next(iter(r.values()))
                    console.print(f"[bold]Reasoning fields:[/bold]     {list(sample_r.keys())}")

    # ── file-path summary table ───────────────────────────────────────────────
    console.print()
    t = Table(title=f"First {min(n, len(fp_keys))} file-path entries", show_lines=True)
    t.add_column("#",            style="cyan", justify="right", no_wrap=True)
    t.add_column("File path (key)",            overflow="fold", max_width=55)
    t.add_column("Episodes",     justify="right", no_wrap=True)
    t.add_column("n_steps",      justify="right", no_wrap=True)
    t.add_column("Instruction",                overflow="fold", max_width=40)
    t.add_column("Reasoning steps", justify="right", no_wrap=True)

    for i, fp in enumerate(fp_keys[:n]):
        fp_val = data[fp]
        if not isinstance(fp_val, dict):
            t.add_row(str(i), fp[-55:], "?", "?", "?", "?")
            continue

        num_eps = str(len(fp_val))
        # Use first episode for details
        first_ep_id = next(iter(fp_val))
        ep = fp_val[first_ep_id]
        meta = ep.get("metadata", {}) if isinstance(ep, dict) else {}
        n_steps = str(meta.get("n_steps", "?"))
        instr = (meta.get("language_instruction") or "")[:40]
        reasoning = ep.get("reasoning", {}) if isinstance(ep, dict) else {}
        n_reasoning = str(len(reasoning))

        t.add_row(str(i), fp[-55:], num_eps, n_steps, instr, n_reasoning)

    console.print(t)

    # ── detailed first episode ────────────────────────────────────────────────
    if steps:
        first_fp = fp_keys[0]
        fp_val = data[first_fp]
        if isinstance(fp_val, dict):
            first_ep_id = next(iter(fp_val))
            ep = fp_val[first_ep_id]
            reasoning = ep.get("reasoning", {}) if isinstance(ep, dict) else {}
            meta = ep.get("metadata", {}) if isinstance(ep, dict) else {}
            features = ep.get("features", {}) if isinstance(ep, dict) else {}

            console.print(f"\n[bold]File path:[/bold]   {first_fp}")
            console.print(f"[bold]Episode ID:[/bold]  {first_ep_id}")
            console.print(f"[bold]Instruction:[/bold] {meta.get('language_instruction', '')}")
            console.print(f"[bold]n_steps:[/bold]     {meta.get('n_steps', '?')}")
            console.print(f"[bold]Caption:[/bold]     {(meta.get('caption') or '')[:80]}")

            # Features summary
            console.print()
            console.print("[bold]Features:[/bold]")
            for feat_key, feat_val in features.items():
                sample = feat_val[:2] if isinstance(feat_val, list) and feat_val else feat_val
                console.print(f"  {feat_key}: {len(feat_val) if isinstance(feat_val, list) else '?'} items — sample: {sample}")

            # Reasoning steps table
            if reasoning:
                console.print()
                st = Table(title=f"{len(reasoning)} annotated reasoning steps", show_lines=True)
                st.add_column("step", justify="right", no_wrap=True)
                st.add_column("task",         overflow="fold", max_width=35)
                st.add_column("subtask",      overflow="fold", max_width=25)
                st.add_column("move",         overflow="fold", max_width=25)
                st.add_column("move_reason",  overflow="fold", max_width=35)

                for step_idx, r in list(reasoning.items())[:20]:
                    st.add_row(
                        str(step_idx),
                        (r.get("task") or "")[:35],
                        (r.get("subtask") or "")[:25],
                        (r.get("move") or "")[:25],
                        (r.get("move_reason") or "")[:35],
                    )
                console.print(st)

                # Full detail of first annotated step
                first_step_key = next(iter(reasoning))
                console.print(f"\n[bold]Full reasoning for step {first_step_key} (raw JSON):[/bold]")
                console.print(Panel(json.dumps(reasoning[first_step_key], indent=2), expand=False))

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
