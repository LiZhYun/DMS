#!/usr/bin/env python3
"""Plot completed development pilots from their saved episode records.

Run from any directory with the repository's existing Python environment:
    .venv/bin/python scripts/plot_pilots.py
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah

Each point averages episodes from one checkpoint of one training seed. There
are no confidence bands: these runs do not estimate across-seed uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    ("dev-mamujoco-halfcheetah-2x3-ippo-101", "HalfCheetah 2x3", "IPPO", 5),
    ("dev-mamujoco-halfcheetah-2x3-dms-soft-101", "HalfCheetah 2x3", "DMS soft", 5),
    ("dev-smac-3s5z-ippo-101", "SMAC 3s5z", "IPPO", 10),
    ("dev-smac-3s5z-dms-soft-101", "SMAC 3s5z", "DMS soft", 10),
]

HISTORICAL_RUNS = [
    ("dev-legacy-halfcheetah-2x3-ippo-101", "HalfCheetah 2x3", "IPPO", 5),
    ("dev-legacy-halfcheetah-2x3-dms-soft-101", "HalfCheetah 2x3", "DMS: original decoder", 5),
    ("dev-legacy-halfcheetah-2x3-dms-latent-decoder-101", "HalfCheetah 2x3", "DMS: latent-only initialization", 5),
]


def read_series(root: Path, name: str, task: str, method: str, episodes: int,
                historical: bool = False) -> dict:
    run = root / "runs" / name
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if config["seed"] != 101 or config["eval_episodes"] != episodes:
        raise ValueError(f"Unexpected training seed or episode count in {run}")
    grouped = defaultdict(list)
    for line in (run / "evaluation.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not math.isfinite(row["return"]):
            raise ValueError(f"Nonfinite evaluation return in {run}")
        grouped[row["steps"]].append(row)
    if not grouped or max(grouped) != config["total_steps"]:
        raise ValueError(f"Final-budget evaluation is missing in {run}")
    checkpoints = []
    for steps, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        if len(rows) != episodes or len({row["seed"] for row in rows}) != episodes:
            raise ValueError(f"Incomplete or repeated evaluation episodes at {run}, step {steps}")
        point = {
            "training_steps": steps,
            "episodes": episodes,
            "episode_seeds": [row["seed"] for row in rows],
            "episode_returns": [row["return"] for row in rows],
            "return_mean": mean(row["return"] for row in rows),
        }
        if task == "SMAC 3s5z":
            outcomes = [row["battle_won"] for row in rows]
            if not all(isinstance(won, bool) for won in outcomes):
                raise ValueError(f"Missing boolean battle outcomes in {run}")
            point.update(episode_wins=outcomes, wins=sum(outcomes), win_rate=mean(outcomes))
        checkpoints.append(point)
    result = {
        "run": name,
        "source": f"runs/{name}/evaluation.jsonl",
        "task": task,
        "method": method,
        "training_seed": config["seed"],
        "training_budget": config["total_steps"],
        "benchmark_track": manifest["environment"]["benchmark_track"],
        "episodes_per_checkpoint": episodes,
        "checkpoints": checkpoints,
    }
    if historical:
        if (config["total_steps"] != 500000 or config.get("scenario") != "HalfCheetah"
                or config.get("agent_conf") != "2x3"
                or not manifest.get("status", "").startswith("completed")
                or "historical" not in result["benchmark_track"]):
            raise ValueError(f"Expected a completed 500k historical HalfCheetah 2x3 pilot in {run}")
        artifacts = [run / "config.json", run / "manifest.json", run / "evaluation.jsonl",
                     run / "source/dms/intention.py", run / "source/dms/train.py"]
        result["provenance"] = {
            "files": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in artifacts},
            "config": config,
            "manifest": manifest,
        }
        if config["algorithm"] == "dms":
            if "decoder_history" in config:
                value, source = config["decoder_history"], "saved config.decoder_history"
            else:
                # Earlier pilots predate the Config field. Verify the saved
                # model default before interpreting the missing declaration.
                saved_model = (run / "source/dms/intention.py").read_text()
                saved_train = (run / "source/dms/train.py").read_text()
                if "decoder_history: bool = True" not in saved_model or "decoder_history=" in saved_train:
                    raise ValueError(f"Cannot establish archived decoder initialization in {run}")
                value, source = True, "archived model default; archived trainer passes no override"
            result["decoder_history"] = value
            result["decoder_history_source"] = source
            result["decoder_interpretation"] = (
                "encoded history and latent initialize the decoder" if value else
                "latent initializes the decoder; previous-action and autoregressive inputs are retained")
    return result


def plot(series: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    styles = {
        "IPPO": {"color": "#C66A22", "linestyle": "--", "marker": "o", "markersize": 6,
                 "markerfacecolor": "white", "markeredgewidth": 1.3},
        "DMS soft": {"color": "#246A9F", "linestyle": "-", "marker": "x", "markersize": 5,
                     "markeredgewidth": 1.5},
    }
    panels = [
        ("HalfCheetah 2x3", "return_mean", "a  HalfCheetah 2x3", "Mean episode return", 1),
        ("SMAC 3s5z", "win_rate", "b  SMAC 3s5z", "Evaluation win rate (%)", 100),
        ("SMAC 3s5z", "return_mean", "c  SMAC 3s5z", "Mean episode return", 1),
    ]
    for axis, (task, metric, title, ylabel, scale) in zip(axes, panels):
        for run in series:
            if run["task"] != task:
                continue
            points = run["checkpoints"]
            axis.plot([p["training_steps"] / 1000 for p in points],
                      [p[metric] * scale for p in points], linewidth=1.7,
                      label=run["method"], **styles[run["method"]])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Training transitions (thousands)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        if task == "HalfCheetah 2x3":
            axis.set_xticks([100, 200, 300, 400, 500])
            axis.text(0.02, 0.96, "Modern MuJoCo adaptation", transform=axis.transAxes,
                      va="top", fontsize=9, color="#555555")
        else:
            axis.set_xticks([50, 100, 150, 200, 250])
        if metric == "win_rate":
            axis.set_ylim(-4, 104)
            axis.set_yticks([0, 25, 50, 75, 100])
    figure.suptitle("Initial development pilots: one training seed (101)", fontsize=14, y=0.96)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.13),
                  ncol=2, frameon=False)
    figure.text(0.5, 0.065,
                "Means over 5 evaluation episodes per HalfCheetah checkpoint; 10 per SMAC checkpoint.\n"
                "Development only: across-seed uncertainty is not estimated; no paper-performance claim.",
                ha="center", va="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.30, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_historical(series: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    styles = [
        {"color": "#C66A22", "linestyle": "--", "marker": "o", "markerfacecolor": "white"},
        {"color": "#246A9F", "linestyle": "-", "marker": "x"},
        {"color": "#7561A8", "linestyle": "-.", "marker": "s", "markerfacecolor": "white"},
    ]
    for run, style in zip(series, styles):
        axis.plot([p["training_steps"] / 1000 for p in run["checkpoints"]],
                  [p["return_mean"] for p in run["checkpoints"]], label=run["method"],
                  linewidth=1.8, markersize=6, markeredgewidth=1.4, **style)
    axis.set_xlabel("Training environment transitions (thousands)")
    axis.set_ylabel("Mean evaluation episode return")
    axis.set_xticks([100, 200, 300, 400, 500])
    axis.margins(y=0.06)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    figure.suptitle("Historical HalfCheetah 2x3 development pilots", fontsize=14, y=0.96)
    figure.text(0.5, 0.89, "Gym v2 compatibility backend | One training seed (101)",
                ha="center", fontsize=10, color="#555555")
    figure.text(0.5, 0.085,
                "Means over 5 evaluation episodes at every saved checkpoint.\n"
                "Single development seed: across-seed uncertainty is not estimated.\n"
                "Not a paper comparison; historical binary parity remains unverified.",
                ha="center", va="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.105, right=0.975, top=0.81, bottom=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--track", choices=["initial", "historical-halfcheetah"], default="initial")
    parser.add_argument("--output", type=Path, help="PNG path; default filename follows --track")
    parser.add_argument("--data", type=Path, help="JSON path; default filename follows --track")
    args = parser.parse_args()
    historical = args.track == "historical-halfcheetah"
    basename = "historical-halfcheetah-pilots" if historical else "initial-pilots"
    output = args.output or args.root / "reports" / f"{basename}.png"
    data_path = args.data or args.root / "reports" / f"{basename}.json"
    runs = HISTORICAL_RUNS if historical else RUNS
    series = [read_series(args.root, *spec, historical=historical) for spec in runs]
    payload = {
        "purpose": ("historical_halfcheetah_single_seed_development_pilots_not_paper_comparison"
                    if historical else "initial_single_seed_development_pilots_not_paper_benchmark"),
        "aggregation": "arithmetic episode means at each recorded checkpoint",
        "uncertainty": "One independent training seed; no confidence intervals or across-seed uncertainty estimated.",
        "checkpoint_selection": "All saved evaluations, including each final planned training budget.",
        "series": series,
    }
    if historical:
        payload["generator"] = {"path": "scripts/plot_pilots.py",
                                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "track": args.track}
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (plot_historical if historical else plot)(series, output)
    print(f"Saved {output} and {data_path}")


if __name__ == "__main__":
    main()
