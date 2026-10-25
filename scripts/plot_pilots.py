#!/usr/bin/env python3
"""Plot completed development pilots from their saved episode records.

Run from any directory with the repository's existing Python environment:
    .venv/bin/python scripts/plot_pilots.py
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah-kl
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah-paired

Each point averages episodes from one checkpoint of one training seed. There
are no confidence bands. The paired track is a descriptive development table,
and requires all six 500k runs to finish before producing any output.
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

HISTORICAL_KL_RUNS = [
    (HISTORICAL_RUNS[0][0], "HalfCheetah 2x3", "IPPO", 5),
    (HISTORICAL_RUNS[1][0], "HalfCheetah 2x3", "DMS history init, beta=1", 5),
    (HISTORICAL_RUNS[2][0], "HalfCheetah 2x3", "DMS latent init, beta=1", 5),
    ("dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101",
     "HalfCheetah 2x3", "DMS latent init, beta=0.1", 5),
    ("dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-001-101",
     "HalfCheetah 2x3", "DMS latent init, beta=0.01", 5),
]


def read_series(root: Path, name: str, task: str, method: str, episodes: int,
                historical: bool = False, training_seed: int = 101) -> dict:
    run = root / "runs" / name
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if config["seed"] != training_seed or config["eval_episodes"] != episodes:
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
            "saved_source_sha256": {
                str(path.relative_to(run / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((run / "source").rglob("*.py"))
            },
        }
        result["provenance"]["manifest_matches_saved_source"] = (
            manifest.get("source_sha256") == result["provenance"]["saved_source_sha256"])
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


def source_comparison(series: list[dict]) -> dict:
    """Expose snapshot differences without certifying a development comparison."""
    mappings = [run["provenance"]["saved_source_sha256"] for run in series]
    paths = sorted(set().union(*(mapping.keys() for mapping in mappings)))
    differences = {
        path: {run["run"]: mapping.get(path) for run, mapping in zip(series, mappings)}
        for path in paths if len({mapping.get(path) for mapping in mappings}) > 1
    }
    return {
        "all_saved_source_snapshots_identical": not differences,
        "all_manifests_match_saved_source": all(
            run["provenance"]["manifest_matches_saved_source"] for run in series),
        "differing_files": differences,
        "interpretation": "Recorded source differences are confounders; this is not a matching-source release gate.",
    }


def paired_endpoints(root: Path) -> dict:
    """Read complete paired development endpoints; never substitute partial runs."""
    pairs, series = [], []
    for seed in (101, 103, 107):
        ippo = read_series(root, f"dev-legacy-halfcheetah-2x3-ippo-{seed}",
                           "HalfCheetah 2x3", "IPPO", 5, True, seed)
        dms = read_series(root, f"dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-{seed}",
                         "HalfCheetah 2x3", "DMS latent init, beta=0.1", 5, True, seed)
        left, right = ippo["checkpoints"][-1], dms["checkpoints"][-1]
        if left["episode_seeds"] != right["episode_seeds"]:
            raise ValueError(f"Endpoint evaluation episode seeds differ for training seed {seed}")
        configs = [run["provenance"]["config"] for run in (ippo, dms)]
        if configs[1].get("decoder_history") is not False or configs[1].get("kl_beta") != 0.1:
            raise ValueError(f"Unexpected DMS decoder/KL setting for training seed {seed}")
        differences = {key: {"ippo": configs[0].get(key), "dms": configs[1].get(key)}
                       for key in configs[0].keys() | configs[1].keys()
                       if configs[0].get(key) != configs[1].get(key)}
        if set(differences) - {"algorithm", "output", "decoder_history", "kl_beta"}:
            raise ValueError(f"Unexpected paired configuration differences: {differences}")
        pairs.append({
            "training_seed": seed,
            "training_steps": 500000,
            "episodes_per_method": 5,
            "episode_seeds": left["episode_seeds"],
            "ippo_return_mean": left["return_mean"],
            "dms_return_mean": right["return_mean"],
            "dms_minus_ippo": right["return_mean"] - left["return_mean"],
            "ippo_episode_returns": left["episode_returns"],
            "dms_episode_returns": right["episode_returns"],
            "config_differences": differences,
            "source_comparison": source_comparison([ippo, dms]),
        })
        series.extend([ippo, dms])
    return {
        "purpose": "descriptive_paired_development_endpoints_not_final_hyperparameter_selection",
        "training_seeds": [101, 103, 107],
        "training_budget_per_run": 500000,
        "episodes_per_endpoint": 5,
        "pairing": "Same training-seed labels and evaluation episode seeds; not identical random streams or parameterization.",
        "uncertainty": "Three development seeds, with seed 101 used for selection; no confidence interval or confirmatory claim.",
        "paper_reproduction_pass": False,
        "release_reference_gate_evaluated": False,
        "pairs": pairs,
        "mean_of_training_seed_means": {
            "ippo": mean(row["ippo_return_mean"] for row in pairs),
            "dms": mean(row["dms_return_mean"] for row in pairs),
            "dms_minus_ippo": mean(row["dms_minus_ippo"] for row in pairs),
        },
        "source_comparison": source_comparison(series),
        "series": series,
    }


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


def plot_historical_kl(series: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    colors = ["#555555", "#2C70A3", "#7651A8", "#16846B", "#C66A22"]
    markers = ["o", "x", "s", "D", "^"]
    labels = ["IPPO", "DMS history init\nbeta=1", "DMS latent init\nbeta=1",
              "DMS latent init\nbeta=0.1", "DMS latent init\nbeta=0.01"]
    for index, (run, color, marker) in enumerate(zip(series, colors, markers)):
        points = run["checkpoints"]
        axes[0].plot([point["training_steps"] / 1000 for point in points],
                     [point["return_mean"] for point in points], color=color, marker=marker,
                     linewidth=1.7, markersize=5, label=run["method"])
        endpoint = points[-1]["return_mean"]
        axes[1].scatter(endpoint, index, color=color, marker=marker, s=65, zorder=3)
        axes[1].annotate(f"{endpoint:,.1f}", (endpoint, index), xytext=(8, 0),
                         textcoords="offset points", va="center", color=color, fontsize=10)
    axes[0].set(title="a  All saved checkpoints", xlabel="Training transitions (thousands)",
                ylabel="Mean evaluation episode return", xticks=[100, 200, 300, 400, 500])
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    axes[1].set(title="b  Final 500k checkpoint", xlabel="Mean evaluation episode return",
                yticks=range(len(series)), yticklabels=labels, xlim=(0, 4050), ylim=(4.6, -0.6))
    axes[1].grid(axis="x", alpha=0.2)
    for axis in axes:
        axis.set_axisbelow(True)
    figure.suptitle("Historical HalfCheetah 2x3: decoder and KL development trials", fontsize=14, y=0.97)
    figure.text(0.5, 0.91, "Gym v2 compatibility backend | One training seed (101) | Five evaluation episodes per checkpoint",
                ha="center", fontsize=10, color="#555555")
    figure.text(0.5, 0.075,
                "Latent init retains previous-action and autoregressive inputs. Beta uses the current per-sequence KL reduction.\n"
                "Development only: no across-seed uncertainty or causal mechanism claim; not a paper comparison.",
                ha="center", va="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.07, right=0.975, top=0.82, bottom=0.22, wspace=0.51)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_paired_table(payload: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.axis("off")
    rows = [[str(row["training_seed"]), f'{row["ippo_return_mean"]:,.2f}',
             f'{row["dms_return_mean"]:,.2f}', f'{row["dms_minus_ippo"]:+,.2f}']
            for row in payload["pairs"]]
    overall = payload["mean_of_training_seed_means"]
    rows.append(["Descriptive mean", f'{overall["ippo"]:,.2f}', f'{overall["dms"]:,.2f}',
                 f'{overall["dms_minus_ippo"]:+,.2f}'])
    table = axis.table(cellText=rows,
                       colLabels=["Training seed", "IPPO", "DMS latent init, beta=0.1", "DMS minus IPPO"],
                       cellLoc="center", colWidths=[0.21, 0.17, 0.37, 0.25], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#EAF0F5")
            cell.set_text_props(weight="bold")
    figure.suptitle("Historical HalfCheetah 2x3: paired development endpoints", fontsize=14, y=0.96)
    figure.text(0.5, 0.86, "500,000 transitions per run | Five evaluation episodes per trained seed", ha="center")
    source_note = ("Source snapshots differ; per-file differences are recorded in the JSON."
                   if not payload["source_comparison"]["all_saved_source_snapshots_identical"] else
                   "Saved source snapshots match across these runs.")
    figure.text(0.5, 0.10,
                "Development seeds 101/103/107; seed 101 informed setting selection. No confirmatory or paper-performance claim.\n"
                + source_note, ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.04, right=0.96, top=0.79, bottom=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--track", choices=["initial", "historical-halfcheetah",
                        "historical-halfcheetah-kl", "historical-halfcheetah-paired"], default="initial")
    parser.add_argument("--output", type=Path, help="PNG path; default filename follows --track")
    parser.add_argument("--data", type=Path, help="JSON path; default filename follows --track")
    args = parser.parse_args()
    historical = args.track != "initial"
    basenames = {"initial": "initial-pilots", "historical-halfcheetah": "historical-halfcheetah-pilots",
                 "historical-halfcheetah-kl": "historical-halfcheetah-kl-pilots",
                 "historical-halfcheetah-paired": "historical-halfcheetah-beta01-paired"}
    basename = basenames[args.track]
    output = args.output or args.root / "reports" / f"{basename}.png"
    data_path = args.data or args.root / "reports" / f"{basename}.json"
    if args.track == "historical-halfcheetah-paired":
        # Build everything before writing, so an unfinished run produces no
        # misleading partial endpoint artifact.
        payload = paired_endpoints(args.root)
        payload["generator"] = {"path": "scripts/plot_pilots.py", "track": args.track,
                                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        plot_paired_table(payload, output)
        print(f"Saved {output} and {data_path}")
        return
    runs = HISTORICAL_KL_RUNS if args.track == "historical-halfcheetah-kl" else (
        HISTORICAL_RUNS if historical else RUNS)
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
        payload["source_comparison"] = source_comparison(series)
        payload["generator"] = {"path": "scripts/plot_pilots.py",
                                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "track": args.track}
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    render = plot_historical_kl if args.track == "historical-halfcheetah-kl" else (
        plot_historical if historical else plot)
    render(series, output)
    print(f"Saved {output} and {data_path}")


if __name__ == "__main__":
    main()
