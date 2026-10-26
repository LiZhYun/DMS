#!/usr/bin/env python3
"""Plot completed development pilots from their saved episode records.

Run from any directory with the repository's existing Python environment:
    .venv/bin/python scripts/plot_pilots.py
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah-kl
    .venv/bin/python scripts/plot_pilots.py --track historical-halfcheetah-paired
    .venv/bin/python scripts/plot_pilots.py --track halfcheetah-10m-progress

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
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import os
import platform

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


def complete_evaluations(records: list[dict], episodes: int = 10,
                         first_seed: int = 10001) -> tuple[list[dict], list[dict]]:
    """Keep complete checkpoint groups; retain excluded groups in the snapshot."""
    grouped = defaultdict(list)
    expected = list(range(first_seed, first_seed + episodes))
    for row in records:
        if (not isinstance(row["steps"], int) or row["steps"] <= 0
                or not math.isfinite(row["return"])):
            raise ValueError("Invalid evaluation step or return")
        grouped[row["steps"]].append(row)
    complete, excluded = [], []
    for steps, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["seed"])
        seeds = [row["seed"] for row in rows]
        if len(set(seeds)) != len(seeds) or set(seeds) - set(expected):
            raise ValueError(f"Repeated or unexpected evaluation seeds at step {steps}")
        if seeds != expected:
            excluded.append({"training_steps": steps, "reason": "incomplete episode group",
                             "observed_episode_seeds": seeds, "required_episode_seeds": expected})
            continue
        complete.append({"training_steps": steps, "episodes": episodes,
                         "episode_seeds": seeds,
                         "return_mean": mean(row["return"] for row in rows)})
    return complete, excluded


def read_jsonl_snapshot(path: Path) -> tuple[list[dict], dict]:
    """Read one append-only prefix; an unfinished final line is never a record."""
    exists = path.exists()
    raw = path.read_bytes() if exists else b""
    end = raw.rfind(b"\n") + 1
    prefix, tail = raw[:end], raw[end:]
    rows = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    return rows, {"exists": exists, "captured_bytes": len(raw), "complete_prefix_bytes": end,
                  "complete_record_count": len(rows), "partial_line_count": int(bool(tail)),
                  "captured_sha256": hashlib.sha256(raw).hexdigest(),
                  "complete_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                  "unparsed_tail": tail.decode("utf-8")}


def progress_series(root: Path, job: dict, role: str) -> dict:
    run = (root / job["output"]).resolve()
    config_bytes = (run / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if (config["total_steps"] != 10_000_000 or config["eval_every"] != 1_000_000
            or config["eval_episodes"] != 10 or config["eval_seed"] != 10001
            or config["scenario"] != "HalfCheetah" or config["agent_conf"] != "2x3"
            or config["mamujoco_backend"] != "legacy"):
        raise ValueError(f"Unexpected 10M development protocol in {run}")
    # Training metrics precede evaluation writes. Reading evaluation first means
    # the following metrics prefix can account for every captured checkpoint.
    records, evaluation_prefix = read_jsonl_snapshot(run / "evaluation.jsonl")
    metrics, metrics_prefix = read_jsonl_snapshot(run / "metrics.jsonl")
    points, excluded = complete_evaluations(records)
    observed_steps = metrics[-1]["steps"] if metrics else 0
    if observed_steps > config["total_steps"] or any(row["steps"] > observed_steps for row in records):
        raise ValueError(f"Evaluation exceeds observed training progress in {run}")
    if any(row["steps"] % config["eval_every"] for row in records):
        raise ValueError(f"Unexpected evaluation checkpoint in {run}")
    manifest_bytes = (run / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    sources = {str(path.relative_to(run / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((run / "source").rglob("*.py"))}
    if sources != manifest["source_sha256"]:
        raise ValueError(f"Manifest/source mismatch in {run}")
    assets = {str(path.relative_to(run / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((run / "source").rglob("*.xml"))}
    return {"run": run.name, "path": str(run.relative_to(root.resolve())),
            "job_id": str(job["job_id"]), "role": role, "algorithm": config["algorithm"],
            "training_seed": config["seed"], "planned_training_steps": config["total_steps"],
            "observed_training_steps": observed_steps,
            "last_complete_evaluation_steps": points[-1]["training_steps"] if points else None,
            "fixed_endpoint_available": bool(points and points[-1]["training_steps"] == 10_000_000),
            "manifest_status": manifest["status"], "checkpoints": points,
            "raw_episode_records": records, "excluded_incomplete_groups": excluded,
            "complete_checkpoint_count": len(points), "incomplete_checkpoint_count": len(excluded),
            "excluded_episode_record_count": sum(len(group["observed_episode_seeds"]) for group in excluded),
            "last_observed_training_record": metrics[-1] if metrics else None,
            "captured_training_record_count": len(metrics),
            "provenance": {"config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                           "manifest": manifest, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                           "evaluation_prefix": evaluation_prefix, "metrics_prefix": metrics_prefix,
                           "saved_source_sha256": sources, "saved_assets_sha256": assets,
                           "manifest_matches_saved_source": True}}


def halfcheetah_progress(root: Path) -> dict:
    """Capture all seven predeclared runs, without averaging training seeds."""
    started = datetime.now(timezone.utc).isoformat()
    ledger_names = ["development-halfcheetah-10m-seed101.json",
                    "development-halfcheetah-10m-replications.json",
                    "development-halfcheetah-aux-lr-10m-seed101.json"]
    ledgers, ledger_hashes = [], {}
    for name in ledger_names:
        raw = (root / "reports" / name).read_bytes()
        ledgers.append(json.loads(raw))
        ledger_hashes[f"reports/{name}"] = hashlib.sha256(raw).hexdigest()
    series = [progress_series(root, job, "matched")
              for ledger in ledgers[:2] for job in ledger["jobs"]]
    series.append(progress_series(root, ledgers[2], "lower_aux_lr"))
    matched = [run for run in series if run["role"] == "matched"]
    if {(run["training_seed"], run["algorithm"]) for run in matched} != {
            (seed, method) for seed in (101, 103, 107) for method in ("ippo", "dms")} or len(matched) != 6:
        raise ValueError("Expected exactly three matched IPPO/DMS training-seed pairs")
    pairs = []
    for seed in (101, 103, 107):
        pair = sorted((run for run in matched if run["training_seed"] == seed), key=lambda r: r["algorithm"])
        configs = [run["provenance"]["config"] for run in pair]
        differences = {key: [config.get(key) for config in configs]
                       for key in configs[0].keys() | configs[1].keys()
                       if configs[0].get(key) != configs[1].get(key)}
        if set(differences) - {"algorithm", "output"}:
            raise ValueError(f"Unexpected paired settings for seed {seed}: {differences}")
        pairs.append({"training_seed": seed, "run_order": [run["run"] for run in pair],
                      "config_differences": differences, "source_comparison": source_comparison(pair)})
    lower = series[-1]
    if (lower["algorithm"] != "dms" or lower["training_seed"] != 101
            or lower["provenance"]["config"]["intention_lr"] != 0.0001):
        raise ValueError("Unexpected lower-auxiliary-learning-rate condition")
    for run in series:
        config = run["provenance"]["config"]
        expected_lr = 0.0001 if run["role"] == "lower_aux_lr" else 0.001
        if config["algorithm"] == "dms" and (config["intention_lr"] != expected_lr
                or config["decoder_history"] is not False or config["kl_beta"] != 0.1):
            raise ValueError("Unexpected standard DMS setting")
    return {"kind": "halfcheetah_10m_development_progress", "schema_version": 1,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "capture_started_at_utc": started,
            "capture_semantics": "One fixed snapshot assembled from captured append-only prefixes; files are not read atomically together. All subsequent rendering uses this JSON only.",
            "planned_endpoint_steps": 10_000_000, "training_seeds": [101, 103, 107],
            "diagnostic_episode_seeds": list(range(10001, 10011)),
            "aggregation": "Arithmetic mean of exactly ten diagnostic episodes at one checkpoint of one training seed; no cross-seed aggregation or interpolation.",
            "uncertainty": "No uncertainty band or estimate of training-seed uncertainty. Diagnostic episodes are not independent trained seeds.",
            "selection": "All complete recorded checkpoint groups; incomplete groups and unfinished lines are retained but not plotted. Fixed10M endpoint, no best-checkpoint selection.",
            "interpretation": "Development only; seed101 informed setting selection. Historical binary parity is unverified and no paper-performance or final reproduction claim is made.",
            "ledger_sha256": ledger_hashes, "pairs": pairs, "series": series,
            "source_comparison": source_comparison(series),
            "generator": {"path": "scripts/plot_pilots.py", "track": "halfcheetah-10m-progress",
                          "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "executed_path": str(Path(__file__).resolve()),
                          "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node()}}


def plot_progress(payload: dict, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=True, sharey=True)
    series = payload["series"]
    panels = [[run for run in series if run["role"] == "matched" and run["training_seed"] == seed]
              for seed in (101, 103, 107)]
    panels.append([run for run in series if run["training_seed"] == 101 and run["algorithm"] == "dms"])
    titles = ["a  Training seed 101", "b  Training seed 103", "c  Training seed 107",
              "d  Auxiliary learning rate: seed 101"]
    all_returns = [point["return_mean"] for run in series for point in run["checkpoints"]]
    low, high = min(all_returns, default=0), max(all_returns, default=1)
    span = max(high - low, 1)
    for axis, runs, title in zip(axes.flat, panels, titles):
        for run in sorted(runs, key=lambda row: (row["algorithm"] != "ippo", row["role"] == "lower_aux_lr")):
            ippo, lower = run["algorithm"] == "ippo", run["role"] == "lower_aux_lr"
            color = "#C66A22" if ippo else ("#16846B" if lower else "#246A9F")
            label = "IPPO" if ippo else ("DMS aux lr=1e-4" if lower else "DMS aux lr=1e-3")
            label += f" (logged {run['observed_training_steps'] / 1e6:.2f}M)"
            points = run["checkpoints"]
            axis.plot([point["training_steps"] / 1e6 for point in points],
                      [point["return_mean"] for point in points], color=color,
                      linestyle="--" if ippo else "-", marker="o" if ippo else ("^" if lower else "s"),
                      markerfacecolor="white" if ippo else color, markersize=4.5, linewidth=1.8,
                      label=label)
        axis.axvline(10, color="#888888", linewidth=0.9, linestyle=":", zorder=0)
        axis.set(title=title, xlim=(0, 10.2), ylim=(low - 0.08 * span, high + 0.25 * span),
                 xticks=[0, 2, 4, 6, 8, 10], xlabel="Training transitions (millions)",
                 ylabel="Mean diagnostic episode return")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        axis.tick_params(labelbottom=True)
    captured = datetime.fromisoformat(payload["captured_at_utc"]).strftime("%Y-%m-%d %H:%M:%S")
    figure.suptitle("HalfCheetah 2x3: fixed 10M development budget", fontsize=15, y=0.98)
    figure.text(0.5, 0.94, f"Snapshot {captured} UTC | Historical Gym v2 compatibility backend | DMS latent decoder, beta=0.1",
                ha="center", fontsize=10, color="#555555")
    figure.text(0.5, 0.075,
                "Each point: ten diagnostic episodes, seeds 10001-10010. Legends show logged training progress; incomplete evaluations are omitted.\n"
                "Training seeds remain separate; no cross-seed mean or uncertainty bands. Dotted line marks the fixed 10M endpoint.\n"
                "Development only: seed 101 informed setting selection; historical binary parity and paper-performance claims remain unverified.",
                ha="center", va="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.18, hspace=0.33, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


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
                        "historical-halfcheetah-kl", "historical-halfcheetah-paired",
                        "halfcheetah-10m-progress"], default="initial")
    parser.add_argument("--output", type=Path, help="PNG path; default filename follows --track")
    parser.add_argument("--data", type=Path, help="JSON path; default filename follows --track")
    parser.add_argument("--snapshot", type=Path,
                        help="Render an existing 10M progress JSON without reading live runs")
    args = parser.parse_args()
    if args.snapshot and args.track != "halfcheetah-10m-progress":
        parser.error("--snapshot applies only to --track halfcheetah-10m-progress")
    historical = args.track != "initial"
    basenames = {"initial": "initial-pilots", "historical-halfcheetah": "historical-halfcheetah-pilots",
                 "historical-halfcheetah-kl": "historical-halfcheetah-kl-pilots",
                 "historical-halfcheetah-paired": "historical-halfcheetah-beta01-paired",
                 "halfcheetah-10m-progress": "halfcheetah-10m-development-progress"}
    basename = basenames[args.track]
    output = args.output or args.root / "reports" / f"{basename}.png"
    data_path = args.data or args.root / "reports" / f"{basename}.json"
    if args.track == "halfcheetah-10m-progress":
        if output.exists() or (not args.snapshot and data_path.exists()):
            raise FileExistsError("Choose new output paths to preserve fixed snapshots")
        if args.snapshot:
            payload = json.loads(args.snapshot.read_text())
            if payload.get("kind") != "halfcheetah_10m_development_progress":
                raise ValueError("Expected a 10M HalfCheetah progress snapshot")
        else:
            payload = halfcheetah_progress(args.root)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with data_path.open("x") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        plot_progress(payload, output)
        print(f"Saved {output}; snapshot: {args.snapshot or data_path}")
        return
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
