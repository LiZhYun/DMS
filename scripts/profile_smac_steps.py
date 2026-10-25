#!/usr/bin/env python3
"""Bounded SC2 stepping profile, separate from training and benchmark results.

Four independently seeded games receive deterministic, legal scripted actions.
Only env.step is threaded; launches, resets, history, and model calls remain on
the main thread. Both modes use at most 2,048 environment transitions each.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from dms.smac_env import SMACEnv
from dms.train import Config, Learner, advance_history, seed_everything


FIELDS = ("obs", "rewards", "terminated", "truncated", "action_mask", "observed_actions", "visibility")


def scripted_actions(states, step):
    actions = []
    for e, state in enumerate(states):
        row = []
        for agent, available in enumerate(state.action_mask):
            legal = np.flatnonzero(available)
            if not len(legal):
                raise RuntimeError("Environment returned no legal action")
            row.append(legal[(step + 3 * agent + 5 * e) % len(legal)])
        actions.append(np.asarray(row, dtype=np.int64))
    return actions


def timed_step(env, action):
    started = time.perf_counter()
    state = env.step(action)
    return state, time.perf_counter() - started


def timing(samples):
    return {"count": len(samples), "total_seconds": float(sum(samples)),
            "mean_ms": float(np.mean(samples) * 1000),
            "median_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.quantile(samples, 0.95) * 1000)}


def model_profile(env, states, history, visibility, seed):
    """Fixed CPU DMS inputs; no simulator steps, PPO, or parameter updates."""
    config = Config(env="smac", algorithm="dms", device="cpu", num_envs=4,
                    warmup_episodes=0, seed=seed, torch_threads=1)
    seed_everything(seed)
    learner = Learner(config, env)
    contexts, actions = [], []
    for index in range(60):
        started = time.perf_counter()
        inputs = learner.contexts(states, history, visibility, use_intentions=True)
        context_time = time.perf_counter() - started
        started = time.perf_counter()
        learner.action(*inputs, deterministic=True)
        action_time = time.perf_counter() - started
        if index >= 10:
            contexts.append(context_time)
            actions.append(action_time)
    return {"config": asdict(config), "warmup_calls": 10, "measured_calls": 50,
            "purpose": "untrained_fixed_batch_cpu_inference_only_not_training_throughput",
            "contexts": timing(contexts), "action": timing(actions),
            "valid_history_entries": int(visibility.sum())}


def run_mode(mode, args, include_model=False):
    environments, pool = [], None
    trace, batch_times, individual_times, reset_times = [], [], [], []
    resets = [0] * 4
    seed_everything(args.seed)
    startup = time.perf_counter()
    try:
        for index in range(4):
            environments.append(SMACEnv(args.map_name, seed=args.seed + 1000 * index))
        states = [env.reset() for env in environments]
        startup_seconds = time.perf_counter() - startup
        trace.append({"kind": "initial", "states": states})
        history = np.zeros((4, 32, environments[0].n_agents, environments[0].n_agents,
                            environments[0].action_dim), dtype=np.float32)
        visible = np.zeros(history.shape[:-1], dtype=bool)
        if mode == "threads":
            pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="smac-profile")
        for step in range(args.steps_per_env):
            if step == args.warmup_steps:
                wall_started = time.perf_counter()
            actions = scripted_actions(states, step)
            started = time.perf_counter()
            if pool is None:
                results = [timed_step(env, action) for env, action in zip(environments, actions)]
            else:
                futures = [pool.submit(timed_step, env, action)
                           for env, action in zip(environments, actions)]
                # Drain every submitted task before propagating an exception.
                # Closing a controller while another worker uses it is unsafe.
                wait(futures)
                results = [future.result() for future in futures]
            duration = time.perf_counter() - started
            if step >= args.warmup_steps:
                batch_times.append(duration)
                individual_times.extend(row[1] for row in results)
            next_states = [row[0] for row in results]
            trace.append({"kind": "step", "step": step, "actions": actions, "states": next_states.copy()})
            advance_history(history, visible, next_states)
            for index, state in enumerate(next_states):
                if state.episode_done:
                    started = time.perf_counter()
                    next_states[index] = environments[index].reset()
                    if step >= args.warmup_steps:
                        reset_times.append(time.perf_counter() - started)
                    resets[index] += 1
                    history[index] = 0
                    visible[index] = False
                    trace.append({"kind": "reset", "step": step, "environment": index,
                                  "states": [next_states[index]]})
            states = next_states
        wall_seconds = time.perf_counter() - wall_started
        measured_transitions = 4 * (args.steps_per_env - args.warmup_steps)
        result = {
            "mode": mode, "startup_seconds": startup_seconds,
            "simulated_transitions": 4 * args.steps_per_env,
            "measured_transitions": measured_transitions,
            "step_batch": timing(batch_times), "individual_step": timing(individual_times),
            "serial_resets": timing(reset_times) if reset_times else {"count": 0, "total_seconds": 0},
            "step_only_transitions_per_second": measured_transitions / sum(batch_times),
            "instrumented_loop_seconds": wall_seconds,
            "instrumented_loop_transitions_per_second": measured_transitions / wall_seconds,
            "completed_episodes_per_environment": resets,
            "native_force_restarts": [env._env.force_restarts for env in environments],
            "environment": environments[0].metadata,
        }
        if include_model:
            result["cpu_dms_inference"] = model_profile(environments[0], states, history, visible, args.seed)
        return result, trace
    finally:
        # Futures cannot forcibly cancel a running RPC. PySC2's socket timeout
        # remains the bound; always join workers before closing their games.
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        close_errors = []
        for env in environments:
            try:
                env.close()
            except Exception as error:
                close_errors.append(repr(error))
        if close_errors:
            raise RuntimeError(f"Environment cleanup failed: {close_errors}")


def compare_traces(reference, candidate):
    mismatches = []
    mismatch_count = 0
    def note(index, field):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 20:
            mismatches.append({"event": index, "field": field})
    if len(reference) != len(candidate):
        note(-1, "event_count")
    for index, (left, right) in enumerate(zip(reference, candidate)):
        for key in ("kind", "step", "environment"):
            if left.get(key) != right.get(key):
                note(index, key)
        if "actions" in left and not np.array_equal(left["actions"], right.get("actions")):
            note(index, "actions")
        if len(left["states"]) != len(right["states"]):
            note(index, "state_count")
        for env_index, (a, b) in enumerate(zip(left["states"], right["states"])):
            for field in FIELDS:
                if not np.array_equal(getattr(a, field), getattr(b, field)):
                    note(index, f"environment_{env_index}.{field}")
            if a.info != b.info:
                note(index, f"environment_{env_index}.info")
    return {"exact_trace_match": mismatch_count == 0, "mismatched_fields": mismatch_count,
            "first_mismatches": mismatches, "reference_events": len(reference),
            "candidate_events": len(candidate),
            "scope": "scripted actions and all returned arrays/info, including reset snapshots; one paired seed schedule only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-name", default="3s5z")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps-per-env", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=32)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this simulator profile inside a Slurm allocation")
    if not 0 <= args.warmup_steps < args.steps_per_env <= 512:
        parser.error("Require 0 <= warmup < steps_per_env <= 512 (2,048 transitions/mode)")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "scripts/triton/profile_smac_steps.sbatch",
               *sorted((root / "src/dms").glob("*.py")),
               root / "reference/smac/smac/env/starcraft2/starcraft2.py"]
    hashes = {}
    for source in sources:
        relative = source.relative_to(root)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    result = {
        "purpose": "bounded_systems_profile_not_training_or_performance_evidence",
        "settings": {"map_name": args.map_name, "num_envs": 4, "seed": args.seed,
                     "environment_seeds": [args.seed + 1000 * i for i in range(4)],
                     "steps_per_env_per_mode": args.steps_per_env, "warmup_steps_per_env": args.warmup_steps,
                     "mode_order": ["serial", "threads"], "thread_workers": 4,
                     "worker_rule": "one outstanding step per environment; ordered batch barrier; resets serial",
                     "action_rule": "sorted legal actions indexed by (step + 3*agent + 5*env) modulo legal count",
                     "models_in_step_measurement": False},
        "runtime": {"host": platform.node(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "partition": os.environ.get("SLURM_JOB_PARTITION"), "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
                    "python": sys.version, "torch_threads": torch.get_num_threads()},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "smac", "pysc2", "s2clientprotocol", "protobuf", "websocket-client")},
        "source_sha256": hashes, "results": [], "status": "running",
        "limitations": ["One serial-then-threaded pair; order and shared-node load can affect timings.",
                        "Python observation/protobuf work still uses the GIL; only independent RPC waits can overlap.",
                        "Native SMAC may internally restart after protocol failure, touching shared launch/port state.",
                        "Exact agreement here does not guarantee simulator or trained-policy reproducibility."],
    }
    result_path = args.output / "profile.json"
    try:
        serial, reference = run_mode("serial", args, include_model=True)
        result["results"].append(serial)
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"mode": "serial", "step_tps": serial["step_only_transitions_per_second"]}), flush=True)
        threaded, candidate = run_mode("threads", args)
        result["results"].append(threaded)
        result["trajectory_comparison"] = compare_traces(reference, candidate)
        result["step_only_speedup"] = serial["step_batch"]["total_seconds"] / threaded["step_batch"]["total_seconds"]
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["source_unchanged_during_profile"] = all(
            hashlib.sha256((root / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(result_path), "step_only_speedup": result["step_only_speedup"],
                      "trajectory_comparison": result["trajectory_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
