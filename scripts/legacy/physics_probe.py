"""Capture fixed-action legacy physics for comparison across installations.

Run with a native legacy Python or through scripts/legacy/container.sh. Arrays
include the initial state followed by each post-action state; episode reset
observations are stored separately. This script does not instantiate a learner.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import mujoco_py
import numpy as np

import dms
from dms.legacy_env import LegacyMaMuJoCoEnv


CASES = [
    (scenario, partition, "native")
    for scenario, partitions in [
        ("HalfCheetah", ["2x3", "3x2", "6x1"]),
        ("Ant", ["2x4", "4x2", "8x1"]),
        ("Walker2d", ["2x3", "3x2", "6x1"]),
        ("ManyAgentSwimmer", ["10x2"]),
    ]
    for partition in partitions
] + [("Ant", partition, "computed") for partition in ["2x4", "4x2", "8x1"]]
STATE_FIELDS = ("obs", "rewards", "terminated", "truncated", "observed_actions", "visibility")


def capture_case(scenario, partition, contact_mode, seed, steps):
    env = LegacyMaMuJoCoEnv(
        scenario, partition, agent_obsk=1, ant_contact_forces=contact_mode
    )
    rows = {name: [] for name in (*STATE_FIELDS, "qpos", "qvel", "contact")}
    resets = []
    episodes = 0

    def record(state):
        for name in STATE_FIELDS:
            rows[name].append(getattr(state, name).copy())
        rows["qpos"].append(env.native.sim.data.qpos.copy())
        rows["qvel"].append(env.native.sim.data.qvel.copy())
        rows["contact"].append(env.native.sim.data.cfrc_ext.copy())

    try:
        state = env.reset(seed=seed)
        resets.append(state.obs.copy())
        record(state)
        indices = np.arange(env.n_agents * env.action_dim).reshape(
            env.n_agents, env.action_dim
        )
        for step in range(steps):
            # Deterministic controls in [-0.5, 0.5]; independent of PyTorch and RNG.
            controls = ((indices + step) % 11 - 5).astype(np.float32) / 10
            state = env.step(controls)
            record(state)
            if state.episode_done:
                episodes += 1
                state = env.reset(seed=seed + episodes)
                resets.append(state.obs.copy())
        arrays = {name: np.stack(values) for name, values in rows.items()}
        arrays["reset_obs"] = np.stack(resets)
        if not all(np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("Non-finite simulator output")
        metadata = {
            "case": f"{scenario}-{partition}-{contact_mode}",
            "steps": steps,
            "episodes": episodes,
            "seed_start": seed,
            "environment": env.metadata,
        }
        return arrays, metadata
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024, help="Transitions per case")
    args = parser.parse_args()
    if args.steps <= 0 or args.output.suffix != ".npz":
        parser.error("--steps must be positive and --output must end in .npz")
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Choose fresh output paths; existing artifacts are not replaced")

    arrays, cases = {}, []
    for index, (scenario, partition, contact_mode) in enumerate(CASES):
        values, metadata = capture_case(
            scenario, partition, contact_mode, 400901 + 10000 * index, args.steps
        )
        arrays.update({f"{metadata['case']}/{name}": value for name, value in values.items()})
        cases.append(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    package = Path(dms.__file__).resolve().parent
    source_hashes = {
        str(path.relative_to(package.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".xml")
    }
    metadata = {
        "purpose": "Fixed-action physics/adapter comparison, not learning performance",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "python": platform.python_version(),
        "used_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "gym", "mujoco-py")
        },
        "engine_version": mujoco_py.functions.mj_version(),
        "package_root": str(package / "__init__.py"),
        "source_sha256": source_hashes,
        "cases": cases,
        "arrays": len(arrays),
        "data_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "transitions": args.steps * len(cases),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
