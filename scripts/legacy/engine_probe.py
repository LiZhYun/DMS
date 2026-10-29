"""Compare old MuJoCo stacks using identical initial states and controls.

Run this script once per isolated runtime, on a compute node. It accepts Gym
0.17.2 / mujoco-py 2.0.2.8 / engine 200 and the supported Gym 0.23.1 /
mujoco-py 2.1.2.14 / engine 210 stack. Gym 0.23.1 on engine 200 is also
accepted to separate Gym changes from engine/binding changes. It does not relax the public adapter's
version guards. No policy is loaded, trained, or evaluated.

Native same-seed resets are recorded separately. Dynamics use a shared NumPy
generator to specify qpos/qvel explicitly, avoiding Gym's different seed APIs.
Controls are fixed in [-0.5, 0.5]. Unwrapped dynamics continue after terminal
states to keep the input schedule matched; these are not episode returns.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform

import gym
import mujoco_py
import numpy as np

from dms import legacy_graph
from dms.legacy_ant import make_computed_ant
from dms.legacy_swimmer import ASSET_PATH, make_swimmer


CASES = (
    ("HalfCheetah", "native", ("2x3", "3x2", "6x1")),
    ("Ant", "native", ("2x4", "4x2", "8x1")),
    ("Walker2d", "native", ("2x3", "3x2", "6x1")),
    ("ManyAgentSwimmer", "native", ("10x2",)),
    ("Ant", "computed", ("2x4", "4x2", "8x1")),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(scenario, mode, partitions, seed, steps):
    if scenario == "ManyAgentSwimmer":
        env = make_swimmer()
        asset = ASSET_PATH
    elif mode == "computed":
        env = make_computed_ant()
        from gym.envs.mujoco.ant import AntEnv

        asset = Path(inspect.getfile(AntEnv)).parent / "assets" / "ant.xml"
    else:
        wrapped = gym.make(f"{scenario}-v2")
        env = wrapped.unwrapped
        name = {"HalfCheetah": "half_cheetah", "Ant": "ant", "Walker2d": "walker2d"}[scenario]
        asset = Path(inspect.getfile(type(env))).parent / "assets" / f"{name}.xml"
    try:
        if gym.__version__ == "0.17.2":
            env.seed(seed)
            env.reset()
        else:
            env.reset(seed=seed)
        seeded_qpos = env.sim.data.qpos.copy()
        seeded_qvel = env.sim.data.qvel.copy()

        rng = np.random.default_rng(seed)
        scale = 0.005 if scenario == "Walker2d" else 0.1
        qpos = env.init_qpos + rng.uniform(-scale, scale, size=env.model.nq)
        if scenario in {"Ant", "HalfCheetah"}:
            qvel = env.init_qvel + rng.standard_normal(env.model.nv) * scale
        else:
            qvel = env.init_qvel + rng.uniform(-scale, scale, size=env.model.nv)
        env.sim.reset()
        env.set_state(qpos, qvel)
        if mode == "computed":
            mujoco_py.functions.mj_rnePostConstraint(env.model, env.sim.data)

        graphs = {}
        for partition in partitions:
            parts, edges = legacy_graph.graph_for(scenario, partition)
            categories = legacy_graph.observation_categories(scenario, 1)
            graphs[partition] = ([legacy_graph.distance_layers(p, edges, 1) for p in parts], categories)

        rows = {name: [] for name in ("qpos", "qvel", "contact", "native_obs", "reward", "done")}
        rows.update({f"local_obs/{partition}": [] for partition in partitions})

        def record(obs, reward, done):
            for name, values in (("qpos", env.sim.data.qpos), ("qvel", env.sim.data.qvel),
                                 ("contact", env.sim.data.cfrc_ext), ("native_obs", obs)):
                rows[name].append(values.copy())
            rows["reward"].append(float(reward))
            rows["done"].append(bool(done))
            for partition, (layers, categories) in graphs.items():
                rows[f"local_obs/{partition}"].append(np.concatenate([
                    legacy_graph.local_observation(env.sim.data, layer, categories)
                    for layer in layers
                ]).astype(np.float32))

        record(env._get_obs(), 0.0, False)
        controls = np.stack([
            ((np.arange(env.model.nu) + step) % 11 - 5).astype(np.float64) / 10
            for step in range(steps)
        ])
        for control in controls:
            obs, reward, done, _ = env.step(control)
            record(obs, reward, done)
        arrays = {name: np.asarray(values) for name, values in rows.items()}
        arrays.update(input_qpos=qpos, input_qvel=qvel, controls=controls,
                      seeded_qpos=seeded_qpos, seeded_qvel=seeded_qvel)
        assert all(np.isfinite(values).all() for values in arrays.values())
        terminal = np.flatnonzero(arrays["done"])
        metadata = {
            "case": f"{scenario}-{mode}-{seed}", "scenario": scenario,
            "contact_mode": mode, "seed": seed, "steps": steps,
            "first_terminal_step": int(terminal[0]) if terminal.size else None,
            "xml_sha256": digest(asset), "frame_skip": env.frame_skip,
            "timestep": float(env.model.opt.timestep), "partitions": list(partitions),
            "graph_radius": 1,
        }
        return arrays, metadata
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1 or args.output.suffix != ".npz":
        parser.error("--steps must be positive and --output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose fresh output paths")
    versions = {name: importlib.metadata.version(name) for name in ("gym", "mujoco-py", "numpy")}
    engine = int(mujoco_py.functions.mj_version())
    if (versions["gym"], versions["mujoco-py"], engine) not in {
        ("0.17.2", "2.0.2.8", 200), ("0.23.1", "2.0.2.8", 200),
        ("0.23.1", "2.1.2.14", 210),
    }:
        parser.error("Unregistered runtime; declare a new comparison before running")
    arrays, cases = {}, []
    for scenario, mode, partitions in CASES:
        for seed in (400831, 400832, 400833):
            values, metadata = capture(scenario, mode, partitions, seed, args.steps)
            arrays.update({f"{metadata['case']}/{name}": value for name, value in values.items()})
            cases.append(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "purpose": "Fixed-input simulator comparison, not training or evaluation returns",
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
        "python": platform.python_version(), "versions": versions, "engine_version": engine,
        "module_paths": {"gym": gym.__file__, "mujoco_py": mujoco_py.__file__},
        "cases": cases, "arrays": len(arrays), "data_sha256": digest(args.output),
        "script_sha256": digest(__file__), "graph_sha256": digest(legacy_graph.__file__),
        "inputs": "Explicit identical qpos/qvel and controls; separate native same-seed reset arrays",
        "limits": "Dynamics continue beyond terminal states; no episode-return or learning-parity claim",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"engine": engine, "cases": len(cases), "arrays": len(arrays), "output": str(args.output)}))


if __name__ == "__main__":
    main()
