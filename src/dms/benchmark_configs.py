"""Generate matched IPPO/DMS configurations for the manuscript benchmark matrix.

This command writes configuration files only. It never constructs environments,
starts training, or replaces existing files. Run seeds in the manifest override
the development placeholder in each generated configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .train import Config


_TEMPLATES = {
    "mamujoco": "halfcheetah_2x3_dms.json",
    "smac": "smac_3s5z_dms.json",
}
_PRIMARY_VARIANTS = ("ippo", "dms-soft")
_ABLATION_VARIANTS = ("dms-causal", "dms-all", "dms-no-history-filter")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    contents = path.read_bytes()
    value = json.loads(contents)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(contents).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"task ID {value!r} has no usable filename characters")
    return slug


def generate_configs(
    targets: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 10_000_000,
    agent_obsk: int = 1,
    include_ablations: bool = False,
) -> Path:
    """Validate and write the registry's complete matched experiment matrix.

    Reference single-task templates must be next to ``targets``. Full Config
    defaults fill omitted fields; task identity, budget, rollout counts, seed,
    warmup, and variant changes are then explicit. Final-training seeds remain
    in the manifest so run scheduling can choose them without adding unknown
    keys to the trainer's configuration schema.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if agent_obsk not in (0, 1):
        raise ValueError("agent_obsk must be 0 (paper table) or 1 (paper prose)")
    targets, output_dir = Path(targets), Path(output_dir)
    registry, registry_hash = _read_json(targets)
    evaluation = registry["reproduction_evaluation_policy"]
    templates = {}
    template_provenance = {}
    for environment, filename in _TEMPLATES.items():
        path = targets.parent / filename
        supplied, digest = _read_json(path)
        config = Config(**supplied)
        config.validate()
        templates[environment] = asdict(config)
        template_provenance[environment] = {"path": str(path), "sha256": digest}

    variants = _PRIMARY_VARIANTS + (_ABLATION_VARIANTS if include_ablations else ())
    files: dict[str, dict[str, Any]] = {}
    entries = []
    seen_tasks = set()
    for environment in ("mamujoco", "smac"):
        for target in registry[environment]:
            task = target["id"]
            if not isinstance(task, str):
                raise ValueError("every target must have a string ID")
            identity = (environment, task)
            if identity in seen_tasks:
                raise ValueError(f"duplicate task {environment}/{task}")
            seen_tasks.add(identity)
            base = dict(templates[environment])
            base.update(
                env=environment,
                num_envs=4,
                total_steps=steps,
                rollout_steps=1000 if environment == "mamujoco" else 160,
                seed=101,
                warmup_episodes=100,
                filter_history=True,
                agent_obsk=agent_obsk,
            )
            if environment == "mamujoco":
                base.update(scenario=target["environment"], agent_conf=target["agent_configuration"])
            else:
                base.update(map_name=task)
            task_slug = f"{environment}-{_slug(task)}"
            for variant in variants:
                values = dict(base)
                values.update(
                    algorithm="ippo" if variant == "ippo" else "dms",
                    coordination={"dms-causal": "causal", "dms-all": "all"}.get(variant, "soft"),
                    filter_history=variant != "dms-no-history-filter",
                    output=f"runs/{task_slug}-{variant}-101",
                )
                config = Config(**values)
                config.validate()
                filename = f"{task_slug}-{variant}.json"
                if filename in files:
                    raise ValueError(f"task IDs collide at generated filename {filename!r}")
                files[filename] = asdict(config)
                entries.append({"task": task, "environment": environment, "variant": variant, "config": filename})

    manifest = {
        "schema_version": 1,
        "targets": {"path": str(targets), "sha256": registry_hash},
        "reference_templates": template_provenance,
        "total_environment_steps": steps,
        "agent_obsk": agent_obsk,
        "observation_track": "paper_prose_k1" if agent_obsk == 1 else "paper_table_k0_no_observed_peers",
        "placeholder_seed": 101,
        "training_seeds": list(evaluation["training_seeds"]),
        "development_seeds": list(evaluation["development_seeds"]),
        "seed_usage": "Override --seed and --output for every run; use development seeds for tuning and training_seeds for the locked final protocol.",
        "rollout_protocol": {
            "mamujoco": {"num_envs": 4, "rollout_steps": 1000},
            "smac": {"num_envs": 4, "rollout_steps": 160},
            "provenance": "Declared matched independent-PPO redesign; SMAC uses four environments instead of the paper's twenty.",
        },
        "final_evaluation": {
            "checkpoint_selection": evaluation["checkpoint_selection"],
            "mamujoco_episodes_per_seed": evaluation["mamujoco_final_evaluation_episodes_per_seed"],
            "smac_episodes_per_seed": evaluation["smac_final_evaluation_episodes_per_seed"],
        },
        "experiments": entries,
    }
    files["manifest.json"] = manifest

    # Validate the entire matrix and check every destination before writing any
    # file. Exclusive creation also protects against a concurrent writer.
    for filename in files:
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to replace existing file: {output_dir / filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in files.items():
        with (output_dir / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return output_dir / "manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("configs/paper_targets.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--agent-obsk", type=int, choices=(0, 1), default=1)
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = generate_configs(
        args.targets,
        args.output_dir,
        steps=args.steps,
        agent_obsk=args.agent_obsk,
        include_ablations=args.include_ablations,
    )
    count = len(json.loads(manifest_path.read_text())["experiments"])
    print(f"Generated {count} configurations; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
