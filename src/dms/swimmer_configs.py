"""Write supplemental swimmer study configs; never train or submit jobs.

The seven configurations share one reference condition across both manuscript
figures. This keeps the plotted studies distinct from the prose-only delta=0.4
setting and avoids scheduling an identical reference twice. These provisional
settings are separate from the twelve primary benchmark tasks and their gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .train import Config


DEFAULT_TEMPLATE = Path("configs/development/swimmer10x2-dms-causal-pilot.json")
DEFAULT_TARGETS = Path("configs/paper_targets.json")


def _read(path: Path):
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def generate_configs(
    output_dir: str | Path,
    *,
    template: str | Path = DEFAULT_TEMPLATE,
    targets: str | Path = DEFAULT_TARGETS,
    agent_obsk: int = 1,
) -> Path:
    """Write the ten-million-step supplemental matrix for one sensing track.

    k=1 follows the manuscript prose. Selecting k=0 explicitly produces a
    separately named sensitivity: its strict sensor exposes no peer controls.
    Final seeds are declarations in the manifest, never launched by this code.
    """
    if isinstance(agent_obsk, bool) or agent_obsk not in (0, 1):
        raise ValueError("agent_obsk must be 0 (table sensitivity) or 1 (prose track)")
    supplied, template_provenance = _read(Path(template))
    registry, target_provenance = _read(Path(targets))
    base = Config(**supplied)
    base.validate()
    required = {
        "algorithm": "dms", "env": "mamujoco", "mamujoco_backend": "legacy",
        "scenario": "ManyAgentSwimmer", "agent_conf": "10x2", "coordination": "causal",
        "filter_history": True,
    }
    for field, expected in required.items():
        if getattr(base, field) != expected:
            raise ValueError(f"swimmer study template requires {field}={expected!r}")
    values = asdict(base)
    values.update(
        total_steps=10_000_000, agent_obsk=agent_obsk, seed=101,
        coordination_threshold=0.6, intention_threshold=10_000, short_history=4,
        eval_every=1_000_000, eval_episodes=10,
    )
    settings = [
        ("ippo", {"algorithm": "ippo"}, "engineering_control"),
        ("reference", {}, "shared_plotted_reference"),
        ("delta-02", {"coordination_threshold": 0.2}, "plotted_coordination_threshold"),
        ("delta-08", {"coordination_threshold": 0.8}, "plotted_coordination_threshold"),
        ("delta-04-prose", {"coordination_threshold": 0.4}, "prose_only_not_in_swimmer_plot"),
        ("longshort-k100-l6", {"intention_threshold": 100, "short_history": 6}, "plotted_joint_long_short_setting"),
        ("longshort-k1e6-l2", {"intention_threshold": 1_000_000, "short_history": 2}, "plotted_joint_long_short_setting"),
    ]
    files, entries = {}, []
    for identity, changes, evidence in settings:
        name = f"swimmer10x2-k{agent_obsk}-{identity}"
        config = Config(**{**values, **changes, "output": f"runs/supplemental-{name}-101"})
        config.validate()
        filename = name + ".json"
        files[filename] = asdict(config)
        entries.append({"id": identity, "config": filename, "evidence": evidence})

    evaluation = registry["reproduction_evaluation_policy"]
    manifest = {
        "schema_version": 1,
        "scope": "supplemental_manyagent_swimmer_parameter_studies",
        "status": "unverified_provisional_configurations_not_results",
        "primary_benchmark_gate": False,
        "task": "ManyAgentSwimmer_10x2",
        "environment_track": "historical_gym_v2_mujoco210_compatibility",
        "historical_binary_parity_verified": False,
        "template": template_provenance,
        "targets": target_provenance,
        "total_environment_steps": 10_000_000,
        "placeholder_seed": 101,
        "development_seeds": evaluation["development_seeds"],
        "reserved_final_training_seeds": evaluation["training_seeds"],
        "seed_usage": "Development placeholder only; override seed/output per run. Freeze common settings and source before using reserved final seeds.",
        "agent_obsk": agent_obsk,
        "observation_track": "paper_prose_k1" if agent_obsk else "explicit_paper_table_k0_sensitivity",
        "observed_peer_controls": bool(agent_obsk),
        "sensor_note": "Strict k0 exposes only own controls; k1 exposes adjacent-joint controls. IPPO receives identical local sensors.",
        "model_note": "Provisional latent-initialized decoder and beta objective from the template. decoder_history=false disables the decoder's direct encoded-history path, not the history-based prior, action anchor, or visible-action observations.",
        "warmup_note": "Template intention and causal warmups are declared engineering settings; record gates after threshold activation.",
        "studies": {
            "coordination_threshold": {
                "source_label": "coordinatioh_mujoco",
                "source_path": "paper/source/Coordination_Threshold_10x2_Agent_Swimmer.pdf",
                "plotted_experiment_ids": ["delta-02", "reference", "delta-08"],
                "plotted_values": [0.2, 0.6, 0.8],
                "prose_only_experiment_ids": ["delta-04-prose"],
                "prose_only_values": [0.4],
                "coordination_mode": "causal",
            },
            "long_short_intention": {
                "source_label": "long_short_mujoco",
                "source_path": "paper/source/Long_Short_Term_Intention_Threshold.pdf",
                "plotted_experiment_ids": ["longshort-k100-l6", "reference", "longshort-k1e6-l2"],
                "joint_K_l_settings": [[100, 6], [10_000, 4], [1_000_000, 2]],
                "interpretation": "Joint parameter changes, not an independent factorial test of K and l. Fixing causal delta at 0.6 is the declared reproduction protocol.",
            },
        },
        "required_diagnostics": ["peer_gate_fraction_after_causal_warmup", "history_rejection_rate", "peer_forecast_baselines_with_matched_coverage", "independent_training_seed_returns"],
        "final_evaluation": {
            "status": "protocol_declaration_only_not_a_passed_gate",
            "episodes_per_seed": evaluation["mamujoco_final_evaluation_episodes_per_seed"],
            "checkpoint_selection": evaluation["checkpoint_selection"],
        },
        "experiments": entries,
    }
    files["manifest.json"] = manifest
    output_dir = Path(output_dir)
    for filename in files:
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to replace existing file: {output_dir / filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        with (output_dir / filename).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return output_dir / "manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--agent-obsk", type=int, choices=(0, 1), default=1)
    args = parser.parse_args(argv)
    manifest = generate_configs(args.output_dir, template=args.template, targets=args.targets, agent_obsk=args.agent_obsk)
    print(f"Generated 7 supplemental configurations; unverified manifest: {manifest}")


if __name__ == "__main__":
    main()
