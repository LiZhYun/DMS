"""Evaluate a local experiment checkpoint on new episode seeds."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .train import Config, Learner, create_env, evaluate, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    # These checkpoints are produced by train.py and include Python RNG state.
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    manifest_path = args.checkpoint.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    config = Config(**checkpoint['config'])
    config.device = args.device
    torch.set_num_threads(config.torch_threads)
    seed_everything(args.seed)
    env = create_env(config)
    try:
        learner = Learner(config, env)
    finally:
        env.close()
    learner.agents.load_state_dict(checkpoint['agents'])
    learner.intentions.load_state_dict(checkpoint['intentions'])
    learner.completed_episodes = checkpoint['episodes']
    evaluation_metadata = {}
    records = evaluate(learner, args.episodes, args.seed, metadata=evaluation_metadata)
    summary = {'checkpoint':str(args.checkpoint.resolve()),'training_steps':checkpoint['steps'],
               'training_seed':config.seed,'episode_seed_start':args.seed,'episodes':args.episodes,
               'config':asdict(config),'manifest':manifest,
               'evaluation_environment':evaluation_metadata['environment'],
               'evaluation_versions':evaluation_metadata['versions'],
               'evaluation_runtime':evaluation_metadata['runtime'],
               'evaluation_source_sha256':{str(path.relative_to(Path(__file__).parent.parent)):
                   hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(Path(__file__).parent.glob('*.py'))},
               'evaluation_protocol':{'deterministic':True,'episode_seeds':'consecutive from episode_seed_start'},
               'return_mean':float(np.mean([row['return'] for row in records])),
               'return_std':float(np.std([row['return'] for row in records])),
               'results':records}
    if all(row['battle_won'] is not None for row in records):
        summary['win_rate'] = float(np.mean([row['battle_won'] for row in records]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in summary.items()
                      if key not in {'results','config','manifest'}}))


if __name__ == '__main__':
    main()
