"""Complete-episode, fixed-policy peer-latent interventions for DMS soft attention.

These downstream interventions are neither retrained baselines nor a benchmark
performance claim. Run native simulations inside a Slurm allocation only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

import dms
from dms.train import Config, Learner, advance_history, create_env, empty_history, seed_everything


MODES = ('normal', 'hold', 'masked')


class PeerIntervention:
    """Modify [environment, observer, peer, latent] inputs after ordinary gating.

    Hold caches the first eligible latent separately for every observer/peer.
    A cache is used only when that peer's current ordinary gate is true. The
    observer's own latent always stays current because it is the attention query.
    """

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f'Unknown intervention mode: {mode}')
        self.mode = mode
        self.reset()

    def reset(self):
        self.cache = self.known = None
        self.eligible_peer_entries = 0
        self.captured_peer_entries = 0

    @torch.no_grad()
    def apply(self, observations, contexts, gates, available):
        peers = ~torch.eye(contexts.shape[1], dtype=torch.bool, device=contexts.device)
        eligible = gates.bool() & peers[None]
        self.eligible_peer_entries += int(eligible.sum().item())
        if self.mode == 'normal':
            return observations, contexts, gates, available
        if self.mode == 'masked':
            # Ordinary gates are computed first; this explicit downstream mask
            # removes only peers. It never substitutes an artificial self query.
            return observations, contexts, gates & ~peers[None], available
        if self.cache is None:
            self.cache = torch.zeros_like(contexts)
            self.known = torch.zeros_like(gates, dtype=torch.bool)
        first = eligible & ~self.known
        self.cache[first] = contexts[first]
        self.known |= first
        self.captured_peer_entries += int(first.sum().item())
        altered = torch.where(eligible[..., None], self.cache, contexts)
        return observations, altered, gates, available


def state_hash(learner):
    """Hash parameters and buffers, including observation/value normalizers."""
    digest = hashlib.sha256()
    for name, module in (('agents', learner.agents), ('intentions', learner.intentions)):
        for key, value in module.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update(f'{name}.{key}:{value.dtype}:{tuple(value.shape)}\n'.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    """Include fixed simulator assets in the unchanged-source assertion."""
    package = Path(dms.__file__).resolve().parent
    paths = {f'dms/{p.relative_to(package)}': p for p in package.rglob('*.py')}
    paths.update({f'dms/{p.relative_to(package)}': p
                  for p in (package / 'assets').rglob('*.xml')})
    paths['scripts/evaluate_interventions.py'] = Path(__file__).resolve()
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())}


@torch.no_grad()
def evaluate_modes(learner, seeds, max_episode_steps=10000, env_factory=None):
    """Paired initial seeds; each mode follows its own resulting trajectory.

    The completed-training-episode counter is frozen. Each evaluation episode
    starts with empty histories and an empty intervention cache. Incomplete
    episodes fail the probe instead of being reported as completed outcomes.
    """
    if learner.config.algorithm != 'dms' or learner.config.coordination != 'soft':
        raise ValueError('The intervention probe supports DMS soft attention only')
    if learner.completed_episodes < learner.config.warmup_episodes:
        raise ValueError('Checkpoint intentions are still in their inactive warmup phase')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or max_episode_steps < 1:
        raise ValueError('Use distinct episode seeds and a positive step bound')
    env_factory = create_env if env_factory is None else env_factory
    learner.agents.eval()
    learner.intentions.eval()
    initial_hash, phase = state_hash(learner), learner.completed_episodes
    records, environments = [], {}
    for mode in MODES:
        env = env_factory(learner.config)
        metadata = {}
        try:
            intervention = PeerIntervention(mode)
            for seed in seeds:
                seed_everything(seed)
                state = env.reset(seed=seed)
                if not metadata:
                    metadata = {'environment': deepcopy(getattr(env, 'metadata', {})),
                                'initial_runtime': deepcopy(getattr(env, 'runtime_info', {}))}
                history, visible = empty_history(learner, 1)
                intervention.reset()
                total = 0.0
                for length in range(1, max_episode_steps + 1):
                    inputs = learner.contexts([state], history, visible, use_intentions=True)
                    altered = intervention.apply(*inputs)
                    actions, _, _ = learner.action(*altered, deterministic=True)
                    state = env.step(actions[0].cpu().numpy())
                    advance_history(history, visible, [state])
                    total += float(state.rewards.mean())
                    if state.episode_done:
                        records.append({'mode': mode, 'seed': seed, 'return': total,
                                        'length': length, 'battle_won': state.info.get('battle_won'),
                                        'eligible_peer_entries': intervention.eligible_peer_entries,
                                        'captured_peer_entries': intervention.captured_peer_entries})
                        break
                else:
                    raise RuntimeError(f'{mode}, seed {seed}: episode exceeded {max_episode_steps} steps')
                if state_hash(learner) != initial_hash or learner.completed_episodes != phase:
                    raise RuntimeError('Intervention evaluation changed model state or training phase')
        finally:
            metadata['final_runtime'] = deepcopy(getattr(env, 'runtime_info', {}))
            environments[mode] = metadata
            env.close()
    return records, environments


def summarize(records):
    summary = {}
    normal = {r['seed']: r['return'] for r in records if r['mode'] == 'normal'}
    for mode in MODES:
        rows = [r for r in records if r['mode'] == mode]
        returns = np.array([r['return'] for r in rows])
        entry = {'episodes': len(rows), 'return_mean': float(returns.mean()),
                 'return_episode_sd': float(returns.std(ddof=0))}
        if mode != 'normal':
            differences = [r['return'] - normal[r['seed']] for r in rows]
            entry.update(paired_return_difference_mean=float(np.mean(differences)),
                         paired_return_difference_sd=float(np.std(differences, ddof=0)))
        summary[mode] = entry
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=300001)
    parser.add_argument('--max-episode-steps', type=int, default=10000)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Native intervention evaluation requires a Slurm allocation')
    if args.episodes < 1 or args.seed < 0 or args.max_episode_steps < 1:
        parser.error('Episode count and step bound must be positive; seed must be nonnegative')
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=args.device, weights_only=False)
    config = Config(**checkpoint['config'])
    config.device = args.device
    config.validate()
    if config.algorithm != 'dms' or config.coordination != 'soft':
        parser.error('The intervention probe supports DMS soft attention only')
    torch.set_num_threads(config.torch_threads)
    seed_everything(args.seed)
    sources = source_hashes()
    dimensions = create_env(config)
    try:
        learner = Learner(config, dimensions)
    finally:
        dimensions.close()
    learner.agents.load_state_dict(checkpoint['agents'])
    learner.intentions.load_state_dict(checkpoint['intentions'])
    learner.completed_episodes = checkpoint['episodes']
    initial_hash = state_hash(learner)
    seeds = list(range(args.seed, args.seed + args.episodes))
    records, environments = evaluate_modes(learner, seeds, args.max_episode_steps)
    if state_hash(learner) != initial_hash or source_hashes() != sources:
        raise RuntimeError('Model state or diagnostic source changed during evaluation')
    manifest_path = args.checkpoint.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    versions = {}
    for package in ('torch', 'numpy', 'gymnasium', 'gymnasium-robotics', 'mujoco',
                    'gym', 'mujoco-py', 'smac', 'pysc2', 's2clientprotocol', 'protobuf'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        'purpose': 'paired_complete_episode_representation_intervention_not_benchmark_evaluation',
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
        'training_steps': checkpoint['steps'], 'training_seed': config.seed,
        'training_episodes': checkpoint['episodes'], 'evaluation_config': asdict(config),
        'checkpoint_source_sha256': manifest.get('source_sha256'),
        'diagnostic_source_sha256': sources, 'source_unchanged_during_evaluation': True,
        'model_state_sha256': initial_hash, 'model_state_unchanged': True, 'parameter_updates': 0,
        'episode_seeds': seeds, 'episodes_per_mode': args.episodes,
        'out_of_training_transitions': sum(r['length'] for r in records),
        'max_episode_steps': args.max_episode_steps, 'environments': environments,
        'versions': versions,
        'runtime': {'host': platform.node(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                    'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'), 'device': args.device,
                    'diagnostic_package_root': str(Path(dms.__file__).resolve().parent),
                    'python_version': platform.python_version()},
        'protocol': {
            'actions': 'deterministic checkpoint actions in all modes; no normalizer updates',
            'eligibility': 'ordinary soft gates computed from actual past-only contexts on each mode trajectory',
            'hold': 'cache each peer at first eligible observation per episode; reuse only when normally eligible',
            'masked': 'disable peer gates downstream of ordinary gate computation; preserve contexts',
            'preserved': 'current self query, local observations, legal-action mask, weights and training phase',
            'reset': 'empty action histories and peer caches for every episode',
            'pairing': 'same initial episode seeds across modes; subsequent trajectories may diverge',
            'interpretation': 'downstream representation intervention, not a retrained baseline or proof of superior coordination',
            'uncertainty': 'episode SD and paired differences describe one checkpoint, not independent training seeds',
        },
        'records': records, 'summary': summarize(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'summary': result['summary']}))


if __name__ == '__main__':
    main()
