"""Measure HalfCheetah action-cost sensitivity without changing a saved policy.

Load a trusted, completed run with its archived source on PYTHONPATH. Evaluation
uses the original bounded controls and deterministic policy. A second return
charges the raw Gaussian command instead, on exactly the same trajectory.
This is an interface sensitivity, not a retrained policy or historical parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import dms.train as train


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    package = Path(train.__file__).resolve().parent
    return {str(p.relative_to(package.parent)): sha(p)
            for p in sorted(package.glob('*.py'))}


def state_hash(learner):
    """Include normalization buffers and the intention activation counter."""
    digest = hashlib.sha256(str(learner.completed_episodes).encode())
    for name, module in [('agents', learner.agents), ('intentions', learner.intentions)]:
        for key, value in sorted(module.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update((name + key + str(value.dtype) + str(tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ControlCostAudit:
    """Observe controls/rewards while forwarding every native state unchanged."""

    def __init__(self, env, keep_trace=False):
        self.env = env
        self.episodes = []
        self.trace = [] if keep_trace else None
        self.current = None
        metadata = env.metadata
        if (metadata.get('scenario') != 'HalfCheetah'
                or metadata.get('native_env_id') != 'HalfCheetah-v2'
                or metadata.get('control_processing') != 'clip_native_bounds_before_simulation_and_reward'):
            raise ValueError('This diagnostic requires the bounded historical HalfCheetah-v2 adapter.')
        for i, n in enumerate(env.action_dims):
            np.testing.assert_array_equal(env.action_low[i, :n], -np.ones(n))
            np.testing.assert_array_equal(env.action_high[i, :n], np.ones(n))
        np.testing.assert_array_equal(env.native.unwrapped.model.actuator_ctrlrange,
                                      np.tile([-1., 1.], (6, 1)))
        if not env.native.unwrapped.model.actuator_ctrllimited.all():
            raise ValueError('Native actuator limits must be enabled.')

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        self.current = dict(seed=seed, bounded_return=0., raw_command_cost_return=0.,
                            excess_control_cost=0., length=0, steps_with_projection=0,
                            projected_control_coordinates=0, total_control_coordinates=0,
                            max_absolute_raw_command=0.)
        return self.env.reset(seed=seed)

    def step(self, actions):
        # Policy axes: [agent, padded action]. Route using the declared partition.
        actions = np.asarray(actions, dtype=np.float32)
        raw = np.zeros(self.env.native.action_space.shape, dtype=np.float32)
        for i, indices in enumerate(self.env.action_ids):
            raw[indices] = actions[i, :len(indices)]
        bounded = np.clip(raw, -1, 1)
        state = self.env.step(actions)
        raw_cost = .1 * np.square(raw).sum()
        bounded_cost = .1 * np.square(bounded).sum()
        if not np.isclose(state.info['reward_ctrl'], -bounded_cost, rtol=0, atol=1e-12):
            raise ValueError('Native control penalty does not match the declared convention.')
        reward = float(state.rewards.mean())
        expected = np.full_like(state.rewards, state.info['reward_run'] - bounded_cost)
        if not np.array_equal(state.rewards, expected):
            raise ValueError('Native reward does not match the HalfCheetah-v2 formula.')
        alternative = float(np.full_like(state.rewards, state.info['reward_run'] - raw_cost).mean())
        changed = raw != bounded
        row = self.current
        row['bounded_return'] += reward
        row['raw_command_cost_return'] += alternative
        row['excess_control_cost'] += float(raw_cost - bounded_cost)
        row['length'] += 1
        row['steps_with_projection'] += int(changed.any())
        row['projected_control_coordinates'] += int(changed.sum())
        row['total_control_coordinates'] += raw.size
        row['max_absolute_raw_command'] = max(row['max_absolute_raw_command'], float(np.abs(raw).max()))
        if self.trace is not None:
            self.trace.append((raw.copy(), state.info['reward_run'], reward, alternative))
        if state.episode_done:
            if abs(row['bounded_return'] - row['raw_command_cost_return'] - row['excess_control_cost']) > .01:
                raise ValueError('Episode cost decomposition failed.')
            self.episodes.append(row.copy())
        return state

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='completed run containing latest.pt and manifest.json')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=400101, help='first development evaluation seed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, help='optional new .npz file of native-order controls and step rewards')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error('episodes must be positive and seed nonnegative')
    if args.output.exists() or (args.trace and args.trace.exists()):
        parser.error('output/trace already exists; choose new paths')
    if args.trace and (args.trace.suffix != '.npz' or args.trace.resolve() == args.output.resolve()):
        parser.error('trace must be a distinct .npz path')
    checkpoint_path, manifest_path = args.run / 'latest.pt', args.run / 'manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = source_hashes()
    expected_sources = {k: v for k, v in manifest['source_sha256'].items() if k.endswith('.py')}
    if sources != expected_sources:
        parser.error('Import the run\'s archived source: set DMS_SOURCE_ROOT (bootstrap) or PYTHONPATH to <run>/source.')
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'gym', 'mujoco-py']}
    if any(versions[k] != manifest['versions'][k] for k in versions):
        parser.error('The relevant package versions must match the run manifest.')
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    config = train.Config(**checkpoint['config'])
    if (not manifest.get('status', '').startswith('completed')
            or checkpoint['steps'] != config.total_steps
            or manifest.get('completed_steps') != checkpoint['steps']):
        parser.error('Use the completed final checkpoint; this diagnostic never selects a best checkpoint.')
    if config.env != 'mamujoco' or config.mamujoco_backend != 'legacy' or config.scenario != 'HalfCheetah':
        parser.error('Only historical HalfCheetah-v2 is supported.')
    config.device = 'cpu'
    torch.set_num_threads(config.torch_threads)
    train.seed_everything(args.seed)
    env = train.create_env(config)
    try:
        # Absolute asset paths may differ after moving an otherwise identical run.
        environment = env.metadata
        if {k: v for k, v in environment.items() if k != 'xml_asset'} != {
                k: v for k, v in manifest['environment'].items() if k != 'xml_asset'}:
            raise ValueError('Native environment metadata differs from the archived run.')
        learner = train.Learner(config, env)
    finally:
        env.close()
    learner.agents.load_state_dict(checkpoint['agents'])
    learner.intentions.load_state_dict(checkpoint['intentions'])
    learner.completed_episodes = checkpoint['episodes']
    before = state_hash(learner)
    original_create, original_step = train.create_env, torch.optim.Adam.step
    audited = []

    def create(c):
        native = original_create(c)
        try:
            wrapper = ControlCostAudit(native, keep_trace=args.trace is not None)
        except BaseException:
            native.close()
            raise
        audited.append(wrapper)
        return wrapper

    def forbidden_step(*unused_args, **unused_kwargs):
        raise RuntimeError('Evaluation must not optimize parameters.')

    try:
        train.create_env, torch.optim.Adam.step = create, forbidden_step
        records = train.evaluate(learner, args.episodes, args.seed)
    finally:
        train.create_env, torch.optim.Adam.step = original_create, original_step
    if len(audited) != 1 or state_hash(learner) != before:
        raise RuntimeError('Evaluation changed model state or created an unexpected environment count.')
    episodes = audited[0].episodes
    if len(episodes) != args.episodes or any((a['seed'], a['bounded_return'], a['length']) !=
            (b['seed'], b['return'], b['length']) for a, b in zip(episodes, records)):
        raise RuntimeError('Audited episodes do not exactly match the original evaluator.')
    if sha(checkpoint_path) != checkpoint_hash or sha(manifest_path) != manifest_hash or source_hashes() != sources:
        raise RuntimeError('The checkpoint, manifest or source changed during evaluation.')
    summary = {key + '_mean': statistics.fmean(r[key] for r in episodes)
               for key in ['bounded_return', 'raw_command_cost_return', 'excess_control_cost']}
    summary.update(projected_step_fraction=sum(r['steps_with_projection'] for r in episodes) / sum(r['length'] for r in episodes),
                   projected_coordinate_fraction=sum(r['projected_control_coordinates'] for r in episodes) / sum(r['total_control_coordinates'] for r in episodes),
                   max_absolute_raw_command=max(r['max_absolute_raw_command'] for r in episodes))
    result = dict(purpose='raw_command_cost_sensitivity_not_retrained_performance',
                  checkpoint_sha256=checkpoint_hash, manifest_sha256=manifest_hash,
                  source_sha256=sources, script_sha256=sha(Path(__file__)),
                  versions=versions, environment=environment, device='cpu',
                  training_seed=config.seed, training_steps=checkpoint['steps'],
                  deterministic=True, optimizer_steps=0, state_unchanged=True,
                  checkpoint_manifest_source_unchanged=True, state_sha256=before,
                  summary=summary, episodes=episodes)
    if args.trace:
        rows = audited[0].trace
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open('xb') as stream:
            np.savez_compressed(stream, raw_controls=np.stack([r[0] for r in rows]),
                                run_reward=np.asarray([r[1] for r in rows]),
                                bounded_reward=np.asarray([r[2] for r in rows], dtype=np.float32),
                                raw_command_reward=np.asarray([r[3] for r in rows], dtype=np.float32),
                                episode_seeds=np.asarray([r['seed'] for r in episodes]),
                                episode_lengths=np.asarray([r['length'] for r in episodes]))
        result['trace'] = dict(path=str(args.trace), sha256=sha(args.trace))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)))


if __name__ == '__main__':
    main()
