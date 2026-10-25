"""Measure collection/update throughput on one allocated compute node.

This short systems check does not measure learning performance. CPU and CUDA
conditions run sequentially on the same host with explicit seeds and order.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

import dms
from dms.train import Config, Learner, collect, create_env, empty_history, seed_everything, update


def source_hashes():
    package = Path(dms.__file__).resolve().parent
    paths = {f'dms/{p.relative_to(package)}': p for p in package.rglob('*.py')}
    paths['scripts/profile_runtime.py'] = Path(__file__).resolve()
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())}


def host_memory():
    """Linux RSS plus process-wide high-water mark; the latter is cumulative."""
    values = {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                values['current_rss_bytes'] = int(line.split()[1]) * 1024
    return values


def cuda_preflight():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable in this allocation')
    value = torch.ones((8, 8), device='cuda')
    if (value @ value).sum().item() != 512:
        raise RuntimeError('CUDA matrix multiplication preflight failed')
    torch.cuda.synchronize()
    return {'name': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)),
            'compiled_architectures': torch.cuda.get_arch_list(),
            'torch_cuda': torch.version.cuda, 'kernel_preflight_passed': True}


def profile_condition(config, warmup_iterations, timed_iterations, order):
    """Warmups perform real updates; all measured rounds keep the resulting state."""
    seed_everything(config.seed)
    if config.device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    environments, rows = [], []
    started = time.perf_counter()
    try:
        for _ in range(config.num_envs):
            environments.append(create_env(config))
        states = [env.reset(seed=config.seed + 1000*i) for i, env in enumerate(environments)]
        learner = Learner(config, environments[0])
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in [('agents', learner.agents), ('intentions', learner.intentions)]}
        counts['total'] = sum(counts.values())
        history, visible = empty_history(learner, config.num_envs)
        episodes, returns = 0, np.zeros(config.num_envs)
        for iteration in range(warmup_iterations + timed_iterations):
            if config.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            data, states, history, visible, episodes, returns, _ = collect(
                learner, environments, states, history, visible, episodes, returns)
            if config.device == 'cuda':
                torch.cuda.synchronize()
            collected = time.perf_counter()
            metrics = update(learner, data)
            if config.device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            # Metrics may contain structured diagnostics, such as the agent
            # permutation. Validate every numeric leaf through strict JSON.
            try:
                json.dumps(metrics, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f'Invalid profile update: {metrics}') from exc
            if config.algorithm == 'dms' and metrics['peer_gate_fraction'] <= 0:
                raise RuntimeError('DMS profile did not exercise active peer intentions')
            row = dict(backend=config.mamujoco_backend, scenario=config.scenario,
                       agent_conf=config.agent_conf, device=config.device,
                       algorithm=config.algorithm, condition_order=order, iteration=iteration,
                       phase='warmup' if iteration < warmup_iterations else 'timed',
                       collection_seconds=collected-start, update_seconds=end-collected,
                       total_seconds=end-start,
                       transitions=config.num_envs*config.rollout_steps,
                       transitions_per_second=config.num_envs*config.rollout_steps/(end-start),
                       finite_update=True, metrics=metrics, host_memory=host_memory())
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            del data
        return {'condition_order': order, 'config': asdict(config),
                'environment': getattr(environments[0], 'metadata', {}),
                'initial_environment_seeds': [config.seed + 1000*i for i in range(config.num_envs)],
                'parameter_counts': counts, 'iterations': rows,
                'condition_wall_seconds': time.perf_counter()-started,
                'host_memory': host_memory(),
                'gpu_memory': {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                              if config.device == 'cuda' else None}
    finally:
        for env in environments:
            env.close()


def summarize(condition, margin):
    rows = [row for row in condition['iterations'] if row['phase'] == 'timed']
    seconds = sum(row['total_seconds'] for row in rows)
    transitions = sum(row['transitions'] for row in rows)
    hours = 10_000_000 * seconds / transitions / 3600
    return {'timed_iterations': len(rows), 'timed_transitions': transitions,
            'timed_total_seconds': seconds, 'transitions_per_second': transitions/seconds,
            'collection_seconds_mean': float(np.mean([r['collection_seconds'] for r in rows])),
            'update_seconds_mean': float(np.mean([r['update_seconds'] for r in rows])),
            'iteration_seconds_sd': float(np.std([r['total_seconds'] for r in rows], ddof=0)),
            'estimated_10m_training_hours': hours, 'planning_margin_multiplier': margin,
            'estimated_10m_hours_with_margin': hours*margin}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['modern', 'legacy'], default='modern')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    parser.add_argument('--algorithms', nargs='+', choices=['ippo', 'dms'], default=['ippo', 'dms'])
    parser.add_argument('--scenario', default='HalfCheetah')
    parser.add_argument('--agent-conf', nargs='+', default=['2x3'])
    parser.add_argument('--warmup-iterations', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--decoder-history', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--kl-beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--planning-margin', type=float, default=1.5)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run this profiling workload inside a compute allocation.')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup_iterations < 0 or args.iterations < 1 or args.seed < 0:
        parser.error('Use nonnegative warmup/seed and at least one timed iteration')
    if not math.isfinite(args.planning_margin) or args.planning_margin < 1:
        parser.error('The planning margin must be finite and at least one')
    for values in (args.devices, args.algorithms, args.agent_conf):
        if len(set(values)) != len(values):
            parser.error('Repeated devices, algorithms, or partitions are not independent conditions')
    package = Path(dms.__file__).resolve().parent
    selected = os.environ.get('DMS_SOURCE_ROOT')
    if selected is not None and package != (Path(selected).resolve()/'dms'):
        raise RuntimeError('Imported package differs from DMS_SOURCE_ROOT')
    sources = source_hashes()
    versions = {}
    for name in ('torch', 'numpy', 'gym', 'mujoco-py', 'gymnasium', 'gymnasium-robotics', 'mujoco'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    torch.set_num_threads(1)
    gpu = cuda_preflight() if 'cuda' in args.devices else None
    conditions = []
    for partition in args.agent_conf:
        for device in args.devices:
            for algorithm in args.algorithms:
                config = Config(env='mamujoco', algorithm=algorithm, device=device,
                                mamujoco_backend=args.backend, scenario=args.scenario,
                                agent_conf=partition, agent_obsk=1, seed=args.seed,
                                num_envs=4, rollout_steps=args.rollout_steps,
                                total_steps=4*args.rollout_steps*(args.warmup_iterations+args.iterations),
                                minibatch_size=args.minibatch_size, hidden_dim=64, latent_dim=8,
                                history_length=32, warmup_episodes=0, entropy_coef=0.0,
                                decoder_history=args.decoder_history, kl_beta=args.kl_beta)
                config.validate()
                condition = profile_condition(config, args.warmup_iterations, args.iterations, len(conditions))
                condition['summary'] = summarize(condition, args.planning_margin)
                conditions.append(condition)
                print(json.dumps({'condition_order': condition['condition_order'],
                                  'summary': condition['summary']}, allow_nan=False), flush=True)
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    if sources != source_hashes():
        raise RuntimeError('Profiling source changed during execution')
    cpu_models = [line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')] if Path('/proc/cpuinfo').exists() else []
    report = {'purpose': 'systems_profile_not_learning_performance_evidence',
              'job_id': os.environ['SLURM_JOB_ID'], 'partition': os.environ.get('SLURM_JOB_PARTITION'),
              'allocated_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
              'host': platform.node(), 'cpu_model': cpu_models[0] if cpu_models else None,
              'python': platform.python_version(), 'torch': torch.__version__,
              'gpu': gpu, 'backend': args.backend, 'devices': args.devices,
              'versions': versions, 'source_sha256': sources, 'source_root': str(package.parent),
              'source_unchanged_during_profile': True, 'seed': args.seed,
              'protocol': {'condition_order': 'partition, then device, then algorithm, in CLI order',
                           'warmup_iterations': args.warmup_iterations, 'timed_iterations': args.iterations,
                           'warmup': 'Real collect/PPO/auxiliary updates; timings retained but excluded from estimates',
                           'torch_threads': 1, 'environment_steps_parallel': False,
                           'timing': 'CUDA synchronized before collection, after collection, and after PPO/auxiliary updates',
                           'host_peak_memory': 'Process-wide cumulative RSS high-water mark, not a per-condition isolated peak',
                           'scope': 'One host and short early-training trajectories; repeated rounds are not independent trials',
                           'extrapolation': 'Linear 10M transition estimate excludes evaluation, checkpoint I/O and startup; margin is planning allowance, not uncertainty interval'},
              'conditions': conditions,
              'results': [dict(row, config=c['config'], environment=c['environment'])
                          for c in conditions for row in c['iterations']]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
