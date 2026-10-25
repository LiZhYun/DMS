"""Measure collection/update throughput on one allocated compute node.

This is a short systems check, not a learning-performance experiment. Run with
the same number of environments as the intended benchmark configuration.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dms.train import Config, Learner, collect, create_env, empty_history, seed_everything, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['modern', 'legacy'], default='modern')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run this profiling workload inside a compute allocation.')
    if args.output.exists():
        raise FileExistsError(args.output)
    versions = {}
    for package in ('torch', 'numpy', 'gym', 'mujoco-py', 'gymnasium', 'gymnasium-robotics', 'mujoco'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    torch.set_num_threads(1)
    results = []
    for device in args.devices:
        for algorithm in ('ippo', 'dms'):
            config = Config(env='mamujoco', algorithm=algorithm, device=device,
                            mamujoco_backend=args.backend,
                            num_envs=4, rollout_steps=128, total_steps=1024,
                            warmup_episodes=0, entropy_coef=0.0)
            seed_everything(config.seed)
            envs = [create_env(config) for _ in range(config.num_envs)]
            try:
                states = [env.reset(seed=config.seed+1000*i) for i,env in enumerate(envs)]
                learner = Learner(config,envs[0])
                history,visible = empty_history(learner,config.num_envs)
                episodes,returns = 0,np.zeros(config.num_envs)
                for iteration in range(2):
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    data,states,history,visible,episodes,returns,_ = collect(
                        learner,envs,states,history,visible,episodes,returns)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    collected = time.perf_counter()
                    update(learner,data)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    row = dict(backend=args.backend,device=device,algorithm=algorithm,iteration=iteration,
                               collection_seconds=collected-start,update_seconds=end-collected,
                               transitions_per_second=config.num_envs*config.rollout_steps/(end-start),
                               config=asdict(config),environment=getattr(envs[0], 'metadata', {}))
                    results.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k not in {'config', 'environment'}}),flush=True)
            finally:
                for env in envs:
                    env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'purpose':'systems_profile_not_performance_evidence',
                   'job_id':os.environ['SLURM_JOB_ID'],'torch':torch.__version__,
                   'backend':args.backend,'devices':args.devices,'versions':versions,
                   'host':platform.node(),'python':platform.python_version(),
                   'gpu':torch.cuda.get_device_name(0) if 'cuda' in args.devices else None,
                   'results':results},stream,indent=2)


if __name__ == '__main__':
    main()
