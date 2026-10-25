# SMAC stepping profile

Ordered threaded stepping improved the measured environment-step throughput from **154.3848 to 273.4999 transitions/s (1.77155×)** in one bounded CPU experiment. All returned trajectories matched exactly in the paired test. This measures simulator stepping with scripted actions; it does not establish a corresponding training speedup, learning improvement, or universal determinism guarantee.

The result is saved in [profile.json](../runs/smac-step-profile-21193014/profile.json), with [stdout](../logs/dms-smac-step-profile_21193014.out), [stderr](../logs/dms-smac-step-profile_21193014.err), and archived source files beside the JSON. Job **21193014** completed with exit **0:0** on **milan2 / batch-milan** in **1 min 35 s**. It requested four CPUs, 8 GB, and 15 minutes; Slurm recorded peak RSS **4,297,820 KiB**, approximately 4.10 GiB. No existing training job was changed.

## Protocol

[profile_smac_steps.py](../scripts/profile_smac_steps.py) creates four independent original-SMAC 3s5z environments with seeds **101, 1101, 2101, 3101**. The serial and threaded modes each execute **512 steps per environment**, or **2,048 environment transitions per mode**. The first 32 steps per environment are warmup, leaving **1,920 measured transitions per mode**. The total simulated budget is 4,096 transitions, separate from every training run.

Actions are deterministic functions of the step, agent index, environment index, and sorted legal-action set. Model inference is excluded from this comparison. Threaded mode uses one persistent `ThreadPoolExecutor(max_workers=4)`, with at most one outstanding call per environment. Worker affinity is not fixed. Futures are drained and their results assembled in environment-index order before advancing the batch. Initial launches and all episode resets remain serial on the main thread. The recorded order is serial first, then threaded.

The runtime used PyTorch 2.6.0+cu126 on CPU with one PyTorch thread, NumPy 2.2.6, SMAC 1.0.0, PySC2 3.0.0, protobuf 3.20.3, and websocket-client 1.9.2. The JSON records simulator settings and source hashes; source files remained unchanged during the profile.

## Measured timings

| Measurement | Serial | Four workers |
| --- | ---: | ---: |
| Environment-step throughput | 154.3848 transitions/s | 273.4999 transitions/s |
| Four-environment step batch, mean | 25.9093 ms | 14.6252 ms |
| Four-environment step batch, median | 25.9985 ms | 14.5033 ms |
| Four-environment step batch, p95 | 30.4552 ms | 18.4637 ms |
| Individual environment call, mean | 6.4747 ms | 10.7986 ms |
| Loop throughput including scripted actions, history, trace recording, and resets | 148.7159 transitions/s | 255.3849 transitions/s |
| Total measured serial reset time, 40 resets | 0.3116 s | 0.3404 s |
| Initial four-game startup | 32.9431 s | 26.7087 s |

Parallel calls overlap, but individual calls become slower. This is consistent with contention and Python-side work limiting the gain. The instrumented loop improved by approximately **1.717×**. Startup is excluded from both throughput measures. Different startup times and the fixed mode order are additional reasons to treat this as a promising single measurement rather than a precise general speedup estimate.

A separate fixed-input CPU DMS check used a newly initialized learner, history length 32, four environments, active intentions, ten warmup calls, and fifty measured calls. Mean `Learner.contexts` time was **47.3765 ms**; mean `Learner.action` time was **7.3718 ms**. Its history contained 4,372 valid visibility entries. This probe performed no simulator steps or parameter updates. It excludes PPO, auxiliary training, and the collector's extra value work, and does not measure GPU inference. Environment stepping therefore accounts for only part of DMS runtime.

## Trajectory and failure checks

The two modes produced **553 matching events**: one initial state batch, 512 step batches, and forty reset snapshots. The comparison checks scripted actions, observations, rewards, termination/truncation flags, available-action masks, observed-action features, visibility, and info dictionaries. It found **zero mismatched fields**. Each environment completed ten episodes; both modes recorded **zero native forced restarts**.

The simulator log contains no Python traceback, `ProtocolError`, or `ConnectionError`. It contains eight SC2 launch messages and shutdown websocket-frame messages. Successful completion and zero forced restarts support the normal-path result; these logs do not exercise a real controller-timeout or simultaneous-restart failure.

The prototype waits for every submitted future before reading results and joins the pool before closing environments. A running RPC cannot be stopped by cancelling its future; PySC2's configured socket timeout still bounds that wait. Upstream SMAC can automatically close and relaunch a game after a protocol failure inside `step`, which would execute in a worker and touch process/port lifecycle state. That recovery path remains untested here and must not be advertised as validated concurrency support.

A separate fake-environment failure check passed: the first worker raised an injected exception while three other workers waited 50 ms. All three completed before any environment closed, the original exception propagated, and all four environments closed. This checks the prototype's future-draining logic without launching StarCraft; it does not validate upstream restart behavior.

## Implication for the collector

The current [collector](../src/dms/train.py) samples a synchronous action batch, steps each environment sequentially, computes terminal bootstrap values, records the final observations, then resets ended episodes. Independent `env.step` calls can plausibly be threaded while retaining that order:

1. Keep all model calls, random action sampling, histories, episode counters, and policy updates on the main thread.
2. Submit one step per independent environment and wait for the whole ordered batch; never issue concurrent calls on one controller.
3. Preserve terminal observations for bootstrapping before any reset, and retain the existing reset/cache-invalidation behavior.
4. Drain outstanding workers before cleanup on any exception. Preserve the adapter's treatment of protocol failures as infrastructure failures, not losses.

Before a production change, broaden the mock checks to simultaneous episode endings and run a small paired collector test that compares actions, context/gate tensors, values, histories, and terminal flags. Counterbalance serial/threaded measurement order across another seed schedule and time the complete collection/update loop on the intended hardware. Keep the parallel setting explicit in experiment provenance and freeze it before held-out seeds. Exact equality on this scripted seed schedule alone does not guarantee equality under learned stochastic policies, different maps, later library versions, or recovery from simulator failures.

To reproduce the bounded profile from the repository directory, after creating `logs/`:

```bash
SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817 \
  sbatch scripts/triton/profile_smac_steps.sbatch
```

The [wrapper](../scripts/triton/profile_smac_steps.sbatch) requests only CPU resources and writes a fresh `runs/smac-step-profile-$SLURM_JOB_ID/` directory. No training implementation has been changed by this prototype.
