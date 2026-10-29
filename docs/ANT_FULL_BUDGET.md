# Ant memory at the full training budget

**Cancelled before training, 2026-10-29:** the user removed full-budget runs
as a requirement. Both training jobs, both evaluations and all three collectors
were still pending and are now cancelled. [The scope-change record](../reports/full-budget-scope-change-20261029.json)
verifies ownership, zero training and scheduler cancellation. The native checks
and frozen preparation below remain available; commands describe an optional
experiment, not an active campaign or release requirement.

The [cancelled comparison](../reports/development-ant-memory-10m-seed101.json)
tests fresh IPPO and DMS policies on native Ant 4x2 with 32-step local policy
memory, training seed 101, matched H100 hardware, and **10M transitions**.
The completed 3M endpoints were **1,055.41 / 1,111.47**, below the paper's
full-budget 1,700 reference. This fills a budget gap and selects no new
architecture or hyperparameter.

DMS's ten-episode training diagnostics rose from 994.93 at 1M to 1,000.80 at
2M and 1,294.79 at 3M. These use seeds 10001–10010 and differ from the fixed
100-episode block. Their late increase motivates measuring the full budget;
it does not establish that the 1,700 target will be reached.

Both arms retain four environments, 1,000 steps per rollout, independent PPO,
native contact buffers, initial log standard deviation −1, zero entropy
coefficient, native rewards and local sensors. DMS keeps soft coordination,
concatenating intention inputs, auxiliary LR 1e-3, two auxiliary steps per
rollout and no PPO gradients through the prior. Auxiliary updates remain after
PPO. This is separate from the entropy-0.1 and residual auxiliary-first trials.

Source is frozen from the current package, with added optional fields disabled.
The [native validation](../reports/ant-full10m-native-validation.json), job
**21289136**, passed on A100 before submission: exact launcher execution,
both saved-source checkpoint evaluations and both archived-source comparisons
over 8,000 transitions. Model/optimizer states, Python/NumPy/Torch/CUDA RNG
states, metrics excluding time and diagnostic episodes match exactly. Only
smoke warmup is zero; full runs retain 100 episodes.

| Arm | Training job | Fixed 100-episode evaluation |
| --- | --- | --- |
| IPPO | 21289400 | 21289401 |
| DMS | 21289402 | 21289403 |

Completion collectors are **21289404–21289406**. Both training jobs remain
cancelled; no 10M performance result was produced.

The original 3M metric files are frozen as prefix references. Each fresh run's
first 750 updates are compared with that reference, excluding elapsed time.
Discrepancies are retained and investigated, never automatically rerun or
discarded. Short native equality alone does not guarantee the H100 trajectory.

Historical H100 3M times were 3.57 / 5.14 hours for IPPO/DMS. Linear 10M planning
gives about 11.89 / 17.13 hours; allocations are 16 / 22 hours. Each endpoint
receives a separate deterministic 100-episode CPU evaluation, seeds
400101–400200, with source/checkpoint hashes, frozen state and zero optimizer
steps checked. Automatic collectors retain failures too. Held-out seeds remain
unused.

Public configs: [IPPO](../configs/development/ant-4x2-ippo-history32-10m-h100-101.json)
and [DMS](../configs/development/ant-4x2-dms-history32-10m-h100-101.json). After
preparing the [historical environment](PORTABLE_LEGACY_SETUP.md), a separate
DMS run uses a fresh output directory:

```bash
sbatch --gpus=h100:1 --cpus-per-task=1 --mem=4G --time=22:00:00 \
  scripts/triton/train_legacy_gpu.sbatch \
  configs/development/ant-4x2-dms-history32-10m-h100-101.json \
  runs/my-ant-memory-full-budget 101
```
