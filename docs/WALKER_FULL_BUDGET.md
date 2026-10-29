# Walker full-budget development pair

The completed 3M Walker3x2 residual model scored **636.86**, compared with
**524.26** for its trained no-input control. The full-budget reference is
**1,100** at 10M transitions. This experiment measures that full endpoint
directly, using the same residual setting and a matched local-history IPPO
control. [The 3M pair audit](../reports/residual-walker-results-20261029.json)
retains its single-seed limitations.

Both new runs start from initialization with development training seed **101**,
local observation history 32, native Walker2d-v2 compatibility rewards, k=1,
and **H100**. They use identical frozen source and configurations differing
only in algorithm and output. DMS retains soft coordination, residual fusion,
auxiliary learning rate 1e-3, zero prior RL learning rate, and its 100-episode
input warmup. The residual configuration changes only budget and output from
its completed 3M reference.

| Setting | Full budget | Training job | Fixed 100-episode evaluation | Requested time |
| --- | ---: | --- | --- | ---: |
| IPPO | 10M | 21284886 | 21284887 | 12 hours |
| Residual DMS | 10M | 21284888 | 21284889 | 16 hours |

The planning estimates from the previous H100 runs are 8.81 and 12.37 hours
respectively, excluding the separate endpoint evaluation. The requested times
provide additional margin. Jobs have already been submitted; the
[campaign ledger](../reports/development-walker-residual-10m-seed101.json)
records their state and three dependent collectors.

Evaluation uses the fixed 10M checkpoint and 100 CPU episodes, seeds
400101–400200, with saved training source and unchanged learner state. No
held-out training seeds or final-test episode block are used. This is one
development seed; the independent [seed-103 shorter-budget comparison](WALKER_REPLICATION.md)
continues separately. A positive full-budget endpoint alone would not establish
final readiness or robust causal coordination.

The collector compares all 750 updates in each new run's first 3M transitions
with immutable earlier traces, excluding elapsed time. Any discrepancy is
recorded for investigation. It does not trigger automatic replacement of a run.

## Launch validation

[Native job 21284863](../reports/walker-full10m-native-validation.json) passed
on A100 in 2m55s: each setting completed 8,000 transitions and two separate
saved-source evaluation episodes. An additional archived IPPO run exactly
matched current IPPO's model state, optimizer state, training metrics excluding
elapsed time, and diagnostic evaluation episodes over 8,000 transitions. This
checks the inactive optional-code paths at one seed and short budget; full
source identities and future prefix comparisons remain recorded explicitly.
The submission verified nine evidence files before queuing the full pair.

## Reader commands

Prepare and validate the [legacy environment](PORTABLE_LEGACY_SETUP.md) first.
The [public CUDA launcher](../scripts/triton/train_legacy_gpu.sbatch) uses that
environment, checks its source override and a real CUDA kernel, and preserves
the native dependency record. An [A100 validation](../reports/public-legacy-gpu-launcher-validation.json)
completed 8,000 residual-DMS transitions and two saved-source CPU evaluation
episodes. It also checked rejection of a missing source, an existing output,
and a modern-backend configuration before creating the requested run.

Submit from the repository directory after creating `logs/`. Use fresh output
directories. A bounded CUDA check is:

```bash
SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817 \
  sbatch --time=00:30:00 scripts/triton/train_legacy_gpu.sbatch \
  configs/legacy_gpu_smoke.json runs/reader-legacy-gpu-smoke-101 101
```

After inspecting that check, the full matched pair is:

```bash
export SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817
sbatch --gpus=h100:1 --time=12:00:00 scripts/triton/train_legacy_gpu.sbatch \
  configs/development/walker-3x2-ippo-fusion-10m-h100-101.json \
  runs/reader-walker-ippo-10m-101 101
sbatch --gpus=h100:1 --time=16:00:00 scripts/triton/train_legacy_gpu.sbatch \
  configs/development/walker-3x2-residual-fusion-10m-h100-101.json \
  runs/reader-walker-residual-10m-101 101
```

The saved source can also be selected through `DMS_SOURCE_ROOT`. Keep the
actual GPU family and numerical-library versions with the results; a common
seed does not guarantee identical trajectories across hardware families.
