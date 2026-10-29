# Ant exploration diagnostic

The native-contact Ant 4x2 history-32 pair finishes every fixed evaluation
episode at the 1,000-step limit, but its 3M returns remain below the 1,700
target: **1,055.41 IPPO** and **1,111.47 DMS**. Surviving longer is therefore
insufficient for these checkpoints. We are testing whether preserving action
exploration helps learn movement under the unchanged native reward.

## Paired update probe

[Probe evidence](../reports/development-ant-history32-entropy-update-probe.json),
job **21282086**, uses the completed 3M IPPO checkpoint and one fresh native
4,000-transition rollout. Each temporary PPO update restores the same learner,
Adam state, rollout and update RNG. The original checkpoint and source remain
unchanged. The mean action standard deviation before updating is **0.10710**,
compared with the initial **exp(−1) = 0.36788**.

| Entropy coefficient | Mean change in log standard deviation | Smallest component change | Largest component change |
| --- | ---: | ---: | ---: |
| 0 | −0.000997 | −0.003897 | +0.001584 |
| 0.001 | −0.000926 | −0.003820 | +0.001659 |
| 0.01 | −0.000228 | −0.003126 | +0.002325 |
| 0.1 | +0.006451 | +0.003438 | +0.008675 |

All four cases perform the same 80 gradient-clipping calls. Only
coefficient **0.1** increases all eight log-standard-deviation components in
this batch. This measures an exploration gradient, not a return improvement.
It is one IPPO checkpoint and one rollout, with no independent training-seed
replication. Higher action variance can also increase control costs or harm
survival.

The reference executable defaults to **0.001** in
[config_mujoco.py](../reference/DMS/configs/config_mujoco.py); its help text is
inconsistent with that default. Coefficient 0.1 is an explicit redesign choice,
not an asserted paper setting. The reproduction's zero-entropy reference is
retained unchanged. Both reference checkpoints are now included in the
[verified partition-memory bundle](ARTIFACTS.md#ant-and-halfcheetah-partition-memory-controls).

## Declared learning comparison

The [current campaign ledger](../reports/development-ant-entropy-3m-seed101-attempt2.json)
freezes one fresh IPPO/DMS pair at **3M transitions**, development training
seed **101**, on **H100**. Each configuration changes only entropy coefficient
**0 → 0.1** and output path relative to its corresponding completed reference.
Both use native contacts, local history 32, four environments, initial log
standard deviation −1 and the exact archived reference source. DMS retains
soft pooled intention conditioning, auxiliary learning rate 1e-3 and its
100-episode warmup. This is separate from the newer residual-fusion trials.

The final checkpoint is evaluated on **100 episodes**, seeds
**400101–400200**, using the saved source and immutable learner state. There is
no best-checkpoint selection. Both favorable and unfavorable outcomes will be
retained. Held-out training seeds remain unused. The frozen source, settings,
validation hashes and job IDs are recorded in the ledger.

Native validation runs both algorithms for 8,000 transitions and separately
loads their checkpoints for two CPU evaluation episodes each. It checks
finite updates, collection log-probability replay, active DMS peer inputs,
auxiliary updates, environment metadata and source integrity. The first
attempt (**21282251**) failed because the harness incorrectly expected an
intention model for every IPPO agent; IPPO has none. Its artifacts are retained.
The retry (**21282286**) passed on A100 in 4m52s after changing that assertion
and output locations only. Both saved-source CPU evaluations passed with
unchanged learner state and zero optimizer steps. Neither algorithm source
nor the declared training settings changed.
[Native validation](../reports/ant-entropy-native-validation.json).

The [first full submission](../reports/development-ant-entropy-3m-seed101.json)
passed its eight evidence checks, but jobs **21282324 / 21282326** failed before
training. The launcher called `train.runtime_metadata`, which is absent from
the unchanged archived trainer. Only dependency manifests were written: both
runs collected **zero transitions**, and their dependent evaluations were
cancelled. The failed scripts, declarations, logs and output directories remain
intact. The earlier native checks exercised training and evaluation, but did
not execute this launcher metadata call.

The retry records runtime provenance directly in the launcher, without changing
the archived algorithm. It uses separate output/configuration paths and the
same H100 requirement, seed, entropy setting and 3M budget. Native job
**21282799** passed both algorithms through the exact revised launcher at
8,000 transitions, then separately evaluated each saved checkpoint. The
100-episode warmup stays intact in this launcher check; the earlier active-DMS
smoke remains required evidence. Source, checkpoints and evaluation state stayed
unchanged, with zero optimizer updates during evaluation.
[Exact-launcher validation](../reports/ant-entropy-launcher-validation.json).
The launcher records accelerator and library details before `train.run`. Its
numerical flags describe that preflight moment: the archived seeding routine
subsequently enables deterministic cuDNN and disables cuDNN benchmarking.
Do not interpret the preflight flags as a measurement of the later training state.

The retry submission verified **13 evidence files** and queued:

| Algorithm | Fresh 3M training job | Fixed 100-episode evaluation job |
| --- | --- | --- |
| IPPO | 21282828 | 21282829 |
| DMS | 21282830 | 21282831 |

Three dependent collectors (**21282832–21282834**) preserve the outcomes. The
first attempt remains a failed launcher check, not an unsuccessful learning run.

No learning-performance result is available for this comparison yet.
