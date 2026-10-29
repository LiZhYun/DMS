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
retained unchanged.

## Declared learning comparison

The [campaign ledger](../reports/development-ant-entropy-3m-seed101.json)
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

The guarded launcher checked all eight required evidence files and submitted:

| Algorithm | Fresh 3M training job | Fixed 100-episode evaluation job |
| --- | --- | --- |
| IPPO | 21282324 | 21282325 |
| DMS | 21282326 | 21282327 |

Three dependent collectors (**21282328–21282330**) preserve the outcomes.
Both training jobs are queued for the same H100 family as their references.

No learning-performance result is available for this comparison yet.
