# Frozen history-filter diagnostic

[diagnose_history_filter.py](../scripts/diagnose_history_filter.py) measures the long/short-intention filter on fresh local trajectories from one frozen DMS checkpoint. It reuses the existing `diagnose_intentions.collect_fresh` collector and never steps an optimizer, updates normalization statistics, or chooses thresholds from returns. Constructing the shared `Learner` creates unused optimizer objects. The probe does not run alternate policies at the tested thresholds.

## What is measured

The deployed filter in [intention.py](../src/dms/intention.py) computes:

```text
long  = prior(full sensed history, full visibility mask)
short = prior(last l history slots, their visibility mask)
score = KL(long || short), summed over latent coordinates
reject = score > K and at least one action was visible in the last l slots
```

The manuscript does not specify KL direction. The implementation's chosen direction is **full-history prior to recent-history prior**; reversing it generally changes the score. There is no division by latent dimension, history length, or observed-action count. Rejection uses strict `>` and masks only the recent suffix, retaining older visibility and peer identities. These scores are distinct from the posterior/prior KL in the auxiliary training objective and from full-policy/peer-removed KL in causal coordination.

For every observer/peer pair and each setting, the diagnostic records eligible prefix indices, raw KL values, full/recent observed-history counts, rejected counts, rejection fraction, and the number of visible action entries removed. The three paper **joint** settings are `(K,l)=(100,6), (10000,4), (1000000,2)`; the checkpoint's own setting is also included if different. Varying K and l together does not isolate their separate effects.

Per-setting summaries pool eligible entries and report peer-only and self-only statistics separately. The production filter also applies to the observer's own intention query, so self results are retained but never mixed into peer rates. A peer without a sensed recent action contributes no sample, even if an older action remains in the history. A peer with no observable action component is excluded; inconsistent supplied visibility is rejected. Empty groups have null rates/quantiles, not artificial zero KL. The probe checks its predicted decisions and masks against the loaded `filtered_history` implementation.

Full histories retain the checkpoint's fixed-length unknown padding after reset. They are not trimmed to create a different full-history encoder input. If l exceeds the available history window, the report records the shorter effective length. Policy collection follows the original checkpoint filter and warmup phase; the diagnostic compares all settings on those same raw, unfiltered past histories. Future-action and reward fields are not inputs to the measurement.

Descriptive development quantiles at 0, 25, 50, 75, 90, 95, 99, and 100 percent show the observed score scale. The corresponding fractions strictly above those values are descriptive only. No value is selected or called optimal. Overlapping prefixes, peer entries, and episodes from one checkpoint are correlated observations, not independent training-seed replicates.

## Frozen inputs and execution

The script requires a Slurm allocation, a checkpoint copy with all write permission bits removed, and a fresh output path. It reads the checkpoint once, then verifies its bytes and read-only protection after collection. Model parameters, buffers, completed-training-episode counter, diagnostic source, helper script, and packaged XML hashes must remain unchanged; otherwise the probe fails without publishing a result.

Use an immutable package snapshot containing `dms/`, including any `assets/*.xml`, and place immutable copies of both diagnostic scripts together in a tools directory. The companion helper is imported by file path relative to the probe, not through the current working directory. On an allocated legacy CPU node, run:

```bash
DMS_SOURCE_ROOT=/absolute/probe/source bash scripts/legacy/bootstrap.sh run \
  /absolute/probe/tools/diagnose_history_filter.py \
  --checkpoint /absolute/run/history-filter-checkpoint-01.pt \
  --output /absolute/probe/history-filter-01.json \
  --steps 128 --num-envs 4 --seed 400001 --device cpu
```

Keep the copied checkpoint beside its training `manifest.json` to retain the training source and environment fields. The result separately records the actual diagnostic package path, source/XML hashes, checkpoint hash, actual environment metadata, and original training metadata when available. A stable diagnostic package need not be identical to the training package; inspect both records before interpreting differences. This is a mechanism artifact, not a final benchmark evaluation.

## Verified Ant probe

Job **21197238** completed with exit `0:0` in **10 seconds** on Triton `milan2`, using four CPU cores, 8 GB RAM, and a 30-minute limit. The [submission and verification ledger](../reports/development-ant-history-filter.json) records the exact command, script/source/checkpoint hashes, runtime, raw artifact path, and 26 passing verification checks.

The immutable checkpoint was copied once from the ongoing historical Ant 4x2 DMS run and contains **600,000 training transitions and 1,860 completed training episodes**, seed 101. Its intended 1M budget had not completed at capture. The probe loaded all 16 Python files from that run's own archived source. It used the existing soft-coordination, latent-only decoder, beta 0.1, k1 policy without changing its weights or original filter setting `(10000,4)`.

Collection used diagnostic seeds **400001–400004**, four environments, and 128 transitions per environment: **512 fresh transitions**, with no complete episode. Intentions and the configured history filter were active. For each joint setting, all 12 ordered observer/peer pairs contributed 508 eligible prefixes, totaling **6,096 peer entries**; the four initial empty-history prefixes per pair were ineligible. These are correlated prefixes from four episode beginnings, not complete-episode or independent-seed evaluations.

| Joint `(K,l)` | Eligible peer entries | Rejected | Mean KL | 99th percentile | Maximum KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(100,6)` | 6,096 | 0 (0%) | 0.9625 | 1.8537 | 2.3374 |
| `(10000,4)` | 6,096 | 0 (0%) | 2.7440 | 4.8039 | 6.7414 |
| `(1000000,2)` | 6,096 | 0 (0%) | 9.5881 | 15.9145 | 19.6575 |

Self-intention rows also had zero rejections: **0/2,032** for every setting. Their maximum KL values were 2.5777, 7.4478, and 25.3333, respectively. All eligible KL values were finite and nonnegative. The script verified the measured strict-threshold decisions and resulting suffix masks against the loaded implementation; post-run verification independently checked every recorded raw decision and the pooled counts.

The exact imported package matched the training snapshot. Model parameters/buffers and completed-training-episode count, checkpoint bytes/read-only protection, diagnostic scripts, and package source stayed unchanged. All four environment metadata records matched the training manifest. The external Gym Ant XML hash recorded at each environment's construction also matched the training hash and a post-run file check; this asset is outside the frozen Python package. No parameter or normalizer updates occurred.

All three thresholds were inactive **on this checkpoint and these sampled prefixes**; each threshold substantially exceeded the corresponding observed maximum. This does not establish inactivity throughout training, reproduce the swimmer sensitivity figure, or demonstrate a return advantage from changing a cutoff. No new cutoff was selected from these scores, and no alternate-policy rollout was performed.

## Verified Swimmer probe

Job **21198220** completed with exit `0:0` in **27 seconds** on Triton
`milan1`, using two CPU cores, 8 GB RAM, and a 15-minute limit. The
[verified ledger](../reports/development-swimmer-history-filter.json) records
41 passing checks and the immutable checkpoint captured at **200,000 training
transitions and 200 completed episodes**, at the causal-selection activation
boundary. It used the training run's exact 17 Python files and packaged XML.

The probe collected **512 fresh transitions** from diagnostic seeds beginning
at 400001, with zero parameter or normalizer updates. Each of the 18 locally
visible ordered peer pairs contributed 508 eligible prefixes; hidden pairs
contributed none. All settings rejected **0/9,144 peer prefixes** and
**0/5,080 self prefixes**.

| Joint `(K,l)` | Mean peer KL | 99th percentile | Maximum |
| --- | ---: | ---: | ---: |
| `(100,6)` | 0.5022 | 0.9284 | 1.5435 |
| `(10000,4)` | 1.4786 | 2.7298 | 4.0326 |
| `(1000000,2)` | 5.0757 | 9.0515 | 10.3117 |

The history filter was inactive on these sampled checkpoint trajectories.
This is separate from the causal coordination gate, which compares policy
distributions rather than history distributions. Neither the sampled
inactivity nor these descriptive quantiles demonstrates that changing a
threshold improves return. No alternative cutoff was selected or trained.
Checkpoint, source, scripts, XML, model state, and environment metadata checks
passed without mutation.

Focused tests use fabricated Gaussian distributions and tiny synthetic trajectories. They verify KL direction/sum, strict boundaries, local visibility, hidden-component invariance, unchanged model state, rejection of source/XML/checkpoint mutation, and the allocation/overwrite guards. These fixtures are software evidence and are not recorded as Ant or swimmer results.
