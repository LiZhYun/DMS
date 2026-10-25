# Complete-episode peer-latent interventions

`scripts/evaluate_interventions.py` probes one frozen DMS soft-attention checkpoint
on complete episodes. It uses the same fresh diagnostic episode seeds and
deterministic actions in three modes:

- `normal`: unchanged inferred contexts and gates.
- `hold`: cache each observer/peer latent at its first normally eligible
  observation in the episode. Reuse it only while that peer is normally eligible.
- `masked`: compute ordinary contexts and gates, then disable peer gates.

Every mode preserves the observer's current self latent (the attention query),
local observations, legal-action masks, model parameters, and normalization
statistics. Ordinary eligibility is computed from actual past-only contexts on
each mode's trajectory. Histories and caches start empty every episode; a peer
without normally eligible observed history cannot initialize or activate a cache.
DMS causal/all variants and checkpoints
still in intention warmup are rejected.

These are downstream representation interventions, not retrained baselines.
Paired seeds match initial conditions; trajectories can diverge after actions
change. Return differences describe this checkpoint's dependence on these inputs,
not superior coordination or independent-training-seed performance. Holding an
early representation may produce stale inputs, and masking peers can put a
trained policy outside its usual input distribution.

Run native environments on an allocated node with an immutable checkpoint and
an immutable copy of the current diagnostic package (the directory containing
`dms/`):

```bash
sbatch scripts/triton/evaluate_interventions_legacy_cpu.sbatch \
  /absolute/run/diagnostic-checkpoint-01.pt \
  /absolute/run/diagnostic-source-03 \
  /absolute/run/episode-interventions-01.json 10 300001
```

The default seeds 300001–300010 are diagnostic seeds, separate from the reserved
final protocol. The script records raw return, length, seed, and eligibility
counts for every episode, descriptive episode SDs and paired return differences,
checkpoint/source hashes, imported package path, dependency versions, and actual
rollout-environment metadata. Parameters, buffers, training phase, and source
hashes must remain unchanged. An episode that exceeds the explicit step bound
fails the probe; it is never reported as a completed outcome. Existing outputs
are rejected rather than overwritten.

## First historical HalfCheetah probe

Job **21196190** completed with Slurm exit `0:0` in 3 min 52 s on milan1.
The checkpoint is the seed-101 historical HalfCheetah 2x3 DMS soft pilot at
500,000 transitions, with a latent-only decoder and KL weight 0.1. The
[raw artifact](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/episode-interventions-01.json)
and [submission ledger](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/episode-interventions-01-submission.json)
retain the exact configuration, seeds, hashes, source selection, and every
episode record. This used the same immutable checkpoint and frozen diagnostic
package as the preceding predictive probe.

All modes completed ten 1,000-step episodes using seeds 300001–300010, totaling
30,000 out-of-training transitions. Each episode had 1,998 eligible peer entries;
hold mode cached exactly two observer/peer representations per episode. The
imported frozen package, source hashes, parameters, and normalizers were verified
unchanged. Thirteen focused tests passed against both the current and frozen
packages before submission; wrapper routing and allocation/overwrite guards
also passed.

| Mode | Mean return | Episode SD | Mean paired difference from normal | Paired-difference SD |
| --- | ---: | ---: | ---: | ---: |
| Normal | 3,084.2811 | 34.5622 | — | — |
| Hold | 2,754.3758 | 29.6455 | −329.9054 | 32.5598 |
| Masked | −74.7843 | 120.8598 | −3,159.0655 | 129.4359 |

SDs are descriptive population SDs (`ddof=0`) over the ten paired evaluation
seeds, not uncertainty across independently trained policies. Hold and masked
returns were lower than normal on all ten seeds. The hold differences ranged
from −387.92 to −290.19; masked differences ranged from −3,385.09 to −3,006.68.

Together with the predictive shuffle probe, this supports useful changing peer
information for this checkpoint. It does not establish superior coordination:
masking can create unfamiliar inputs, no policy was retrained for either
intervention, and only one training seed was probed. HalfCheetah 2x3 has only one
peer per observer, so soft attention has no choice among multiple eligible peers;
this experiment cannot validate peer ranking. The normal return differs from
the earlier five-episode pilot score because these are fresh diagnostic seeds.
