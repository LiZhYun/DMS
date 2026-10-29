# Local training-design audit, 2026-10-29

## Review status

The independent review is **REVIEW_UNAVAILABLE**. The installed research-review
skill requires an explicit reviewer model and ultra reasoning setting. The
exposed Codex MCP interface accepts a model but does not expose a config or
reasoning-effort parameter. No external review call was made. This document is
a local audit by the current executor and carries no independent verdict.

## Objective and evidence

The objective remains a readable DMS implementation with useful intention
inference and causal coordination, and reproducible performance above the
comparable paper targets. The task is unfinished. The sibling
`bpta-agentmixer/` is outside this session's ownership.

The authoritative method contract is [PAPER_CONTRACT.md](docs/PAPER_CONTRACT.md).
The manuscript describes two intention updates before an actor/value update and
soft intention updates; the implementation instead finishes a collected
rollout's PPO epochs/minibatches and then performs `aux_updates=2` auxiliary
optimizer steps per observer. There is no Polyak averaging. This difference is
explicitly documented; it is not a newly discovered coding error.

With the default zero prior RL learning rate, `train.update` uses cached rollout
features throughout PPO. `train.update_intentions` then changes the prior used
by the next rollout. Therefore that last auxiliary change can alter the actual
policy after the PPO surrogate was optimized. This is a possible stabilization
issue, rather than proof that PPO likelihood replay is broken.

Relevant measured evidence:

- [Walker residual pair](reports/residual-walker-results-20261029.json): fixed
  3M, seed 101, H100, 100 evaluation episodes; residual DMS 636.86 versus trained
  no-input 524.26. Both remain below the full-budget 1,100 target.
- The earlier IPPO memory control scores 2,322.92. Its source predates optional
  residual features; [native regression](reports/walker-full10m-native-validation.json)
  exactly matches current IPPO over 8,000 transitions, not the full trajectory.
- [Lower auxiliary LR and pooling trials](reports/stability-pooling-results-20261029.json)
  did not resolve the Walker/SMAC gaps. [Joint-prior RL trials](reports/development-conditioning-diagnosis-3m-seed101.json)
  likewise remained below their matched controls.
- [No-input diagnostic](reports/residual-no-input-rng-diagnostic.json) establishes
  auxiliary RNG consumption and tiny clipping arithmetic differences. It does
  not explain the full return gap or justify choosing a favorable RNG path.
- The current SMAC residual/no-input configurations already use 20 environments
  by 160 steps. A new 20-environment trial would duplicate existing work.

The fixed 10M Walker pair and seed-103 three-arm comparison are submitted.
Their results remain pending. The
[completed SMAC residual pair](reports/residual-smac-results-20261029.json)
scores 0/100 wins with peer inputs and 74/100 with inputs disabled. Held-out
training seeds remain unused.

## Minimal counterfactual

Use the immutable 3M residual Walker checkpoint and one fresh 4,000-transition
native rollout. Compare:

1. Ordinary PPO then auxiliary training, with an uninstrumented replay oracle.
2. Auxiliary first, then PPO using the old cached features.
3. The identical auxiliary update first, then PPO evaluating updated prior
   features from the same past histories with the original filter masks and
   peer gates. Keep the original behavior log probabilities.

Capture the ordinary update's auxiliary RNG state and replay it in both
auxiliary-first branches. Restore the ordinary PPO phase's initial RNG before
each counterfactual PPO phase. Require identical final intention and auxiliary
optimizer states. Require branch 2 to match the ordinary model/optimizer state;
that checks that merely reordering independent updates is not mistaken for a
new mechanism. This phase pairing is a diagnostic intervention, not a proposed
production RNG redesign.

Measure full-policy Gaussian KL, sampled action-ratio clipping fractions and
the fitted PPO surrogate both on the fixed rollout inputs and after actual
observation/intention refresh. Check analytic Gaussian KL against the library
formula. Record changes in the filter masks and gates. The observations,
visibility, future targets, original files and checkpoint must remain unchanged.

## Claims matrix

The [completed probe](reports/development-walker-residual-auxiliary-order-probe.json)
passes the uninstrumented ordinary replay and stale-cache equivalence oracles.
All three variants have identical auxiliary model and optimizer states and
20 PPO optimizer steps per agent. With updated past-only inputs, actual
refreshed-policy KL falls from **0.00917540 to 0.00780247** (14.96%), while the
fitted surrogate rises from **0.00852421 to 0.00938248** (10.07%). No filter or
gate entries change. The effect is concentrated in the observer with the
largest auxiliary-induced shift; it is not a return measurement.

The [completed frozen-policy interventions](reports/residual-walker-mechanism-results-20261029.json)
also lower Walker return when peer inputs are held (527.87) or masked (489.40),
compared with normal inputs (636.86). The
[peer-only forecast extension](reports/peer-forecast-diagnostic-validation.json)
beats both history baselines and replays all previous diagnostic fields exactly.
Therefore the update-order hypothesis concerns an active learned input path.
These results still leave the below-target return and training-seed variance
unresolved.

The [completed SMAC interventions](reports/residual-smac-mechanism-results-20261029.json)
are weaker: normal, held and masked inputs all win 0/100; held peer inputs
change shaped return by only +0.02187, and masking lowers it by 2.30590.
The [peer-only replay](reports/smac-peer-forecast-visibility-validation.json)
beats both history baselines, and matched peer-latent shuffling raises decoder
NLL by a target-weighted 0.41850. Thus poor peer forecasting hidden by self
accuracy is unsupported on this rollout; useful winning coordination is still
unproven. Dead peers account for 16.76% of living observers' enabled peer entries.
That confirms exposure, not a benefit from changing eligibility. The queued
before/after trials still test the input-consistency hypothesis.

| Observation | Supported conclusion | Next action |
| --- | --- | --- |
| Cached auxiliary-first branch matches ordinary update | Ordering alone does not change this update when PPO retains old inputs | Avoid promoting an ordering-only change |
| Re-encoding reduces actual refreshed-policy KL and preserves the auxiliary update | The proposed update corrects a measured input-consistency issue on this batch | Consider a separately declared native learning trial |
| Improvement occurs only on cached inputs | Execution-time refresh remains an unresolved limitation | Diagnose that boundary before changing training |
| No improvement or failed equivalence checks | The proposed explanation is unsupported or the probe is invalid | Retain current training settings and inspect the failed check |

No outcome proves improved benchmark return. This is one late development
checkpoint and one rollout. Repeated windows are not independent training seeds.
Any production change would require its own tests, native smoke, matched fresh
learning controls and fixed-endpoint evaluation. Existing jobs and their frozen
source are preserved.

## Next implementation candidate

Test an optional auxiliary-first update that freezes past-history filter masks
before the auxiliary step, re-encodes the updated prior from those same past
inputs, and evaluates PPO against the original collected log probabilities and
peer gates. Keep the existing default schedule and validate its exact replay.
The first candidate should keep prior RL learning rate zero, matching this
probe; its interaction with joint-prior gradients needs separate validation.

For sequential training, perform the auxiliary step immediately before **that
observer's** PPO update, rather than updating every observer's encoder up front.
Otherwise not-yet-updated actors would already change through their encoders,
while the predecessor correction assumes those actors retain their behavior
policies. Preserve the random agent order and include each processed actor's
encoder change in its likelihood correction. Separate per-observer models make
this ordering possible. Add a test that unprocessed actors stay unchanged and
that predecessor factors include the refreshed context.

Required evidence before a learning claim: information-boundary and gradient
tests, native continuous/categorical smoke, a timing check, matched fresh
development runs with fixed endpoints, and independent training-seed evidence.
Normal training will consume randomness in a different phase order; the paired
diagnostic does not prove identical future trajectories or select a favorable
random stream. The optional implementation is now present and documented in
[AUXILIARY_ORDER.md](docs/AUXILIARY_ORDER.md). The full unit suite passes
543 tests with 32 native-only skips; native GPU validation passes for Walker
and SMAC, including exact default replay over 8,000 transitions per task.
Matched A100 three-arm learning comparisons are submitted. No benchmark
learning result is available for this candidate yet.
