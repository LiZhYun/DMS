# Residual intention inputs

Set `intention_fusion` to `"residual"` in a configuration to retain the local
IPPO actor/critic architecture while learning to use inferred intentions. The
default is `"concat"`, which keeps the preceding trainer's architecture and
initialization. This is an experimental redesign, not a measured improvement.

For one observer, let `x_actor` and `x_critic` be its existing local features:
normalized observations, or separate local-history GRU outputs. Let `c` be the
context computed from locally inferred peer intentions. Soft coordination pools
selected peers with attention; all-peer and causal coordination flatten their
selected contexts. The two input paths become:

```text
actor input  = x_actor  + actor_projection(c)
critic input = x_critic + critic_projection(c)
```

Both projections are bias-free linear maps initialized to zero. The actor and
critic retain their original local input width and weights. With no selected
peers, the correction remains exactly zero even after learning. Each projection
uses its corresponding actor or critic learning rate. Attention query/key
parameters retain the preceding shared design and actor-rate parameter group.
PopArt still adjusts only the final critic layer, preserving raw value outputs.

Optional-module initialization preserves the CPU random-number-generator state.
The initial local parameters, policy outputs, value outputs and sampling RNG
therefore match IPPO for the same seed and settings. This promise applies before
learning: auxiliary posterior sampling and learned corrections subsequently
change the trajectory. It does not promise exact IPPO learning when inputs are
disabled throughout training.

The information boundary is unchanged. Priors use locally observed past actions;
future actions remain auxiliary targets. Selection masks are cached during PPO.
Residual fusion supports soft, all-peer and causal selection. Initial zero
corrections give zero causal intervention KL, so the existing all-peer causal
warmup is needed to learn sensitivity before applying a positive hard threshold.
The existing restriction on joint prior RL gradients with causal selection
still applies. These settings are separate from the choice of fusion.

The motivation comes from completed development trials, not an identified
implementation defect. A DMS model trained without intention inputs improved
Walker and SMAC 8m_vs_9m but still trailed IPPO; those policies also had different
input widths and initialization from IPPO. Lowering auxiliary updates and
changing peer pooling did not resolve the target gaps. Residual fusion removes
that initial local-network difference while keeping a learnable intention path.

The initial [unit evidence](../reports/residual-unit-validation.json) records 25 focused
checks and **516 passing tests**, with 32 native dependency checks skipped.
Tests cover initial parameter/RNG/action/value equivalence, local-only behavior
with absent peers, exclusion of invisible histories and future targets,
independent actor/critic gradients, PopArt and checkpoint round trips. The [native A100 validation](../reports/residual-native-validation.json) passed eight short training runs, exact initial IPPO rollout equivalence on both tasks, exact archived/current default regression, and four saved-source CPU evaluations with zero optimizer steps and unchanged state.
The [installed-wheel check](../reports/public-wheel-residual-validation.json) also passed, using a new installation target with existing dependencies and a small synthetic residual-plus-joint run. No benchmark learning result is available for this option yet.

The submitted comparison uses training seed 101, 3M transitions, H100, and a
fixed 100-episode development evaluation block (400101–400200). Each task has
an intention-enabled residual model and an otherwise matching model whose
`warmup_episodes=1000000000` keeps peer inputs disabled throughout training.
Auxiliary models still train in that ablation. Walker uses local history 32;
SMAC 8m_vs_9m uses history 16 and gamma 0.95. Both keep auxiliary LR 1e-3 and
prior RL LR zero. No held-out training seeds are used, and no checkpoint is
selected by evaluation return.

The [queue ledger](../reports/development-residual-fusion-3m-seed101.json) records training jobs 21280354/56/58/60 and dependent fixed-checkpoint evaluations.

## Causal-selection validation

The [subsequent causal check](../reports/residual-causal-native-validation.json)
passed **27 focused residual tests** and native checks on H200. Two new tests
compare learned residual peer-removal scores with explicit Gaussian and
categorical KL formulas. Native all-peer models trained for 8,000 Walker steps
and 1,280 SMAC steps were then checked on fresh local histories. The source
checkpoint and original model remained unchanged.

| Short-trained policy | Eligible KL range | Selected at threshold 0.6 |
| --- | ---: | ---: |
| Walker3x2, history 32 | 0.000245–0.002041 | 0 |
| SMAC 8m_vs_9m, history 16 | −3.69e−7–4.66e−7 | 0 |

These budgets differ, and the magnitudes should not be compared as a task
performance result. Negative KL values reveal float32 rounding; SMAC's
zero-threshold selections do not establish meaningful influence. Every
predeclared positive threshold (0.01, 0.1, 0.2, 0.6) selected no peers in these
short-trained checkpoints. No threshold is promoted.

Separate cloned models also collected native causal rollouts at thresholds zero
and 0.6, with causal warmup disabled only for this diagnostic. PPO replayed the
stored log probabilities exactly with the original collection batch shape, and
both updates stayed finite. This verifies the selection and update path; longer
training must still establish useful selective coordination. The first full
suite's 516 passing tests precede the two additional causal tests, while the
production source used by these checks is identical.
