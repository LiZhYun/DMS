# Intention objective and frozen-checkpoint diagnosis

Audit dated 2026-10-25. This records a read-only examination of the two completed historical HalfCheetah 2x3 DMS pilots at 500,000 transitions, training seed 101. It changes neither the implementation nor existing experiment configurations. Both pilots remain development evidence; their return difference does not establish the intended coordination mechanism or paper reproduction.

The main finding is that the current loss is a **time-normalized beta objective**, with different denominators for reconstruction and KL. At horizon four, its `kl_beta=1` is not the canonical sequence ELBO. The frozen probes also find little dependence on the current variation in deployed peer latent means, especially for the higher-return decoder variant.

**Evidence and scope.** The implementation examined is [intention.py](../src/dms/intention.py), together with [train.py](../src/dms/train.py), [policy.py](../src/dms/policy.py), and the [diagnostic script](../scripts/diagnose_intentions.py). The archived `source/dms/intention.py` in both pilot directories and the intention source recorded by both diagnostics have the same SHA-256, `1663eba23a90a1ff216c92b3bf7cdb2b2709a8a01a158c6b415d3dcfb2d15189`. Subsequent source revisions should be compared against those archived files before applying this audit to a new run.

The raw diagnostic artifacts are the [original decoder result](../runs/dev-legacy-halfcheetah-2x3-dms-soft-101/mechanism-diagnostic-01.json), job 21195732, and the [latent-initialization decoder result](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-101/mechanism-diagnostic-01.json), job 21195733. Both used immutable diagnostic checkpoint copies. Each collected 512 fresh transitions, comprising four 128-step episode prefixes with diagnostic seed 200001 and pairing seed 200002. Neither completed a diagnostic episode. Parameters, normalization buffers, training phase, and diagnostic source remained unchanged; there were zero parameter updates. These are short mechanism probes, not return-intervention evaluations.

**Verified information paths.** Each observer has an independent action-only CVAE, sharing its encoder across peers. The prior consumes at most 32 past locally sensed actions, visibility flags, and peer identity. The training-only posterior applies the same encoder to the past plus a completed four-step future suffix. Neither accepts observations, global state, or rewards. The analytic Gaussian KL has the correct direction, `KL(q || p)`, and sums latent event dimensions. Reconstruction uses a Monte Carlo sample mean over posterior samples.

The deployed policy receives the deterministic prior mean. Features are detached and cached during PPO; policy gradients do not train the CVAE. Auxiliary reconstruction and KL train it through a separate optimizer. Consequently, predictive information that improves auxiliary likelihood is not automatically information useful to the policy.

With `decoder_history=true`, an encoded history and latent initialize the decoder. With `decoder_history=false`, only the latent initializes it. The latter still receives the last visible action as an anchor, peer identity, the latent at every decoder step, and its own previous prediction. It is therefore accurately described as **latent-only initialization**, not a decoder with no other conditioning. Continuous means start at persistence because the residual output head is initially zero. Predicted means, rather than actual future actions, feed subsequent decoder steps; this path has no teacher-forcing leakage. Its Gaussian likelihood remains a valid conditional sequence likelihood, with means generated deterministically from past inputs and the sampled latent.

**Exact loss normalization.** For one observer's auxiliary minibatch, let:

- `M` be the number of valid future time/peer targets, after visibility and component-availability masking.
- `N` be the number of batch/peer sequences with at least one valid future target.
- `S` be the summed Monte Carlo expected negative log likelihood over valid targets. Continuous action event dimensions are summed within each target.
- `K` be the summed `KL(q || p)` over those `N` eligible sequences.

For nonempty targets, the implemented objective is

```text
L_current = S / M + beta * K / N
          = (S + beta * (M / N) * K) / M.
```

A canonical masked sequence negative ELBO averaged over eligible sequences is `(S + K) / N`. Expressing that same summed objective in units per valid target gives `(S + K) / M`. The current objective instead has effective sequence KL weight `beta * M / N`. With complete visibility and prediction horizon `H=4`, `M=4*N`:

| Configured beta under the current reduction | Effective sequence KL weight |
| --- | ---: |
| 1 | 4 |
| 0.25 | 1 |
| 0.1 | 0.4 |
| 0.01 | 0.04 |

Thus the existing 0.1 and 0.01 trials are valid controlled regularization studies. Their interpretation does not depend on calling the objective a standard ELBO. No result needs to be discarded or silently relabeled with a different beta.

The visibility caveat matters: continuous reconstruction requires two consecutively observed action endpoints, including the final past endpoint for the first future target. Missing endpoints and wholly unobserved peers alter `M/N`. A fixed `beta=1/H` only recovers the canonical relative weighting when every eligible sequence contributes exactly `H` targets. The correct factor with missing targets is the actual minibatch ratio `N/M`, not the nominal horizon. When there are no valid targets, both contributions should remain zero. This audit treats the likelihood as conditional on the supplied observation mask; it makes no missing-at-random claim.

**Recommended minimal correction, not yet implemented.** Normalize the KL sum by the same valid-target denominator as reconstruction:

```text
kl_raw_per_sequence = K / max(N, 1)
kl_for_objective    = K / max(M, 1)
loss               = nll + beta * kl_for_objective
```

At `beta=1`, this is the masked sequence ELBO in per-target units. The denominator remains a minibatch normalization, so variable mask density also affects weighting between minibatches; it should not be described as an exact equal-weight average over sequences. Keep the existing raw per-sequence KL metric for comparability, and separately report `kl_for_objective`, valid target/sequence counts, and beta. This requires no change to the prior, posterior, decoder, permitted information, or continuous event-dimension sum.

Use an explicit configuration choice such as `kl_reduction="per_sequence"|"per_target"`. Loading an old configuration without that field must retain `per_sequence`, while new configurations select their reduction explicitly. Preserve the chosen reduction in checkpoints and run manifests; never reinterpret an archived beta value. The current loss docstring's unqualified “Negative ELBO” should eventually be replaced with the precise objective description. This document does not edit that code or the paper contract.

**What the frozen probes measured.** Pairing uses the same observer and peer, identical visibility patterns, and distinct sensed action histories. The policy probe changes eligible peer prior means while preserving observations, own-query latent, relevance gates, and action masks. The decoder probe changes prior means while preserving the recipient history, last-action anchor, and peer identity. Temporal variation is centered per coordinate within each environment episode and excludes warmup placeholders. Reported shuffling therefore tests the information in the learned deployment means on these collected prefixes.

| Measurement | Original decoder | Latent-only initialization |
| --- | ---: | ---: |
| Final mean return, five evaluation episodes | 2,372.8025 | 3,409.0147 |
| Peer temporal RMS coordinate std, observer 0 → peer 1 | 0.007130 | 0.002581 |
| Peer temporal RMS coordinate std, observer 1 → peer 0 | 0.028675 | 0.003686 |
| Mean policy KL after peer shuffle, observer 0 | 1.5485e-5 | 4.2760e-8 |
| Mean policy KL after peer shuffle, observer 1 | 2.2176e-4 | 2.9001e-7 |
| Peer decoder prediction-change RMSE, 0 → 1 | 3.6854e-4 | 4.6555e-5 |
| Peer decoder prediction-change RMSE, 1 → 0 | 2.4846e-3 | 2.6660e-4 |
| Peer forecast MSE change after shuffle, 0 → 1 | +2.6524e-6 | −8.0466e-7 |
| Peer forecast MSE change after shuffle, 1 → 0 | −5.2005e-5 | +3.4094e-5 |

The return values are independently saved evaluation means, not returns from the shuffled policies. Their five episodes are not independent training-seed replicates. See the [historical pilot plot and raw series](../reports/historical-halfcheetah-pilots.json).

Each temporal-variation estimate uses 508 visible steps across four episode prefixes. Policy comparisons use 508 matched samples per observer for the original decoder and 506 for latent-only initialization. Each peer decoder comparison uses 247 matched histories and 988 target components. These overlapping windows are not independent statistical replicates. Very small float32 policy KL values can approach numerical resolution; their exact positive magnitude should not be overinterpreted.

Both models improve aggregate forecasting over simple baselines on their own fresh trajectories: original decoder RMSE 0.5060 versus persistence 0.9005 and history mean 0.6387; latent-only initialization RMSE 0.5734 versus 0.9591 and 0.6786. That is evidence of predictive fit, but it does not identify the latent as its cause. The models generated different on-policy data, so those RMSE values are not a paired comparison of model quality. Aggregate component metrics also weight each observer's three own-action components against only one sensed peer component; own-action prediction contributes three quarters of the available component targets. Peer-only results are necessary for coordination claims.

**What is and is not established.** The higher-return variant has less temporal peer-latent variation and smaller policy/decoder responses to shuffling. Peer forecast-loss changes are small and mixed in sign. The current evidence therefore does not attribute its improvement to informative, changing inferred peer intentions. It also does not establish that all latent information is absent: swapping nearly constant learned means is a narrow intervention; it does not test arbitrary latent perturbations, posterior samples, latent variances, or whole-episode consequences. A small learned `KL(q || p)` alone is not proof of posterior collapse: a conditional prior can carry substantial history information while closely matching the posterior.

In this two-agent task, excluding self leaves at most one peer. Soft attention then assigns that visible peer weight exactly one; when none is visible it produces zero context. The query/key scores cannot select between peers in this setting. This experiment cannot establish the benefit of learned attention selection.

The training metric `intention_feature_std` mixes variation across latent coordinates, identities, observers, and time. Large constant coordinate offsets can inflate it. Likewise, the soft-mode peer gate fraction mostly measures visibility, not learned selection. Neither is evidence of meaningful temporal intention information.

There are several plausible explanations, still hypotheses. The effective KL weight of four may discourage information specific to future actions. The original decoder can predict through its encoded-history path, and removing that path leaves persistence, autoregression, peer identity, and potentially static latent offsets available. Stochastic actions may reward a smooth conditional-mean predictor without a strong changing latent code. Fairly augmented local actor observations already contain sensed previous actions, which can make short-horizon intention features redundant. Constant features can still alter network optimization, and DMS has a larger actor input than IPPO.

The architecture comparison also does not preserve identical random streams merely by using seed 101. Changing the decoder initialization matrix changes how many random values initialization consumes. Actors are constructed before intention models, but later model initialization and subsequent sampling share the PyTorch random stream. That dependence is verified in the code; whether it explains the return difference remains unknown. One training seed cannot separate architecture effects from these training-trajectory differences.

**Two controlled next experiments.** These are proposals; this audit submits no jobs.

1. **Match objective denominators.** Keep the latent-initialization architecture, horizon, visibility, optimizer settings, and environment fixed. Compare current `per_sequence, beta=1` against explicit `per_target, beta=1`; under complete four-step targets the latter is equivalent in relative weighting to legacy beta 0.25. Existing beta 0.1/0.01 runs remain separately labeled controls. Pair model initialization and initial rollout random state across conditions, then repeat the selected comparison on development seeds 101, 103, and 107 before using held-out seeds. Record both KL metrics, `M/N`, peer-only forecast errors, temporal variation, shuffle-loss changes, and per-training-seed returns. Do not change the likelihood, horizon, or sensing rule in this comparison.
2. **Test temporal latent information in frozen policies over complete episodes.** For each checkpoint, compare its ordinary prior means with a fixed mean for each observer/peer estimated from separate calibration trajectories. Preserve identity offsets, own-query treatment, visibility, and gates; apply the intervention only when that peer would contribute normally. Use fresh paired development episode seeds and explicit matched action-random streams, with no parameter or normalization updates. Measure complete-episode return differences and peer-only forecast differences on a common held-out buffer. A calibration mean avoids the arbitrary scale shift of zeroing all features. This directly tests the usefulness of changing deployed means for the current policy, while still not proving why one training run outperformed another.
