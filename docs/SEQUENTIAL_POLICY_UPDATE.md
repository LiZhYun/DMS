# Optional sequential policy update

`update_mode="independent"` remains the default. Set
`"update_mode": "sequential"` in a training configuration to use a HAPPO-style
predecessor likelihood correction. Use the same mode for the no-intention
baseline and DMS. This is an optimizer comparison; it is not evidence of a
performance improvement.

## The update

For each rollout, draw one random permutation of agents. Begin with correction
`F[t,e] = 1` for each time/environment row. Finish all PPO epochs for the first
agent before training the next. Agent `i` minimizes the actor objective

```text
-mean(F * min(r_i * A_i, clip(r_i, 1-epsilon, 1+epsilon) * A_i))
r_i = pi_i_new(stored_action | cached_input) / pi_i_rollout(stored_action | cached_input)
```

Local advantages are standardized before multiplication by `F`. Entropy and
value losses are unweighted. After that agent's complete update, reevaluate its
stored actions and multiply `F` by its new-to-rollout likelihood ratio. Each
continuous distribution already returns the joint log probability over that
agent's real action coordinates, including heterogeneous action widths.

The implementation accumulates detached log ratios and exponentiates them
without clipping or normalizing the correction. Nonfinite values and underflow
to zero raise an error. Reported factor minimum, maximum, and effective sample
size describe the final product after all agents, including the last agent;
they are diagnostics, not additional training transforms. Every update records
its agent permutation.

## Fixed information and inactive rows

Reevaluation uses the same cached normalized observations, prior intentions,
coordination gates, available-action masks, and bounded observation windows
that generated the rollout. It uses the updated actor and attention parameters.
Intention auxiliary learning occurs only after every agent finishes PPO, and
no policy gradient enters the intention model. Actors and critics still receive
only their local inputs.

The trainer does access teammates' exact detached action likelihood ratios.
This is coordinated training information, explicitly recorded in the manifest;
it is additional to the strict independent default. It neither broadcasts
private observations into actors nor introduces a centralized critic.

For SMAC, a dead predecessor contributes ratio one regardless of
`mask_inactive_agents`. With masking enabled, each current agent trains on its
own active rows while retaining the full rollout indices needed to select
`F`. An agent with no active rows skips optimizer and normalization updates.
The existing team GAE and individual-death semantics are unchanged.

## Relationship to the paper

The manuscript uses random agent order and a sequential policy-ratio
correction. This implementation restores those optimizer features while
retaining the documented deterministic prior means, fixed rollout inputs,
separate auxiliary updates, and bounded-history redesigns. It does not copy
the legacy code's freshly resampled latent likelihoods or interleaved auxiliary
updates, and it does not claim a HAPPO monotonic-improvement guarantee.

The code path for `independent` draws no extra random permutation and keeps
its original operations. An exact comparison against the preceding committed
trainer checked metrics, model/optimizer states, and Python/NumPy/Torch random
states across discrete, continuous, bounded-memory, masked-SMAC, and active-DMS
fixtures. Native validation and experiment evidence are tracked in
[STATUS.md](STATUS.md); software checks alone do not establish learning quality.

Native validation jobs **21198407–21198409** completed successfully: **465 tests
passed** in the modern environment (32 legacy-only skips), **56 native legacy
tests** passed, and sequential DMS completed 128 HalfCheetah 3x2 transitions and
512 original-SMAC transitions with CUDA, local memory, and inactive-unit
masking. The [validation ledger](../reports/sequential-update-validation.json)
records exact source hashes, import checks, factor diagnostics, and checkpoints.
