# SMAC training diagnosis

This is a development diagnosis of the seed-101 `3s5z` pilots, not evidence of
paper-level performance. A concrete omission in PPO's treatment of dead agents
has been corrected behind an opt-in configuration field. Its effect on battle
win rate remains unmeasured. Local policy memory, timeout semantics, and the
redesigned optimization settings remain separate experimental questions.

## Observed learning

The completed IPPO run, job **21195882**, trained for exactly 1,000,000 environment
transitions in 1:29:47. It completed 10,689 training episodes and 16 scheduled SC2
process restarts. Its final deterministic evaluation won **0 of 10 battles**;
mean return was **12.350165695953184**, population standard deviation
**2.4410487768195366**. The actual imported source hashes match its saved source.
The run is
[`dev-smac-3s5z-ippo-1m-threaded-101`](../runs/dev-smac-3s5z-ippo-1m-threaded-101/).

Earlier evaluations show that zero wins did not mean zero reward learning:

| Run / checkpoint transitions | Mean evaluation return | Wins / episodes | Episodes reaching limit |
| --- | ---: | ---: | ---: |
| IPPO / 200,320 | 7.3253 | 0 / 10 | 7 |
| IPPO / 400,000 | 1.4121 | 0 / 10 | 9 |
| IPPO / 600,320 | 10.0661 | 0 / 10 | 8 |
| IPPO / 800,000 | 10.7164 | 0 / 10 | 6 |
| DMS, history decoder, beta 1 / 600,320 | 8.0940 | 0 / 10 | 10 |
| DMS, latent-only decoder, beta .1 / 200,320 | 9.9098 | 0 / 10 | 9 |

The DMS snapshots are from the continuing development runs
[`dev-smac-3s5z-dms-soft-1m-threaded-101`](../runs/dev-smac-3s5z-dms-soft-1m-threaded-101/)
(job 21195883) and
[`dev-smac-3s5z-dms-latent-decoder-beta-01-1m-threaded-101`](../runs/dev-smac-3s5z-dms-latent-decoder-beta-01-1m-threaded-101/)
(job 21196342). Their intermediate results are not final results. All three use
four environments, 160-step rollouts, 640-sample minibatches, five PPO epochs,
and the same native map asset, SC2 build 75689, and difficulty 7. See their raw
`config.json`, manifests, metrics, and evaluation files for exact settings.
Workspace `runs/` links require the corresponding artifacts and are not bundled
with a clean source checkout.

One million transitions is one tenth of the paper's plotted 10-million budget.
Ten deterministic evaluation episodes also provide limited evidence. The old
metrics did not log completed training battle outcomes, so zero evaluation wins
cannot establish zero stochastic training wins. Rising damage rewards and finite
PPO losses make ordinary early learning plausible, but do not demonstrate that
simply extending every run will solve the tasks.

## Verified omission and opt-in correction

SMAC gives a dead agent only action 0, the no-op. Before this change, `update()`
included these rows in advantage normalization, mean actor/critic/entropy losses,
the approximate KL used for early stopping, and observation/value moments.
Masking unavailable actions correctly made the dead actor's own policy gradient
zero. It did not stop dead samples from changing live-sample statistics, diluting
the actor's effective loss weight, or contributing critic gradients to the shared
gradient-norm bound.

The observation-history-present bit in saved running observation means offers a
useful exposure check. Its average over agents was .6367 at IPPO's 800,000-step
checkpoint, .6495 at the history-decoder DMS 600,320-step checkpoint, and .6721 at
the latent-only DMS 200,320-step checkpoint. The complementary fractions include
both dead observations and episode resets; they are **not exact dead fractions**.
They nevertheless show that absent-history observations were common, rather
than a negligible corner case. The checkpoint value standard deviations were
finite, approximately 2.0–2.23, with no observed value-scale explosion.

Set `mask_inactive_agents: true` in a new SMAC development config to enable the
correction. The default remains `false` so loading older configs preserves their
numerical training behavior. The option has no effect on toy or continuous tasks.

The implementation:

- Derives alive status from the availability of any non-noop action, only for
  `env='smac'`.
- Filters each agent's rows before advantage normalization and minibatch
  generation. Actor loss, entropy, value loss, PopArt target moments, observation
  moments, and approximate actor KL then use only its alive rows.
- Skips an entirely inactive agent's PPO optimizer and moments. Clearing its
  gradients without calling `Adam.step()` also prevents a momentum-only update.
  If all agents are inactive, PPO loss metrics are zero and the update count is
  zero rather than NaN.
- Leaves team rewards, GAE recursion, battle boundaries, and time-limit bootstrap
  semantics unchanged. Individual death does not end the shared battle return.
- Leaves the separately optimized intention objective and its visibility masks
  unchanged; the all-inactive skip guarantee concerns the PPO optimizer.

The original checkout supports the decision to mask live policy samples:
`runners/separated/smac_runner.py` constructs individual active masks;
`algorithms/happo_trainer.py` excludes dead rows from advantage statistics and
uses active-mask policy reductions. The SMAC config defaults policy masking to
true and value masking to false; its provided `scripts/train_smac.sh` explicitly
passes `--use_value_active_masks`, enabling value masking for that recipe.
The manuscript does not specify this flag. Excluding dead rows from the new running observation
and PopArt moments is an explicit redesign choice; the original input
normalization used LayerNorm, and its internal normalizer updates are not assumed
identical. These source locations refer to the pinned upstream DMS checkout
described in [LEGACY_AUDIT.md](LEGACY_AUDIT.md).

Critic masking is a consequential choice. Team-return GAE continues after an
individual agent dies. With lambda below one, values at dead states can still
influence an alive action's pre-death advantage even though those states receive
no direct critic supervision under the opted-in filter. At lambda one, internal
value terms telescope over a completed trajectory; that cancellation is not
complete at the configured lambda .95. Following the supplied shell recipe does
not establish that this is optimal for the redesigned local critic. Retaining
dead-state critic supervision should therefore be a separately controlled
alternative if this bias proves material, rather than silently changing death
into a team termination. The current flag selects joint actor/critic filtering;
it must not be described as an actor-only ablation.

The synthetic regression initially failed on the unmodified trainer: adding
inactive rows changed live observation means from roughly 1.2 to 4286.7. The
focused tests now cover exact parameter and optimizer invariance under inactive
padding for IPPO and DMS, actor KL early stopping, existing Adam momentum,
partially/all-inactive batches, active normalization, unchanged toy/continuous
updates, and battle logging independent of the flag. These tests establish
update semantics, not improved SMAC win rate.

Every new SMAC run logs `alive_fraction` whether filtering is enabled or not.
`ppo_update_count` counts agent-minibatch optimizer updates. Per-rollout completed
battle counts, wins, and win rate use `training_battles`,
`training_battles_won`, and `training_win_rate`; the corresponding cumulative
fields end in `_total`. A rate is null when its denominator is zero. Scheduled
simulator restarts do not fabricate completed battle outcomes.

## Remaining diagnostics, in priority order

1. **Evaluate the mask correction with a matched IPPO/DMS pair.** Keep all other
   settings and the source snapshot fixed. Compare training wins, alive fraction,
   live entropy, greedy evaluation, and episode lengths. A finite smoke alone
   cannot show a performance improvement.

2. **Restore a tested local observation-memory option for both methods.** The
   audited pilots use feedforward actors and critics. DMS's recurrent action
   encoder does not give either policy a history of its own local observations.
   The original SMAC entry point imports `configs/config_sc2.py`, whose
   `use_recurrent_policy` default is true; its actor-critic creates and uses an
   RNN. This differs from the unrelated generic config's default. The current
   adapter preserves native local enemy/ally health, relative position, types,
   and visible last actions, but those features do not make a partial observation
   Markov. Memory could help infer changes and weapon timing; that is a hypothesis
   requiring a controlled experiment. Keep memory observer-local and reset it at
   battle boundaries and own-agent death. Do not introduce global state.

3. **Measure timeout values before choosing a finite-horizon sensitivity.** The
   current adapter explicitly enables `continuing_episode=True`; a 150-step
   limit becomes truncation and GAE bootstraps from the final local observation.
   The original wrapper defaults to false, and the SMAC config defaults
   `use_proper_time_limits` to false. Its battle-limit return treatment therefore
   differs. Frequent deterministic timeouts could interact with optimistic
   continuation values. Log timeout frequency and final value estimates, then
   test an explicitly declared finite-battle objective if warranted. This is an
   objective difference, not evidence that generic timeout GAE is incorrect;
   MuJoCo truncation behavior should not change as a side effect.

4. **Separate exploration from greedy-policy failure.** A bounded, paired-seed
   greedy/stochastic evaluation should record stop/move/attack frequencies,
   alive-only action entropy and maximum probability, enemy/ally deaths, and
   battle lengths. Some greedy episodes timed out with zero reward while nearby
   stochastic training episodes earned positive reward. Existing aggregate
   entropy includes dead zero-entropy rows, so it cannot by itself justify an
   entropy increase. The old history-decoder DMS had small posterior KL
   (about .00063 in one later snapshot), whereas the latent-only beta-.1 run had
   KL around 1.42. Latent collapse may matter to the first DMS variant but cannot
   explain IPPO's zero wins, and a healthier KL alone has not established wins.

5. **Then test optimization scale one variable at a time.** The paper lists
   20×160 collected transitions, versus 4×160 here, and actor/critic learning
   rates 1e-5/1e-4 versus 3e-4/1e-3 here. The original checkout's SMAC optimizer
   defaults are instead 5e-4/5e-4, and its shell recipe uses discount .95 versus
   the manuscript's .99. The checkout is not an unambiguous manuscript recipe.
   Batch size, optimizer rates, LayerNorm/ReLU versus running normalization/Tanh,
   and recurrence are separate differences. Recent approximate actor KL values
   were around .0001–.00026, well below the .03 stopping threshold, so premature
   KL stopping is not the leading explanation in those sampled windows.

The code audit found no duplicate reward scaling, agent-count reward inflation,
unavailable-action sampling, reset-observation bootstrap leakage, or auxiliary
future-target crossing of an episode boundary in these paths. Such negative
findings narrow this investigation; they do not certify all possible behaviors.
The next learning experiments must decide whether the identified changes improve
performance, and successful smoke tests must not be reported as reproduction.
