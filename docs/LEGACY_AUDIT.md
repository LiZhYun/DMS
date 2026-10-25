# Historical initial audit

This preserves the first static audit of the legacy code. Local file access has since been restored through approved execution; the manuscript is now extracted and the new implementation lives in `src/dms/`. See PAPER_CONTRACT.md for authoritative manuscript findings. The initial 'implementation gates' below describe the starting plan, not completed results.

# DMS initial audit — 2026-10-25

## Scope and status

User objective: reproduce accepted MARL papers in one readable community repository.
Full redesign is authorized while preserving each paper's core idea. Performance
above the published results is a release target; it has not been demonstrated.

This report concerns DMS only. Two other Codex sessions are working in the same
`/scratch/work/liz23/reproduce` folder. This session owns only this DMS-specific
evidence directory; no shared code or other paper implementation is changed.

Source: https://github.com/LiZhYun/DMS
Audited commit: `ed94ab0795d794f8bf43cab54ec4737f33319af3`.
Evidence below is static source inspection, not execution or measured performance.

The supplied manuscript archive is:
`Coordination_as_Inference_in_Multi_Agent_Reinforcement_Learning__elsevier___revise_2_.zip`.

Local file-reading commands currently fail before running:
`bwrap: Can't mkdir parents for /l/tmp-inst/liz23/codex-daemon-3278648: Permission denied`.
This prevents manuscript extraction, workspace inspection, skill-file reads, tests,
and training. Reading the user-requested
`/home/liz23/.agents/skills/triton-gpu/SKILL.md` failed with the same error.
No Triton job has been submitted. Memory retrieval also failed; memory writes
reported a malformed database, so durable memory persistence is unverified.

## Method contract: verified outline, manuscript details pending

The publisher abstract and indexed introduction describe independent learners,
variational inference of other agents' intentions, and counterfactual causal
filtering of relevant intentions. Selected intentions condition policy and value
networks; a soft-attention variant is also described. The introduction claims no
centralized critic, explicit communication, or shared actions/observations.
Exact permitted observations, inference inputs, timing, equations, and losses
must be checked against the local manuscript before implementing the method.

Primary publication:
https://www.sciencedirect.com/science/article/pii/S0893608024000157
DOI: 10.1016/j.neunet.2024.106101

SMAC and Multi-Agent MuJoCo are verified benchmark families. The complete task
roster, numerical targets, seed count, budgets, versions, and evaluation rules
remain unverified. Repository plot assets contain baseline curves but no DMS
curve, so they cannot define DMS reproduction targets.

## High-confidence source defects and risks

All source paths below refer to the pinned commit.

1. **Repeated training seed.** Bundled launch scripts vary `--running_id` but
   omit `--seed_specify`; both training entry points reset the seed to 8888.
   The SMAC `--seed_specify` branch additionally accesses `runing_id`, a typo.
   Evidence: `scripts/train_mujoco.sh`, `scripts/train_smac.sh`,
   `scripts/train/train_mujoco.py`, `scripts/train/train_smac.py:80–84`.
   Fix contract: one explicit seed, persisted in every run manifest; reject
   duplicate seeds when aggregating purported independent runs.

2. **Bundled commands disable evaluation.** `--use_eval` is a `store_false`
   flag with default true, but the launch scripts pass it.
   Evidence: `configs/config_mujoco.py`, `configs/config_sc2.py`, both scripts.
   Fix contract: unambiguous positive/negative booleans and resolved-config output.

3. **Evaluation may never terminate.** Both runners count each evaluation
   worker once and never clear its counted flag. Requested episodes greater
   than workers cannot complete; SMAC defaults include 1 worker and 5 episodes.
   Evidence: `runners/separated/smac_runner.py:223,278–289`,
   `runners/separated/mujoco_runner.py:222,290–299`.
   Fix contract: count completed episodes, support repeated episodes per worker,
   and test a budget greater than worker count.

4. **SMAC rollout insertion has an invalid argument list.** It passes
   intention recurrent state twice, conflicting with `train_share_obs`.
   Its intention-state transpose also uses four axes where the corresponding
   rollout tensor needs five.
   Evidence: `runners/separated/smac_runner.py:185–189`,
   `utils/separated_buffer.py:107–108`.
   Fix contract: named structured rollout fields and checked shapes.

5. **Previous-action context is temporally misaligned.** MuJoCo collection
   reads `last_actions[step]` before insertion writes the just-executed action
   into that same index. Initial rollout entries are zero; later entries can
   contain stale preceding-rollout context. Training then sees overwritten data.
   Evidence: `runners/separated/mujoco_runner.py:127`,
   `utils/separated_buffer.py:71,124,169`.
   Fix contract: store context for each observation explicitly at matching time
   indices, with episode-reset semantics and a rollout/reevaluation identity test.

6. **Recurrent minibatches mix sequence identities.** Chunks are stacked as
   [chunks,time,...] and flattened directly, while the RNN reconstructs
   [time,chunks,...].
   Evidence: `utils/separated_buffer.py:490–540`,
   `algorithms/utils/rnn.py:34–42`.
   Fix contract: one documented axis convention and sentinel sequence-order tests.

7. **Timeout semantics are lost.** SMAC sets `self.bad_transition` but exports
   a separate false local variable; the MuJoCo runner drops its timeout mask.
   Vector wrappers replace terminal observations with reset observations.
   Evidence: `envs/starcraft2/StarCraft2_Env.py:470,574–589`,
   `runners/separated/mujoco_runner.py:183`,
   `envs/env_wrappers.py:307–314,733–739`.
   Fix contract: explicit terminated/truncated flags and final observations;
   bootstrap time limits from final state and stop recurrence across resets.

8. **Inference tensor axes are inconsistent.** Posterior global-state
   `repeat(num_agents,1)` conflicts with batch-major graph nodes. Decoder
   reshaping [batch*agents,samples,features] directly into
   [batch*samples,agents,features] mixes agent and sample identities.
   Evidence: active `algorithms/encoder_decoder.py`.
   Fix contract: named dimension comments, explicit permutations, and tests with
   distinct batch/agent/sample counts and sentinel values.

9. **Reconstruction objective has a trivial affine collapse.** The same
   trainable output LayerNorm normalizes predictions and targets. Its parameters
   belong to the reconstruction optimizer. Zero scale makes both outputs equal
   to the bias, independent of whether states were reconstructed.
   Evidence: `algorithms/encoder_decoder.py:398,450`,
   `algorithms/happo_trainer.py:75`, `algorithms/happo_policy.py:70`.
   This is an analytic degeneracy; actual collapse has not been measured.
   Fix contract: target definition independent of trainable prediction parameters.

10. **Stochastic PPO likelihood contract is unresolved.** Rollout and
    reevaluation independently resample latent intentions, so unchanged weights
    and history need not give an importance ratio of one. The implementation
    neither reuses latent/base noise nor evaluates a marginal action density.
    Fresh sampling alone does not prove every possible estimator invalid.
    Evidence: `algorithms/encoder_decoder.py:142–143`,
    `algorithms/happo_policy.py:122–127,184–189`,
    `algorithms/happo_trainer.py:283–287`.
    Fix contract: derive the stochastic-policy likelihood first, document its
    conditioning, then test the appropriate unchanged-policy ratio invariant.

## Implementation and evaluation gates

These are proposed engineering gates, not claims about the manuscript protocol.

1. Restore local tool execution; inspect workspace instructions, current files,
   and other sessions' ownership before editing shared infrastructure.
2. Read the manuscript and build a claim ledger: equation/component, benchmark,
   metric, reported target, environment version, training budget, seed protocol,
   evaluation rule, and exact source location. Unknown values remain unknown.
3. Read the Triton GPU skill and use its actual site-specific instructions.
   Profile a short CPU/environment and GPU training smoke run before allocating
   a larger experiment matrix.
4. Build one transparent end-to-end path: environment adapter, rollout buffer,
   return estimation, policy/distribution, optimizer, evaluation, logging.
   Put DMS-specific intention inference, causal filtering, and objectives in a
   small method module. Coordinate shared interfaces with the other two sessions.
5. Verify meaningful invariants before benchmarking: axis identity, temporal
   context, reset masks, time-limit returns, action masking, gradients and
   optimizer ownership, stochastic likelihood, and evaluation completion.
6. Run a shared IPPO baseline and DMS ablations under identical infrastructure.
   Preserve independent-learning information restrictions and demonstrate the
   contribution of inference and causal coordination.
7. Separate development seeds/configuration selection from final evaluation.
   Save resolved config, code revision, environment versions, seeds, logs,
   checkpoints, resource use, and evaluation episodes for every run.
8. Compare against manuscript targets using the same metric and compatible
   benchmark conditions. Report independent-seed uncertainty and all planned
   tasks. Mark changed versions/budgets as a separate comparison.
9. Claim higher performance only after experiments support it. A higher point
   estimate alone is weak evidence; published 100% win rates cannot be exceeded
   on that metric and require an explicitly chosen additional criterion.
10. Publish the repository only after faithful method behavior, runnable
    reproduction commands, documentation, and performance evidence are present.
