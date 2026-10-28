# Joint intention training

`intention_rl_lr=0` preserves the frozen-feature trainer used by the existing
experiments. A positive learning rate enables policy and value gradients into
each observer's intention prior. This is a development option; initial single-seed gains are small and unreplicated. The manuscript permits these gradients (source
line 486); its exact soft-update scheme remains unspecified and is not added.

During rollout, the learner still acts from the prior mean computed from locally
observed past actions. It saves the behavior log probability and value, normalized
observation, context and selection gate. Before that observer's PPO update, it
reconstructs the history-filter mask using the unchanged rollout encoder. Each
minibatch then computes new differentiable means from the same past and mask.
PPO therefore compares the updated actor **and encoder** with the saved behavior
probability. Future actions never enter this path. Sequential predecessor ratios
also use the updated encoder and the original filter mask.

Actor, value and entropy losses share the prior encoder, layer normalization and
Gaussian head. Decoder parameters receive only the auxiliary loss. The existing
PPO Adam optimizer has an additional prior parameter group at `intention_rl_lr`;
the auxiliary optimizer retains `intention_lr` and its own moments. Joint PPO
gradients are clipped together. Auxiliary updates run after every observer's PPO
update is complete. A minibatch without eligible peers bypasses the encoder so
Adam momentum cannot change it through a nominally zero gradient. Inactive SMAC
rows remain excluded when that option is enabled.

The initial option supports `coordination=soft` or `all`. It rejects hard causal
selection because encoder changes can change the intervention threshold's meaning;
that combination needs its own validation. History filtering remains fixed during
PPO, so this is a conditional rollout surrogate, not differentiation through the
threshold. At the next rollout the ordinary filter is recomputed. IPPO has no
intention model and ignores this setting, just as it ignores auxiliary settings.

The current motivation is diagnostic, not a confirmed defect: on the frozen
3M SMAC gamma-0.95 policy, holding each peer's first eligible latent gave 57/100
wins versus 56/100 with normal refresh. That result does not show a benefit from
refreshing peer latents in this block, but it does not establish equivalence or
prove that RL gradients will help. The existing fixed-feature controls and their
negative results remain part of the record.

Validation: the complete modern suite passes 491 tests (32 opt-in legacy native
checks skipped). New invariants cover both RL objectives, unavailable actions,
future-target exclusion, warmup and reset likelihoods, inactive rows, sequential
correction and existing Adam momentum. Four native 8,000-step default-regression
runs reproduce the preceding public trainer exactly. Four enabled native smokes passed on CPU and V100 CUDA, followed by two-game
saved-source evaluations for each checkpoint with zero optimizer steps and
unchanged parameters/normalizers.

Two A100 3M development trials at RL LR 3e-4 completed, each evaluated on
100 fixed episodes after training seed 101:

| Task | Frozen-feature DMS | Joint prior DMS | Matched IPPO |
| --- | ---: | ---: | ---: |
| HalfCheetah 2x3, policy history 32 | 4,332.78 | 4,409.99 | 5,179.43 |
| SMAC 3s5z, gamma 0.95 | 56/100 wins | 59/100 wins | 81/100 wins |

These are descriptive development differences, not evidence of robust improvement.
No new default is selected. [Declaration and results](../reports/development-joint-intention-3m-seed101.json).


The subsequent [controlled diagnosis](../reports/conditioning-diagnosis-results-20261029.json)
on H100 also completed at 3M steps and training seed 101. Walker3x2 history-32
return rose from 391.09 to 529.25 with joint gradients; SMAC 8m_vs_9m remained
at 0/100 wins. A separately trained no-input architecture ablation reached
964.63 and 38/100 wins respectively, still below matched IPPO (2,322.92 and
66/100). The ablation retains DMS initialization, actor/critic architecture and
auxiliary updates but keeps peer gates off throughout training
(`warmup_episodes=1000000000`). It avoids the distribution shift of masking
inputs in a previously trained policy. It is not a DMS performance candidate.
These results leave the joint-gradient option unselected and the conditioning
gap unresolved.
