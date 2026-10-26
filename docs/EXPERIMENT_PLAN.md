# Experiment plan and acceptance

The goal is a readable public reproduction preserving intention inference and selective coordination, with performance higher than the paper's claims. No benchmark result has yet met that goal. [paper_targets.json](../configs/paper_targets.json) is the target ledger; source evidence and unresolved interpretation issues are in [PAPER_CONTRACT.md](PAPER_CONTRACT.md).

## Stage 1: invariants and smoke checks

Run the unit suite and the 512-step synthetic configuration. Check finite losses, meaningful updates, valid actions, explicit sequence boundaries, masked histories, and checkpoint evaluation. Then run the modern MuJoCo and official SMAC smoke configs on allocated Triton nodes. A benchmark smoke must complete collection, a PPO update, an intention update, deterministic evaluation, and checkpoint saving; starting an environment alone is insufficient.

Read `config.json`, the environment manifest, and per-episode evaluation records. Confirm that SMAC sees only native visible ally actions, MuJoCo's added sensor is recorded, timeout bootstrap preserves the final observation, and actual transitions match the chosen smoke budget. Do not interpret synthetic reward or 64/128-transition benchmark smoke scores as reproduction performance.

Measure wall time, peak host memory, GPU memory, and transitions per second separately for collection, updates, and evaluation. The templates' one-GPU/eight-core/32-GB/24-hour training request is provisional. Estimate a full run from measured throughput plus evaluation time, then allow a modest margin. Checkpoint resume does not restore environment trajectories; do not deliberately schedule a final run to exceed its wall-time limit.

## Stage 2: development on separate seeds

Use training seeds **101, 103, 107**. Begin with HalfCheetah 2x3 and SMAC 3s5z using the paired DMS/IPPO configs. Start with explicitly shorter development budgets, then move promising configurations to 10,000,000 transitions. Record every attempted setting and its budget. Keep equal sensor access, environment version, reward semantics, and evaluation settings across algorithms. Match `smac_time_limit_terminal` within each pair: its optional finite-battle objective changes value targets at the native deadline and must be reported separately from continuing-value runs.

Tune a small predeclared set: actor/critic learning rates, history length, auxiliary-loss/update strength, and coordination variant/threshold. Record negative results. Inspect held-out behavior prediction against a persistence predictor and latent/selection diagnostics alongside reward. Flat or collapsed latent features, near-zero selected-peer rates, or gains explainable entirely by a stronger PPO base require investigation.

Use `coordination="all"` for intention without selection, `"soft"` for attention, and `"causal"` for the paper's KL-based selection. Required comparisons are:

- IPPO with the same local sensors and optimizer settings.
- Intention model with all available peer intentions.
- Intention model with soft attention.
- Intention model with causal selection.
- Long/short filtering and missing-action handling sensitivity, with hidden commands still excluded in every variant.

Keep `policy_history_length` equal across a matched IPPO/DMS pair. On representative
tasks, also include an IPPO memory control over the same available observation
and action history: a feedforward IPPO comparison alone cannot distinguish the
benefit of learned intention prediction from the benefit of remembering past
inputs. Forecast shuffles and downstream interventions answer related but
different questions; they do not replace that trained control.

Do not implement a missing-action ablation by revealing hidden true controls. An ablation can instead remove the explicit missingness token or alter how missing local records are encoded, while preserving the information boundary. Check MuJoCo observation radius 0 and 1 because the manuscript contradicts itself. Record the four-environment SMAC redesign and investigate 20 environments before asserting historical optimizer parity.

## Stage 3: freeze the final experiment

Freeze code, environment versions, sensor definitions, selected hyperparameters, maps, budgets, and evaluation rules before final testing. Use six held-out training seeds **11, 23, 37, 53, 71, 89**. Choose a fixed 10,000,000-transition final checkpoint; do not choose each seed's best test checkpoint. The provided main configs divide their budgets exactly by rollout size; preserve this property when editing them.

Training-time diagnostics use evaluation seeds beginning at 10001. Final evaluation uses a disjoint seed range beginning at **500001**. Pair those episode seeds across algorithms for each task and report how randomness is controlled. Evaluate 100 MuJoCo episodes or 1,000 SMAC games for every independently trained seed. These stronger evaluation counts are the new protocol, not values claimed by the paper.

Report raw per-episode outcomes, per-training-seed means, across-seed mean/median/standard deviation, and a seed-level bootstrap confidence interval. Ten or a thousand episodes from one trained policy do not replace independent training seeds. A positive point estimate and statistically supported improvement are different claims; label them accurately. The paper's plot bands cannot supply missing raw seed results.

## Targets and environment tracks

| Task | Paper result | Initial acceptance threshold |
| --- | --- | --- |
| SMAC 3s5z | Exact reported win rate 97.5% | Above 97.5% |
| SMAC 8m_vs_9m | Exact reported win rate 82.6% | Above 82.6% |
| SMAC MMM2 | Exact reported win rate 62.1% | Above 62.1% |
| Ant 2x4 / 4x2 / 8x1 | Raster-plot endpoints, approximately 1,650 / 1,550 / 1,800 | Above 1,900 / 1,700 / 2,000 |
| HalfCheetah 2x3 / 3x2 / 6x1 | Approximately 3,500 / 4,700 / 4,100 | Above 3,800 / 5,200 / 4,300 |
| Walker 2x3 / 3x2 / 6x1 | Approximately 600 / 1,000 / 700 | Above 850 / 1,100 / 1,000 |

MuJoCo thresholds are conservative engineering targets derived from the supplied manuscript figures, not exact recovered numbers. ManyAgentSwimmer 10x2 is a supplementary parameter-study task with an engineering target above 85; its bundled standalone main-result asset is not included in the manuscript's main-result figure. Preserve that distinction in reports.

Original SMAC results require the matched StarCraft build, map files, difficulty, reward and observation flags. Modern MuJoCo v5 scores form a separate adaptation track. Exceeding an old v2 return on v5 does not establish a like-for-like paper improvement. Complete an original-environment comparison or a documented equivalence audit before closing the original MuJoCo reproduction gate. The unresolved `agent_obsk` conflict must remain stated even after both sensitivity runs finish.

## Artifacts and readiness

Each run needs its complete configuration, dependency/environment manifest, source snapshot or identifiable revision, training metrics, checkpoint, individual evaluations, job ID, and logs. Use new output directories; failed jobs remain evidence rather than being overwritten. Maintain a table of every scheduled run, seed, variant, step budget, completed steps, terminal status, and evaluation result.

The reproduction is ready only when all required environments run, all registered final seeds complete matched budgets, all primary targets are exceeded on comparable tracks, uncertainty and ablations support the stated mechanism, and a fresh reader can reproduce the commands. Code and smoke readiness may be reported separately from experimental readiness.

**Operational status, 2026-10-25:** scheduler discovery works with `SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817`. Synthetic, real MuJoCo, and real SMAC GPU smoke jobs completed successfully, each running 119 tests first. These short executions do not establish learning performance. [STATUS.md](STATUS.md) records the job IDs, budgets, evidence paths, and remaining work.
