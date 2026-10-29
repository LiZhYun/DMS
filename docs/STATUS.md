# Reproduction status

Snapshot: **2026-10-29 12:53 UTC**. **Not release-ready.** Development results exceed some
individual paper targets, but consistent improvement across training seeds,
useful intention inference against strong memory controls, and the complete
benchmark matrix remain unproven. Held-out training seeds **11, 23, 37, 53, 71,
89** remain unused. No result bundle or container has been published.

## Completed fixed-checkpoint comparisons

The table below uses **100 separate development evaluation episodes** per policy,
seeds 400101–400200, after training to the stated budget. Checkpoints, model state,
normalizers and source remain unchanged during evaluation; there are zero
optimizer steps. All rows use development training seed **101**. Episode counts
are not independent training replicates.

| Task and budget | Matched control | DMS | Outcome |
| --- | ---: | ---: | --- |
| HalfCheetah 2x3, CUDA, 3M, no policy memory, soft attention | 4,075.60 | 4,434.98 | +359.38; episodes below 1,000: 1 / 0 |
| HalfCheetah 2x3, CUDA, 3M, history-32 GRU, soft attention | 5,179.43 | 4,332.78 | −846.65; episodes below 1,000: 1 / 0 |
| HalfCheetah 2x3, 10M, history 32, matched A100 | 5,724.01 | 4,609.94 | Both above 3,800; DMS remains below the stronger memory control |
| HalfCheetah 3x2, 3M, history 32, H100 | 5,764.90 | 5,248.65 | DMS above 5,200 target on one development seed |
| HalfCheetah 6x1, 3M, history 32, H100 | 5,480.36 | 4,651.34 | DMS above 4,300 target; two episodes below 1,000 |
| SMAC 3s5z, 3M, gamma 0.99, continuing value | 23/100 wins | 0/100 wins | Higher DMS shaped return does not imply more wins |
| SMAC 3s5z, 3M, gamma 0.99, terminal deadline | 13/100 wins | 8/100 wins | DMS below matched baseline |
| SMAC 3s5z, 3M, gamma 0.95, A100 | 81/100 wins | 56/100 wins | Both improve over gamma 0.99 |
| SMAC 3s5z, 10M, gamma 0.95, H100 | 100/100 wins | 80/100 wins | DMS below 97.5% target |
| SMAC 8m_vs_9m, 3M, gamma 0.95, H100 | 66/100 wins | 0/100 wins | Both below full-budget target 82.6% |
| SMAC MMM2, 3M, gamma 0.95, H100 | 13/100 wins | 8/100 wins | Both below full-budget target 62.1% |
| Walker2d 3x2, 10M, no policy memory, CPU | 880.11 | 460.78 | Both below 1,100 target |
| Walker2d 3x2, 3M, history 32, H100 | 2,322.92 | 391.09 | DMS episodes average only 169.48 steps |
| Ant 4x2, 10M, native contacts, CPU | 964.76 | 950.75 | Both below 1,700 target |
| Ant 4x2, 10M, computed contacts, CPU | 606.74 | 905.45 | Separate environment adaptation |
| Ant 4x2, 3M, history 32, native contacts, H100 | 1,055.41 | 1,111.47 | Both below 1,700 target; all episodes last 1,000 steps |
| Ant 2x4, 3M, history 32, native contacts, H200 | 936.76 | — | IPPO only; below 1,900 full-budget target |
| Walker2d 2x3, 3M, history 32, H200 | 412.09 | — | IPPO only; below 850 full-budget target |

[The 3M evaluation ledger](../reports/development-fixed3m-evaluation-queue.json)
and [Walker ledger](../reports/development-walker-10m-seed101.json) retain every
individual episode and validation check. The [completed SMAC learning curves](../reports/smac-objectives-development-completed.png)
show all six full 3M trajectories. Ten-episode training diagnostics use
different seeds and must not be substituted for these endpoints. For example,
the terminal-deadline SMAC baseline had 5/10 final diagnostic wins but 13/100
wins in the separate block. CPU replay exactly reproduced the original ten
CUDA games for both continuing-value SMAC controls; it found no device
discrepancy in that replay. [Replay evidence](../reports/development-smac-evaluation-device-replay.json).
A separate [gamma-0.95 replay](../reports/development-smac-gamma095-evaluation-device-replay.json)
also exactly reproduces all ten original CUDA returns, lengths and outcomes:
IPPO **7/10**, DMS **10/10**. The 100-game endpoints remain **81/100** and
**56/100**; the original ten games do not represent that separate seed block.

**All-peer versus causal coordination:** these no-memory DMS variants share the
same concatenating actor/critic architecture. Soft attention uses a separate
pooled architecture. Returns below are fixed 3M, 100-episode means; parentheses
count episodes below 1,000.

| Training GPU family | All-peer | Threshold 0.6 | Threshold 0.2 | Threshold 0.1 |
| --- | ---: | ---: | ---: | ---: |
| A100 | 4,453.56 (0) | 3,394.71 (3) | — | — |
| H100 | 4,498.75 (2) | 4,277.42 (0) | 4,613.06 (1) | 4,228.60 (0) |

All four H100 variants have exactly matching logged warmup metrics through
200k steps before causal selection activates. Threshold 0.2 exceeds the H100
all-peer control by **114.31**, with paired episode-difference sample SD
**860.02**; 75 episode returns improve and 25 worsen. Threshold 0.6 trails all-peer by **221.32** but has no episodes below 1,000.
These initial comparisons use one training seed. Cross-family trajectories
differ before threshold activation, so A100 and H100 scores do not isolate a
threshold effect. [A100 coordination ledger](../reports/development-halfcheetah-coordination-3m-seed101.json),
[H100 controls](../reports/development-halfcheetah-coordination-h100-3m-seed101.json),
[threshold trials](../reports/development-halfcheetah-causal-threshold-3m-seed101.json).
[All six episode distributions](../reports/halfcheetah-coordination-development-3m.png)
and a [verified six-run bundle](../reports/development-halfcheetah-coordination-artifacts.json)
preserve both hardware families and every threshold result.


The independent **seed-103** H100 replication is now complete: all-peer
**4,570.10**, threshold 0.2 **4,458.25**, a difference of **−111.85**.
Together with seed 101's +114.31, the paired mean across two training seeds is
only **+1.23**. The initial gain did not replicate; **no threshold is promoted**.
[Replication ledger](../reports/development-halfcheetah-causal-replication-3m-seed103.json).

The completed locomotion baselines fail in different ways. Walker reaches the
1,000-step limit in **57/100** episodes; Ant with native contacts does so in
**99/100**. Subtracting the native +1 alive bonus per step leaves a mean residual
return of **+164.14** for Walker and **−27.01** for native-contact Ant. That
residual combines forward velocity reward and action/contact costs; it does not
identify those components separately. The existing episode records establish
early endings as a Walker limitation, while Ant needs a better net movement
reward during long episodes. The same diagnostic for computed-contact Ant gives
87/100 full-length episodes and residual −330.20; it remains a separate adaptation.

**Completed 10M HalfCheetah CPU controls**, training seeds 101/103/107:

| Training seed | IPPO | DMS, auxiliary LR 1e-3 | DMS, auxiliary LR 1e-4 |
| --- | ---: | ---: | ---: |
| 101 | 3,239.77 | 4,897.10 | 4,857.72 |
| 103 | 4,469.78 | 3,957.49 | 2,006.55 |
| 107 | 4,232.56 | 4,574.82 | 3,353.99 |
| Mean across seeds | 3,980.70 | 4,476.47 | 3,406.09 |
| Episodes below 1,000 / 300 | 20 | 35 | 55 |

DMS at LR 1e-3 leads IPPO on two seeds, averaging **+495.77** (paired
seed-difference sample SD **1,092.92**). Reducing the auxiliary LR to 1e-4
lowers mean return on **all three** seeds; the paired mean change is
**−1,070.38** (sample SD **964.62**). Source and native environment are identical
within those LR comparisons, and configurations differ only by auxiliary LR
and output. The seed-101 stability improvement did not replicate. **Do not
promote LR 1e-4**; the registered memory and coordination trials retain 1e-3.
These CPU experiments remain separate from the CUDA controls above.
[Original endpoint evidence](../reports/development-halfcheetah-10m-evaluation-queue.json),
[completed LR replications and aggregate](../reports/development-halfcheetah-aux-lr-10m-replications.json),
[verified nine-run bundle](../reports/development-halfcheetah-10m-artifacts-v2.json).

A [control-cost audit](CONTROL_COST_AUDIT.md) exactly replayed all 700 evaluation
episodes of seven 10M HalfCheetah policies. Charging raw Gaussian commands instead
of bounded controls lowers the three-seed CPU means to **3,914.75 DMS** and
**3,350.12 IPPO**. DMS still leads on two seeds; its seed-103 result falls below
the 3,800 engineering target. The bounded-policy endpoints above remain unchanged.
This reward-only sensitivity does not establish performance after training with
the old interface. Native physics and immutable-state checks passed, and the
public diagnostic reproduced eight episodes and every associated step array.

## Current diagnosis and queued work

All **27 fixed endpoints** in the previous batch have completed. An independent
[collection audit](../reports/development-completed-20261028.json) checked all
54 training/evaluation jobs, validation-file and evaluation-output hashes, the
exact 100-episode seed blocks, immutable model state and zero optimizer steps.
The per-experiment ledgers retain individual episodes and unsuccessful attempts.
The [endpoint collectors](../reports/development-endpoint-collectors.json) updated
their ledgers automatically.

The matched A100 HalfCheetah history-32 pair now has complete 10M endpoints.
Both runs exactly match their original A100 first 750 logged updates through
3M, excluding elapsed time. The separate H100 IPPO result **5,844.91** remains
recorded, but is not the matched control for the A100 DMS result. The 3M and
10M SMAC comparisons likewise differ in GPU family (A100 versus H100), so
extra training is not their only changed condition.

The initial joint RL/ELBO prior trials also completed. On A100, HalfCheetah
history-32 DMS reaches **4,409.99** versus **4,332.78** without prior RL gradients;
SMAC 3s5z gamma 0.95 reaches **59/100** wins versus **56/100**. These small
single-seed differences do not establish robust improvement, and both remain
below their matched IPPO controls. [Joint-training ledger](../reports/development-joint-intention-3m-seed101.json).

The conditioning batch completed all four **3M H100** runs, their fixed
100-episode evaluations and all five collectors. The [independent collection
audit](../reports/conditioning-diagnosis-results-20261029.json) verified all
13 jobs, endpoint hashes, seed blocks and unchanged learner/checkpoint/source.

| Task | Original DMS | Joint prior RL gradients | Inputs disabled during training | Matched IPPO |
| --- | ---: | ---: | ---: | ---: |
| Walker3x2, history 32 | 391.09 | 529.25 | 964.63 | 2,322.92 |
| SMAC 8m_vs_9m, gamma 0.95 | 0/100 wins | 0/100 wins | 38/100 wins | 66/100 wins |

Joint gradients do not close the conditioning gap. Removing inputs throughout
training improves both tasks on this development seed, but still trails IPPO.
It retains DMS's architecture, initialization and auxiliary training; it is an
ablation, not a DMS performance candidate. Comparing it with IPPO does not
isolate conditioning because their architectures and RNG use differ. Walker's
joint run averages only 214.76 steps per episode; the no-input run averages
348.62. [Full batch ledger](../reports/development-conditioning-diagnosis-3m-seed101.json).

The [preceding native validation](../reports/conditioning-native-validation.json)
passed eight short runs, exact archived/current default state comparisons,
gradient-routing checks and four saved-source evaluations. Its earlier failed
replay-tolerance attempts are preserved. Larger PPO batch shapes had small
float32 discrepancies; replay with the collection batch shape was exact.

The stability/pooling batch has also completed all **13 training, evaluation
and collector jobs**. Its four fixed 3M, 100-episode endpoints are:

| Task | Original soft DMS | Auxiliary LR 1e-4 | All-peer concatenation, LR 1e-3 |
| --- | ---: | ---: | ---: |
| Walker3x2, history 32 | 391.09 | 746.10 | 311.11 |
| SMAC 8m_vs_9m, gamma 0.95 | 0/100 wins | 0/100 wins | 1/100 wins |

Lower auxiliary LR improves Walker on this seed, but its episodes still average
only 247.42 steps and its return is below the 1,100 target. Concatenation averages
122.38 steps. Both SMAC variants remain far below the 82.6% target. No setting is
promoted. The [independent audit](../reports/stability-pooling-results-20261029.json)
checks job completion, endpoint hashes, exact evaluation seeds and unchanged
state. The [full ledger](../reports/development-stability-pooling-3m-seed101.json)
retains every episode. All four runs use training seed 101 and matched H100
hardware. Changing concatenation also changes input width and initialization.

The next design under test is [residual intention fusion](RESIDUAL_INTENTIONS.md).
It preserves the local actor/critic trunk and adds separate zero-initialized
projections of inferred intention context. This tests whether a stronger local
starting point improves conditioning. A paired model with inputs disabled during
training remains an ablation, not a DMS candidate. The [unit validation](../reports/residual-unit-validation.json)
passed all 25 new invariants and the complete modern suite (**516 passed,
32 native checks skipped**). The [native validation](../reports/residual-native-validation.json), job
**21279768**, passed all eight short training cases on A100. Both tasks exactly
matched IPPO's initial local parameters, RNG and first rollout with active peer
gates. Archived/current default training matched parameters, optimizer state,
RNG and metrics exactly. All four saved-source CPU evaluations passed with
unchanged state and zero optimizer steps. An [installed-wheel check](../reports/public-wheel-residual-validation.json)
also passed. Its first harness attempt incorrectly expected two toy agents;
the corrected agent-count check passed without source changes.

The guarded submission job **21279947** verified all 18 required evidence files
and submitted the paired fixed-3M H100 experiments:

| Task | Intentions enabled | Inputs disabled during training | Fixed 100-episode evaluation jobs |
| --- | --- | --- | --- |
| Walker3x2, history 32 | 21280354 | 21280356 | 21280355 / 21280357 |
| SMAC 8m_vs_9m, history 16, gamma 0.95 | 21280358 | 21280360 | 21280359 / 21280361 |

Walker residual job **21280354** has reached **1,924,000 steps** and its
trained no-input control **21280356** has reached **2,280,000 steps**, both on
H100. SMAC residual **21280358** has reached **1,353,600 steps** and its
no-input control **21280360** has reached **1,043,200 steps**, also on H100.
All four are running with finite latest metrics. Evaluations and collectors
remain dependent on completion. No fixed-endpoint performance result is
available for residual fusion yet.
[Frozen settings and queue](../reports/development-residual-fusion-3m-seed101.json).

The residual models also have [queued mechanism diagnostics](../reports/development-residual-mechanism-3m-seed101.json):
Walker **21281731** and SMAC **21281732**, each dependent on its fixed-3M
evaluation. Both will replay the full 100 normal episodes before paired hold
and mask interventions, then collect four fresh 128-transition trajectories
for prediction baselines, latent shuffles and peer-removal sensitivity. Native
smoke **21281659** passed on both tasks with exact saved-episode replay and
unchanged model/source/checkpoints, without optimizer updates.
[Validation](../reports/residual-mechanism-native-validation.json).
These are fixed-policy diagnostics, distinct from the trained no-input controls;
their full results remain pending and no threshold is promoted.

Four missing primary partitions now have declared local-history IPPO controls:
Ant **2x4 / 8x1** and Walker **2x3 / 6x1**. They reuse their task's existing
history-32 settings, training seed 101, a fixed 3M budget and native rewards.
Ant keeps native contact-force buffers and initial log standard deviation −1;
Walker keeps −0.5. These controls do not establish DMS performance.
[Configurations and queue](../reports/development-partition-baselines-3m-seed101.json).

The [native partition validation](../reports/partition-baselines-native-validation.json),
job **21280525**, passed all **35 environment checks**, four 8,000-step GPU
training runs, and four saved-source CPU evaluations with unchanged state and
zero optimizer steps. Guarded submission **21280592** verified the evidence
and submitted the four fixed-3M controls:

| Partition | Training job | Fixed 100-episode evaluation job |
| --- | --- | --- |
| Ant 2x4 | 21280632 | 21280633 |
| Ant 8x1 | 21280634 | 21280635 |
| Walker 2x3 | 21280636 | 21280637 |
| Walker 6x1 | 21280638 | 21280639 |

The first two fixed-3M baselines have completed training, separate 100-episode
evaluations and their collectors on H200. An [independent partial audit](../reports/partition-baselines-results-20261029-partial.json)
verified all six jobs, checkpoint/evaluation hashes, source inventories, exact
seed blocks, finite metrics, unchanged evaluation state and zero optimizer steps.

| IPPO partition | Mean return | Mean episode length | Episodes below 1,000 steps |
| --- | ---: | ---: | ---: |
| Ant 2x4 | 936.76 | 1,000.00 | 0 / 100 |
| Walker 2x3 | 412.09 | 153.11 | 100 / 100 |

Both are below the full-10M engineering targets at this shorter development
budget. These are baseline results, not DMS measurements. Ant's return after
subtracting its alive bonus is −63.24; this combines forward movement and
control/contact costs. Walker ends early in every episode. The stronger earlier
Walker3x2 history-32 control does not transfer automatically to this partition;
that comparison also used a different GPU family. Ant 8x1 **21280634** and
Walker 6x1 **21280638** remain running on H100, with evaluations already queued.
Each actual GPU family is recorded and must be matched by that partition's
future DMS comparison. Five dependent collectors preserve the results
automatically. This fills baseline coverage while residual DMS remains under
test on Walker3x2 and SMAC 8m_vs_9m. Held-out seeds remain unused.

The [Ant exploration diagnostic](ANT_EXPLORATION.md), job **21282086**,
restored the same 3M IPPO checkpoint and fresh 4,000-transition batch for four
temporary PPO updates. Entropy coefficient **0.1** increased all eight action
log-standard-deviation components; coefficients 0, 0.001 and 0.01 did not.
This is an update-level observation, not a learning-performance result. A
matched fresh **3M H100 IPPO/DMS** comparison changes only entropy **0 → 0.1**
and output relative to each original history-32 reference. Native smoke
**21282286** passed both 8,000-step training cases and saved-source evaluations.
The first full jobs **21282324 / 21282326** failed before collecting any
transitions because their launcher called a metadata helper absent from the
archived trainer; their dependent evaluations were cancelled. The source and
hyperparameters remain unchanged. Exact-launcher validation **21282799**
passed both 8,000-step cases and separate saved-source evaluations, recording
runtime metadata independently. The retry checked 13 evidence files and
queued training **21282828 / 21282830**, fixed evaluations **21282829 / 21282831**,
and three collectors. [Launcher evidence](../reports/ant-entropy-launcher-validation.json).
[Failed attempt](../reports/development-ant-entropy-3m-seed101.json),
[current retry](../reports/development-ant-entropy-3m-seed101-attempt2.json).


The [Walker2x3 clipping probe](../reports/development-walker-2x3-clipping-update-probe.json),
job **21283451**, replayed one fresh 4,000-transition batch from the completed
IPPO checkpoint. All variants restore the same model, Adam state, batch and
update RNG. The instrumented default exactly matches an ordinary update, and
an independent gradient-vector oracle verifies both clipping scopes.

| Clipping for the temporary update | Calls above the joint cap | Mean actor gradient scale | Policy KL from the updated default |
| --- | ---: | ---: | ---: |
| Joint cap 0.5 | 39 / 40 | 0.52320 | 0 |
| Separate actor/critic caps 0.5 | 39 / 40 before separating | 0.52407 | 0.00000941 |
| Joint cap 10 | 0 / 40 | 1.00000 | 0.00517321 |

Actor gradients contribute **99.74%** of the default squared norm on average.
This rejects critic domination on this batch. Separate clipping changes the
critic more than the actor; increasing the joint cap changes the actor update
materially. Neither observation establishes a better learning update. This is
one late checkpoint and one rollout, not independent training replication.
The original run and source are unchanged.

One [fresh Walker2x3 IPPO cap-10 trial](../reports/development-walker-clipping-3m-seed101.json)
is declared at **3M steps**, seed **101**, on **H200**, matching the completed
baseline. Only the cap and output change. Exact-launcher native check
**21283575** passed 8,000 transitions and two saved-source evaluation episodes.
[Validation](../reports/walker-clipping-native-validation.json) confirms finite
updates, unchanged state and zero optimizer steps during evaluation. Guarded
submission **21283693** verified six evidence files and submitted training
**21283719**, evaluation **21283720** and two collectors. The fixed evaluation uses 100
episodes, seeds 400101–400200. This is an optimizer-control trial, not a DMS
candidate or a promoted final setting. The four residual experiments and the
separate Ant exploration pair remain independent comparisons.

The [learned residual causal-selection check](../reports/residual-causal-native-validation.json)
passed **27 focused tests** and native H200 validation (**21280847**). Manual
peer-removal KL agrees with selection, and causal PPO replay is exact with the
collection batch shape. Threshold **0.6 selects no peers** after these short
all-peer training runs. Walker eligible KL ranges from 0.000245 to 0.002041;
SMAC values are at float32 rounding scale (−3.69e−7 to 4.66e−7). Zero-threshold
selection therefore is not evidence of useful influence. Original checkpoints
remain unchanged, and no threshold or performance claim is promoted.

The [paired auxiliary-drift probe](../reports/development-walker-history32-soft-update-probe.json),
job **21272741**, completed twelve ephemeral CPU updates on four archived
Walker rollouts. Each three-way comparison starts the auxiliary intervention
from exactly the same post-PPO state and update RNG. Ordinary auxiliary Adam
gives mean policy KL **0.003278**; per-step interpolation at rho=0.1 gives
**0.00007398**, as does scaling the learning rate by 0.1. The two stabilized
implementations differ by at most **2.39e-7** in a model parameter. The mean
fraction of sampled action-probability ratios outside [0.8,1.2] falls from
**3.375% to zero**. Original models, source and rollout files remain unchanged.
This supports the local stabilization mechanism, not improved learned returns.
The previously negative HalfCheetah learning-rate replications remain relevant.

A [representation-alignment probe](../reports/development-walker-history32-representation-drift.json),
job **21281883**, exactly reproduced ordinary auxiliary drift on those four
saved rollouts. An affine mapping fitted on two environments reduced mean
policy KL on the other two from **0.00329292 to 0.00017006** (94.84%). Offset
and orthogonal corrections helped much less. Decoded forecasts also changed
at a comparable normalized scale, so they are not established as a more stable
replacement. This is an offline diagnostic on one failed concat/soft checkpoint;
original files remain unchanged, and no new training method is promoted.

A [verified twelve-run local archive](../reports/development-conditioning-artifacts-v2.json)
now preserves all eight completed conditioning/stability/pooling trials and
four matched H100 controls, including all zero-win SMAC policies. It contains
312 payload files and is 104,686,958 bytes. Separate fixed 100-episode endpoints
remain in the checked-in ledgers. No archive has been published.

A completed [Walker history-32 drift probe](../reports/development-walker-history32-policy-drift.json),
job **21251696**, made one ephemeral CPU update from the failed 3M checkpoint
on 4,000 fresh stochastic transitions. PPO alone gives mean policy KL
**0.007549**; observation-moment refresh gives **0.00000105**; auxiliary-feature
refresh gives **0.004832**, with **5.12%** of sampled action-probability ratios
outside [0.8, 1.2]. The latter is concentrated partly in the third agent
(11.2% outside). Original parameters, checkpoints and source remain unchanged.
The [bounded repeat](../reports/development-walker-history32-policy-drift-replicates.json)
(**21252121**) completed on three fresh rollout seeds 400712–400714. Across
all four rollouts at this same checkpoint, auxiliary-refresh KL averages
**0.003958** and the mean fraction outside the PPO interval is
**4.18%**. All original files
remain unchanged. These are rollout repetitions, not independent training seeds.
This differs from the earlier no-memory Walker probe, where auxiliary drift was
small. It establishes a measurable mechanism at this checkpoint, not historical
causation or a return improvement. The KL components are not additive.

## Mechanism findings and development decisions

An optional [joint-training path](JOINT_INTENTION_TRAINING.md) now allows PPO
actor/value gradients into the local intention prior (`intention_rl_lr>0`).
It remains disabled by default; the two completed development results are reported above. The full
modern suite passed **491 tests**, with 32 native legacy checks skipped because
that suite uses the modern environment. A four-run native regression at
8,000 steps per run exactly preserved IPPO/DMS states, optimizer moments,
RNG, metrics and evaluation episodes when disabled. All four enabled CPU/CUDA native smokes passed, followed by saved-source
evaluation of all four checkpoints with zero optimizer steps. Two fresh 3M
development trials on A100 completed: HalfCheetah history 32 (**21218981**)
and SMAC gamma 0.95 (**21218984**), each with a fixed 100-episode endpoint.
Their baseline configurations differ only in `intention_rl_lr=3e-4` and output.
All logged inactive-warmup metrics match their original A100 controls exactly:
25 updates through 100k HalfCheetah steps and one update through 3,200 SMAC
steps. The small observed endpoint gains have not been replicated.
[Trial declaration](../reports/development-joint-intention-3m-seed101.json).
[Unit evidence](../reports/joint-intention-unit-validation.json),
[default regression](../reports/joint-intention-default-regression.json),
[native smoke](../reports/joint-intention-native-smoke.json),
[saved-source checks](../reports/joint-intention-saved-source-evaluation.json).


The frozen 3M HalfCheetah models contain predictive peer information. Forecast
RMSE is **0.2032** without policy memory and **0.6878** with history 32, below
their corresponding persistence/history-mean baselines. Peer-latent shuffling
worsens forecasts. These checks preserve parameters and normalization statistics
and do not establish a return benefit. [Mechanism evidence](../reports/development-halfcheetah-memory-mechanism-3m.json).

A [cross-forecast control](../reports/development-halfcheetah-memory-cross-forecast-3m.json)
scored both frozen predictors on both policy-generated datasets. RMSE:

| Data-generating policy | Predictor from no-policy-memory run | Predictor from history-32 run |
| --- | ---: | ---: |
| No policy memory | 0.2032 | 0.7032 |
| History 32 | 0.8266 | 0.6878 |

Each predictor performs better on its own policy's data. Neither dominates on
both datasets, so replacing one with the other does not resolve the forecasting
gap. The diagonal results exactly replayed the earlier probes. Cross-policy
samples are outside the other model's training distribution; this control does
not identify a single cause of the performance difference.

A [bounded auxiliary-fitting probe](../reports/development-halfcheetah-history32-aux-fit-probe.json)
restored the original auxiliary optimizer into twelve ephemeral history-32
clones: 2/8 updates × batch size 64/256 × three update-RNG seeds. Actor weights,
normalizers, original checkpoints and validation data were unchanged. The
validation RMSE before fitting was **0.68193**. Mean RMSE after 2×64 fitting
was **0.68259** and after 8×256 fitting **0.67915**, while mean policy KL from
changed intention features rose from **0.00080** to **0.00474**. These small,
variable local changes do not justify a full-training change; the 10M recipe
remains as registered. No benchmark return was measured by this probe.

The completed all-peer/causal [mechanism checks](../reports/development-halfcheetah-coordination-mechanism-3m.json)
use 4,000 fixed-policy stochastic transitions each. All-peer forecast RMSE is
**0.2554**, versus **1.1969** for persistence and **0.7761** for the visible
history mean. The causal model's respective values are **0.1890**, **1.2673**
and **0.8124**. Forecast error does not track the return ranking here. On these
stochastic probe trajectories the causal selector retains the visible peer for
99.9% and 94.2% of the two observers' eligible steps; these are not gate counts
from the deterministic 100-episode evaluation.

A separate [frozen-policy threshold intervention](../reports/development-halfcheetah-causal-gate-intervention-3m.json),
job **21214676**, exactly replayed ten original evaluation episodes before changing
only execution threshold 0.6 to 0.1 for 100 paired episode seeds. Mean return
rose from **3,394.71 to 3,448.95**, a paired difference of **54.24** (episode-difference
sample SD **432.49**); 54 episodes improved and 46 worsened. Weights, normalizers
and checkpoint files stayed unchanged, with zero optimizer steps. This small
descriptive gain leaves most of the all-peer gap unresolved. It is a test-time
intervention on a policy trained at 0.6, distinct from the fresh threshold-0.1
learning run and from independent training-seed evidence.

[SMAC prediction probes](../reports/development-smac-gamma-mechanism-3m.json),
jobs **21215316 / 21215317**, now compare the frozen gamma-0.99/0.95 DMS models
on fresh stochastic trajectories. Both preserve parameters, normalizers and
source, and evaluate prediction baselines and visibility-matched peer shuffles.
Their datasets follow different policies, so cross-policy forecast errors are
descriptive. The completed gamma-0.99 model predicts next actions with **77.05%**
accuracy versus **69.00%** for persistence on matched targets, despite its
**0/100** fixed-endpoint wins. Forecast quality alone does not establish useful
coordination. The completed gamma-0.95 model reaches **74.68%** prediction accuracy versus
**68.85%** for persistence on matched targets. Training and diagnostic windows
accept partial past histories; the four-step future must remain within its
episode. A source audit found no exclusion of episodes shorter than history 32.

The [paired SMAC peer-intention intervention](../reports/development-smac-gamma095-peer-interventions-3m.json),
job **21216843**, completed with all parameters, normalizers and source unchanged.
It exactly replayed the original 100 normal episodes before the interventions:

| Frozen-policy input condition | Wins / 100 | Mean return |
| --- | ---: | ---: |
| Normal inferred peer intentions | 56 | 17.8645 |
| Hold each peer's first eligible latent | 57 | 17.8283 |
| Mask peer gates | 0 | 5.5596 |

Holding latents changes individual outcomes: 14 normal wins become losses and
15 normal losses become wins. Similar aggregate win rates are not an equivalence
result. The current own query, local history, legal actions and visibility gates
remain adaptive in the held-latent condition. Masking substantially harms this
frozen policy, but also shifts its input distribution. These interventions show
dependence on the representation; they do not establish a benefit from refreshing
peer intentions here, or superiority over the trained IPPO control (81/100 wins).

Earlier SMAC reward, gradient-clipping and Walker feature-drift diagnostics,
including negative findings and exact default-update regression checks, remain
in [the archived development history](DEVELOPMENT_HISTORY_20261026.md).

## Reader workflow and remaining release requirements

The [CPU and CUDA Apptainer routes](../containers/README.md) run without host Python
packages or Triton library modules. Validation passed **56 native tests with
zero skips**, thirteen physics cases with **130 exactly matching arrays**, and
four short native DMS training/checkpoint checks. The documented wrapper,
public physics probe and saved-source evaluation also passed. The CUDA image separately passed 56 native tests, all 130 matching physics arrays,
four CUDA training/checkpoint smokes, saved-source evaluation and scheduler
visibility checks on A100 (jobs **21215121 / 21215206**). Other hosts, other
GPU models for this image and exact original MuJoCo 2.0 parity remain unverified.
[CPU evidence](../reports/legacy-container-validation.json),
[CUDA evidence](../reports/legacy-cuda-container-validation.json).

The separate [MuJoCo 2.0 audit](MUJOCO200_COMPARISON.md) now runs the original
repository's Gym 0.17.2 / mujoco-py 2.0.2.8 pins. Build **21281422** and
comparison **21281466** passed. All **129 dynamics arrays** match the 210
compatibility runtime exactly across fifteen fixed-input trajectories.
Changing Gym changes all **30 native same-seed reset arrays**; explicitly
matched initial states remove this confound. Native Ant contact buffers are
zero under both engines. This narrows the runtime gap without proving full
training or accepted-paper parity. A verified local archive preserves all
captures, code and logs. [Evidence](../reports/mujoco200-runtime-comparison.json).

A [native public-trainer comparison](../reports/native-public-trainer-bridge.json),
job **21218446**, passed four native 8,000-step runs comparing the public source
with the older winning CPU trainer. Both IPPO and active DMS exactly matched
parameters, buffers, optimizer states, RNG, common metrics and evaluation episodes.
This short regression does not establish exact 10M-trajectory equivalence.
New training/evaluation records now include
accelerator model and numerical settings. [Validation](../reports/runtime-metadata-validation.json)
passed 12 tests, four exact CPU/CUDA before/after state comparisons, and the
documented 4,000-step CUDA reader command with saved-source evaluation.

The [latest fresh installation](../reports/fresh-install-residual-validation.json),
job **21281024**, passed **518 tests with 32 native-only skips** on Python
3.10.19 and PyTorch 2.6.0+cpu. It started with an empty environment and package
cache, downloaded dependencies, verified isolated imports and passed `pip check`.
The frozen checkout remained unchanged. An installed 128-step modern MuJoCo
residual run with local history four and two checkpoint-evaluation episodes
also passed. This verifies the current reader workflow; it is not a historical
benchmark comparison or a fresh installation of the external simulators.

The [latest public-wheel check](../reports/public-wheel-residual-validation.json),
job **21280044**, verified all 22 packaged Python/XML files, optional plotting
metadata and notices. It also exercised residual-plus-joint training in a new
installation target, using existing dependencies. The older
[joint-only wheel record](../reports/public-wheel-joint-validation.json) remains
available for its earlier source snapshot.

[Verified local artifact bundles](ARTIFACTS.md) preserve the nine completed 10M
HalfCheetah runs, all four matched 3M memory controls, all six 3M coordination controls, all six
3M SMAC controls, and nine earlier pilots,
including negative results. A new [six-run memory bundle](../reports/development-partition-memory-artifacts.json)
also preserves both Ant4x2 and HalfCheetah3x2/6x1 controls, with checkpoint
hashes matched to the fixed 100-episode evaluations. They have no public download URLs. Full raw
100-episode endpoint results are also retained in the checked-in ledgers.

Release still requires frozen final settings, useful intention inference against
equal-memory controls, the complete twelve-task matrix and required ablations,
six held-out training seeds at the registered budgets, and public artifacts.
Historical simulator parity and the manuscript's k=0/k=1 ambiguity need explicit
resolution or qualification. Computed Ant contacts change both local inputs and
the contact penalty, so they remain a separate adaptation. See the
[experiment plan](EXPERIMENT_PLAN.md), [paper contract](PAPER_CONTRACT.md), and
[reader audit](READER_AUDIT.md).
