# Reproduction status

Snapshot: **2026-10-26 18:07 UTC**. **Not release-ready.** Development results exceed some
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
| SMAC 3s5z, 3M, gamma 0.99, continuing value | 23/100 wins | 0/100 wins | DMS has higher shaped return but fewer wins |
| SMAC 3s5z, 3M, gamma 0.99, terminal deadline | 13/100 wins | 8/100 wins | DMS still below its matched baseline |
| SMAC 3s5z, 3M, gamma 0.95, continuing value | 81/100 wins | 56/100 wins | Both improve over gamma 0.99; DMS remains below the matched baseline |
| Walker2d 3x2, 10M | 880.11 | Training | Baseline below the paper target of 1,100 |
| Ant 4x2, 10M, native contacts | 964.76 | Training | Baseline below the target of 1,700 |
| Ant 4x2, 10M, computed contacts | 606.74 | Training | Separate environment adaptation; 4/100 negative-return episodes |

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
These remain single-training-seed results; no threshold is promoted. Cross-family trajectories
differ before threshold activation, so A100 and H100 scores do not isolate a
threshold effect. [A100 coordination ledger](../reports/development-halfcheetah-coordination-3m-seed101.json),
[H100 controls](../reports/development-halfcheetah-coordination-h100-3m-seed101.json),
[threshold trials](../reports/development-halfcheetah-causal-threshold-3m-seed101.json).

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

## Active and queued experiments

| Comparison | Training jobs | Registered endpoint and evidence |
| --- | --- | --- |
| HalfCheetah history-32 IPPO/DMS | 21213074 / 21213076 | Fresh 10M runs; only budget/output change from 3M. [Ledger](../reports/development-halfcheetah-history32-10m-seed101.json) |
| Joint intention training: HalfCheetah history 32 / SMAC gamma 0.95 | 21218981 / 21218984 queued | 3M A100, RL LR 3e-4; native gradient and saved-source checks passed. [Ledger](../reports/development-joint-intention-3m-seed101.json) |
| HalfCheetah history-32 A100 IPPO control | 21216270 running | Fresh 10M run matching the A100 DMS family; prior 3M A100 native run validated. [Ledger](../reports/development-halfcheetah-history32-a100-control-10m-seed101.json) |
| HalfCheetah 3x2 / 6x1, matched history-32 IPPO/DMS | 21215078 / 21215080; 21215082 / 21215084 queued | 3M partition coverage; four native CUDA smokes passed. Two chains follow the corresponding 2x3 10M runs. [Ledger](../reports/development-halfcheetah-partition-coverage-3m-seed101.json) |
| SMAC 3s5z gamma 0.95, full budget | 21214432 / 21214434 running | Fresh 10M runs after their verified 3M endpoints; only budget/output change. [Ledger](../reports/development-smac-gamma095-10m-seed101.json) |
| SMAC 8m_vs_9m IPPO/DMS | 21213720 completed, endpoint running / 21213722 training | 3M; same gamma-0.95 settings, only map/output change. [Ledger](../reports/development-smac-map-coverage-gamma095-seed101.json) |
| SMAC MMM2 IPPO/DMS | 21213724 running / 21213726 queued | 3M; follows the corresponding 8m_vs_9m run in each chain. Same ledger |
| Walker2d 3x2 / native Ant 4x2, matched history-32 IPPO/DMS | Walker 21217029 / 21217031 running; Ant 21217033 / 21217035 queued | All four native CUDA smokes passed; 3M H100 pairs follow coordination controls in two chains. [Ledger](../reports/development-walker-ant-history32-3m-seed101.json) |
| Walker2d 3x2 IPPO/DMS | 21210999 completed / 21211001 running | 10M unchanged independent-update recipes. [Ledger](../reports/development-walker-10m-seed101.json) |
| Ant 4x2, native/computed contacts, IPPO/DMS | 21212116 / 21212118 / 21212120 / 21212122 | 10M, initial log std −1.0. [Ledger](../reports/development-ant-10m-contact-seed101.json) |

Each learning run has a separate **100-episode fixed-final-checkpoint evaluation**
with a successful-parent dependency. Collectors preserve results and failures;
they cannot retry training or select the best diagnostic checkpoint. All new
SMAC-map smokes passed: four 1,280-transition runs, finite updates, and active
DMS intention learning. Their two dependency chains replace completed 3s5z
jobs, limiting that queue to two simultaneous full training jobs.

The fresh history-32 HalfCheetah 10M baseline differs from its 3M run from
the first logged update. The earlier run used A100 and the new one H100; this
does not isolate the cause of the difference. DMS's full 750-update logged prefix matches its earlier A100 run through
3M steps. These are fresh development
runs, not exact-state continuations or independent seed replicas. The
[ledger](../reports/development-halfcheetah-history32-10m-seed101.json) records
the prefix audit. An additional A100 IPPO run now restores the same GPU family
for this 10M comparison; all 750 logged updates through 3M exactly match the
prior A100 run, excluding elapsed time. The H100 baseline is retained. Pending HalfCheetah
3x2/6x1 and SMAC MMM2 pairs were restricted to H100 before training, with
before/after scheduler records retained in their ledgers. The 10M SMAC pair
also shares H100, while its 3M references used A100; both logged prefixes
already differ at the first update. Extra training is therefore not the sole
changed condition in comparisons across those two budgets.

## Mechanism findings and development decisions

An optional [joint-training path](JOINT_INTENTION_TRAINING.md) now allows PPO
actor/value gradients into the local intention prior (`intention_rl_lr>0`).
It remains disabled by default and has no performance result yet. The full
modern suite passed **491 tests**, with 32 native legacy checks skipped because
that suite uses the modern environment. A four-run native regression at
8,000 steps per run exactly preserved IPPO/DMS states, optimizer moments,
RNG, metrics and evaluation episodes when disabled. All four enabled CPU/CUDA native smokes passed, followed by saved-source
evaluation of all four checkpoints with zero optimizer steps. Two fresh 3M
development trials are queued on A100: HalfCheetah history 32 (**21218981**)
and SMAC gamma 0.95 (**21218984**), each with a fixed 100-episode endpoint.
Their baseline configurations differ only in `intention_rl_lr=3e-4` and output.
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
[CUDA evidence](../reports/legacy-cuda-container-validation.json). A [native public-trainer comparison](../reports/native-public-trainer-bridge.json),
job **21218446**, passed four native 8,000-step runs comparing the public source
with the older winning CPU trainer. Both IPPO and active DMS exactly matched
parameters, buffers, optimizer states, RNG, common metrics and evaluation episodes.
This short regression does not establish exact 10M-trajectory equivalence.
New training/evaluation records now include
accelerator model and numerical settings. [Validation](../reports/runtime-metadata-validation.json)
passed 12 tests, four exact CPU/CUDA before/after state comparisons, and the
documented 4,000-step CUDA reader command with saved-source evaluation.

[Verified local artifact bundles](ARTIFACTS.md) preserve the nine completed 10M
HalfCheetah runs, all four matched 3M memory controls, all six 3M SMAC controls, and nine earlier pilots,
including negative results. They have no public download URLs. Full raw
100-episode endpoint results are also retained in the checked-in ledgers.

Release still requires frozen final settings, useful intention inference against
equal-memory controls, the complete twelve-task matrix and required ablations,
six held-out training seeds at the registered budgets, and public artifacts.
Historical simulator parity and the manuscript's k=0/k=1 ambiguity need explicit
resolution or qualification. Computed Ant contacts change both local inputs and
the contact penalty, so they remain a separate adaptation. See the
[experiment plan](EXPERIMENT_PLAN.md), [paper contract](PAPER_CONTRACT.md), and
[reader audit](READER_AUDIT.md).
