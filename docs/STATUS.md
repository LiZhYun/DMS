# Reproduction status

Snapshot: **2026-10-26 13:50 UTC**. **Not release-ready.** Development results exceed some
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
| Walker2d 3x2, 10M | 880.11 | Training | Baseline below the paper target of 1,100 |

[The 3M evaluation ledger](../reports/development-fixed3m-evaluation-queue.json)
and [Walker ledger](../reports/development-walker-10m-seed101.json) retain every
individual episode and validation check. Ten-episode training diagnostics use
different seeds and must not be substituted for these endpoints. For example,
the terminal-deadline SMAC baseline had 5/10 final diagnostic wins but 13/100
wins in the separate block. CPU replay exactly reproduced the original ten
CUDA games for both continuing-value SMAC controls; it found no device
discrepancy in that replay. [Replay evidence](../reports/development-smac-evaluation-device-replay.json).

**All-peer coordination:** the separate 3M no-memory DMS run averages **4,453.56**
over 100 episodes, with zero below 1,000. Its concatenating actor/critic differs
from soft attention's pooled architecture. The matched causal variant uses the
same concatenating architecture and has finished training; its separate
100-episode evaluation is running. Only coordination/output differ. [Coordination ledger](../reports/development-halfcheetah-coordination-3m-seed101.json).

**Earlier 10M HalfCheetah CPU controls**, training seeds 101/103/107:

| Training seed | IPPO | DMS, auxiliary LR 1e-3 | DMS − IPPO |
| --- | ---: | ---: | ---: |
| 101 | 3,239.77 | 4,897.10 | +1,657.33 |
| 103 | 4,469.78 | 3,957.49 | −512.29 |
| 107 | 4,232.56 | 4,574.82 | +342.26 |
| Mean across seeds | 3,980.70 | 4,476.47 | +495.77 |

DMS leads on two seeds, with **35/300 episodes below 1,000**, versus **20/300**
for IPPO. The paired seed-difference sample SD is **1,092.92**. Lower auxiliary
LR 1e-4 reaches **4,857.72**, with 2/100 below 1,000, on selection seed 101;
its independent development replicas are running. These older CPU experiments
are distinct from the newer matched CUDA memory controls above.
[Full-budget episode evidence](../reports/development-halfcheetah-10m-evaluation-queue.json),
[learning curves](../reports/halfcheetah-10m-development-completed.png).

## Active and queued experiments

| Comparison | Training jobs | Registered endpoint and evidence |
| --- | --- | --- |
| HalfCheetah history-32 IPPO/DMS | 21213074 / 21213076 | Fresh 10M runs; only budget/output change from 3M. [Ledger](../reports/development-halfcheetah-history32-10m-seed101.json) |
| HalfCheetah auxiliary LR 1e-4, seeds 103/107 | 21208968 / 21208970 | 10M, same settings as seed 101. [Ledger](../reports/development-halfcheetah-aux-lr-10m-replications.json) |
| HalfCheetah all-peer/causal | 21212371 / 21212373 completed | 3M; causal evaluation and both mechanism probes registered. [Ledger](../reports/development-halfcheetah-coordination-3m-seed101.json) |
| HalfCheetah causal thresholds 0.1/0.2 | 21214579 / 21214581 queued | 3M; only threshold/output change from 0.6. Native CUDA smoke passed both variants. [Ledger](../reports/development-halfcheetah-causal-threshold-3m-seed101.json) |
| SMAC terminal deadline | 21208999 / 21209000 completed | Both 100-game endpoints complete. [Ledger](../reports/development-fixed3m-evaluation-queue.json) |
| SMAC gamma 0.95 | 21210585 / 21210586 | 3M; native rewards and continuing value unchanged. [Ledger](../reports/development-smac-gamma095-seed101.json) |
| SMAC 3s5z gamma 0.95, full budget | 21214432 / 21214434 queued | Fresh 10M runs after their verified 3M endpoints; only budget/output change. [Ledger](../reports/development-smac-gamma095-10m-seed101.json) |
| SMAC 8m_vs_9m IPPO/DMS | 21213720 / 21213722 queued | 3M; same gamma-0.95 settings, only map/output change. [Ledger](../reports/development-smac-map-coverage-gamma095-seed101.json) |
| SMAC MMM2 IPPO/DMS | 21213724 / 21213726 queued | 3M; follows the corresponding 8m_vs_9m run in each chain. Same ledger |
| Walker2d 3x2 IPPO/DMS | 21210999 completed / 21211001 running | 10M unchanged independent-update recipes. [Ledger](../reports/development-walker-10m-seed101.json) |
| Ant 4x2, native/computed contacts, IPPO/DMS | 21212116 / 21212118 / 21212120 / 21212122 | 10M, initial log std −1.0. [Ledger](../reports/development-ant-10m-contact-seed101.json) |

Each learning run has a separate **100-episode fixed-final-checkpoint evaluation**
with a successful-parent dependency. Collectors preserve results and failures;
they cannot retry training or select the best diagnostic checkpoint. All new
SMAC-map smokes passed: four 1,280-transition runs, finite updates, and active
DMS intention learning. Their two dependency chains replace completed 3s5z
jobs, limiting that queue to two simultaneous full training jobs.

## Mechanism findings and development decisions

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

The all-peer/causal [mechanism checks](../reports/development-halfcheetah-coordination-mechanism-3m.json)
use 4,000 fixed-policy transitions each. The completed all-peer model has forecast
RMSE **0.2554**, versus **1.1969** for persistence and **0.7761** for the visible
history mean. Causal results follow its endpoint evaluation. Sensitivity to
removing or shuffling a peer is distinct from a measured coordination benefit.

Earlier SMAC reward, gradient-clipping and Walker feature-drift diagnostics,
including negative findings and exact default-update regression checks, remain
in [the archived development history](DEVELOPMENT_HISTORY_20261026.md).

## Reader workflow and remaining release requirements

The [CPU Apptainer route](../containers/README.md) runs without host Python
packages or Triton library modules. Validation passed **56 native tests with
zero skips**, thirteen physics cases with **130 exactly matching arrays**, and
four short native DMS training/checkpoint checks. The documented wrapper,
public physics probe and saved-source evaluation also passed. Other hosts,
GPU container execution and exact original MuJoCo 2.0 parity remain unverified.
[Container evidence](../reports/legacy-container-validation.json).

[Verified local artifact bundles](ARTIFACTS.md) preserve the seven earlier 10M
HalfCheetah runs, all four matched 3M memory controls, and nine earlier pilots,
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
