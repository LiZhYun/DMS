# Reproduction status

Snapshot: **2026-10-25 08:41 UTC**, verified from Slurm accounting and saved run artifacts; live progress below was read at 08:39 UTC. **The reproduction is not release-ready, and performance exceeding the paper has not been established.** All four initial seed-101 pilots and seven historical seed-101 500,000-transition pilots completed. New development work comprises six matched one-million-transition runs, four HalfCheetah seed-103/107 replicates, and two Ant coordination variants. The new HalfCheetah IPPO runs have completed; their DMS counterparts remain active. Both additional SMAC maps pass bounded execution checks. Fresh SMAC runs have executed production-threshold process restarts, but their one-million-transition budgets are unfinished. Lower KL weights retained predictive peer information in frozen probes without establishing a return benefit. No 10-million-transition benchmark or held-out six-seed comparison is recorded.

## Verified compute results

Slurm accounting reports all four jobs `COMPLETED`, exit code `0:0`. The three smoke logs each report **119 tests passed** on their archived source version. Each smoke completed collection, policy/intention updates, deterministic evaluation, and checkpoint saving.

| Job | Check | Exact training transitions | Final evaluation episodes | Hardware / partition | Total job elapsed | Evidence |
| --- | --- | ---: | ---: | --- | --- | --- |
| 21192545 | Synthetic delayed-signal task | 512 | 2 | V100 16 GB, dgx18 / gpu-debug | 47 s | [run config](../runs/toy-smoke-gpu-101/config.json), [log](../logs/dms-smoke_21192545.out) |
| 21192550 | Modern MaMuJoCo HalfCheetah 2x3, k=1 | 128 | 1 | V100 16 GB, dgx18 / gpu-debug | 29 s | [run config](../runs/mamujoco-smoke-101/config.json), [log](../logs/dms-smoke_21192550.out) |
| 21192551 | Original SMAC 3s5z | 64 | 1 | H200, gpu61 / gpu-h200-141g-ellis | 53 s | [run config](../runs/smac-smoke-101/config.json), [log](../logs/dms-smoke_21192551.out) |
| 21192561 | CPU/CUDA runtime comparison | 1,024 per algorithm/device combination | Not a final evaluation | V100 16 GB, dgx18 / gpu-debug | 21 s | [profile JSON](../runs/runtime-profile-21192561.json), [log](../logs/dms-profile_21192561.out) |

All smoke runs used DMS with soft attention, training seed 101, CUDA, and no intention warmup. Their budgets are deliberately tiny. MuJoCo's 128 training transitions did not complete a training episode; its separate evaluation did complete one. Their rewards/win rates are not reported here as performance evidence. The job elapsed times include tests and startup and are not full-run throughput estimates.

The three run directories contain `config.json`, `manifest.json`, `metrics.jsonl`, `evaluation.jsonl`, `latest.pt`, and source provenance. Preserve these artifacts when code changes; a passing earlier snapshot does not validate a later optimization automatically.

Additional original-SMAC coverage completed on milan5 CPU: **8m_vs_9m job 21196095** and **MMM2 job 21196096**, both exit `0:0` in 51 s. Each trained DMS for exactly **64 transitions**, with one environment, zero warmup, two finite PPO/auxiliary updates, and one final diagnostic episode. Both recorded nonempty forecast targets, peer gate fraction 0.96875, legal executed actions, the expected official map hash, build 75689, and the configured 500,000-game-loop restart policy. These short runs did not exercise a scheduled restart or establish learned performance. The [map-smoke ledger](../reports/smac-map-smokes-seed101.json) links both raw runs/logs and records all 17 coverage checks per map.

## Development pilots and current validation

These pilots use four environments and training seed 101. Their shorter budgets support debugging and development decisions, not paper comparisons. All four initial jobs completed with exit code 0:0. Statistics below use the evaluation episodes at each run's final training budget, read directly from `evaluation.jsonl`.

| Job | Task / method | Completed transitions | Evaluation episodes | Mean return | Episode return SD | Wins | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 21192663 | Modern HalfCheetah 2x3, IPPO | 500,000 | 5 | 2,804.8997 | 41.3132 | Not applicable | [raw evaluation](../runs/dev-mamujoco-halfcheetah-2x3-ippo-101/evaluation.jsonl) |
| 21192676 | Modern HalfCheetah 2x3, DMS soft | 500,000 | 5 | 3,243.3617 | 46.3100 | Not applicable | [raw evaluation](../runs/dev-mamujoco-halfcheetah-2x3-dms-soft-101/evaluation.jsonl) |
| 21192664 | SMAC 3s5z, IPPO | 250,000 | 10 | 0.0000 | 0.0000 | 0 / 10 | [raw evaluation](../runs/dev-smac-3s5z-ippo-101/evaluation.jsonl) |
| 21192677 | SMAC 3s5z, DMS soft | 250,000 | 10 | 9.4853 | 1.4548 | 0 / 10 | [raw evaluation](../runs/dev-smac-3s5z-dms-soft-101/evaluation.jsonl) |

The reported SD is descriptive variation over evaluation episodes (`ddof=0`), **not uncertainty across independent training seeds**. DMS has a higher observed return in these short pilots, but no independent-seed improvement has been established. Both SMAC pilots won zero final evaluation games; the higher DMS shaped return does not establish success on the win-rate target. These diagnostic evaluation seeds are not the reserved final test set. Paired pilot settings are saved in [configs/development](../configs/development); CPU/GPU device choices differ and cannot establish a controlled hardware speed comparison.

The [initial pilot figure](../reports/initial-pilots.png) shows every saved evaluation checkpoint, without confidence bands. Its [series JSON](../reports/initial-pilots.json) includes episode records and means; [plot_pilots.py](../scripts/plot_pilots.py) regenerates both artifacts from the four run directories.

All seven historical compatibility pilots below completed **500,000 transitions**, with Slurm status `COMPLETED` and exit `0:0`. Each used four environments, k=1, training seed 101, and exactly five final diagnostic evaluation episodes with seeds 10001–10005. Means and population SDs are calculated from those five final records, not from intermediate checkpoints.

| Job | Historical task / method | Mean return | Episode return SD | CPU node / elapsed | Evidence |
| --- | --- | ---: | ---: | --- | --- |
| 21192876 | HalfCheetah 2x3, IPPO | 1,928.2011 | 16.9708 | milan6 / 6 min 43 s | [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-ippo-101/evaluation.jsonl) |
| 21192877 | HalfCheetah 2x3, DMS soft with decoder history | 2,372.8025 | 41.3165 | milan6 / 23 min 29 s | [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-dms-soft-101/evaluation.jsonl) |
| 21192900 | HalfCheetah 2x3, DMS soft with `decoder_history=false` | 3,409.0147 | 46.2947 | milan2 / 28 min 50 s | [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-101/evaluation.jsonl) |
| 21195750 | HalfCheetah 2x3, latent-only DMS soft, KL weight 0.1 | 3,109.8340 | 50.5669 | milan1 / 24 min 35 s | [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/evaluation.jsonl) |
| 21195751 | HalfCheetah 2x3, latent-only DMS soft, KL weight 0.01 | 875.4621 | 232.7468 | milan1 / 21 min 27 s | [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-001-101/evaluation.jsonl) |
| 21192912 | Ant 4x2, IPPO | 1,015.3063 | 2.0947 | milan6 / 13 min 49 s | [raw evaluation](../runs/dev-legacy-ant-4x2-ippo-101/evaluation.jsonl) |
| 21192913 | Walker2d 3x2, IPPO | 320.7701 | 2.8100 | milan7 / 8 min 51 s | [raw evaluation](../runs/dev-legacy-walker2d-3x2-ippo-101/evaluation.jsonl) |

These runs used `batch-milan`. Historical compatibility returns remain separate from modern v5 returns. The latent-only decoder at KL weight 1 has the highest observed HalfCheetah return in this single-seed comparison, but its mechanism probe does not establish useful peer intentions. KL weights 0.1 and 0.01 retain predictive peer information in their probes while achieving lower observed returns. Five evaluation episodes from one trained seed are development evidence, not a release result or an estimate of uncertainty across training seeds.

Both longer SMAC jobs **failed with exit `1:0`** before their planned one-million-transition budgets. The counts below are the last completed updates recorded in `metrics.jsonl`, not completed training budgets.

| Job | Task / method | Last completed transitions | Compute / elapsed | Evidence |
| --- | --- | ---: | --- | --- |
| 21192903 | SMAC 3s5z, IPPO | 261,120 | CPU, milan6 / 35 min 20 s | [metrics](../runs/dev-smac-3s5z-ippo-1m-101/metrics.jsonl), [error log](../logs/dms-cpu_21192903.err) |
| 21192904 | SMAC 3s5z, DMS soft | 260,480 | A100, gpu43 / 55 min 12 s | [metrics](../runs/dev-smac-3s5z-dms-soft-1m-101/metrics.jsonl), [error log](../logs/dms-train_21192904.err) |

Both errors report that SMAC ended without battle results after an SC2 protocol failure. These infrastructure failures must not be counted as lost games. Investigation identified an SC2 process-lifetime boundary near `2^19` game loops. The repair schedules process restarts at episode boundaries before reaching that limit and passed 25 focused SMAC tests. Native check **21195862** also completed with exit `0:0`: its [report](../runs/smac-lifetime-21195862/report.json) records 329 transitions, six completed episodes, two scheduled restarts, and zero backend forced restarts. All 329 adapter/native transition comparisons and six reset-history checks passed, including an episode after the second restart. This deliberately shortened check used a 1,000-game-loop restart threshold; a successful long training rerun has not yet validated the production threshold.

Fresh SMAC jobs **21195882** (IPPO, milan1 CPU) and **21195883** (DMS soft, gpu51 H200) are running with four environments, `parallel_env_steps=true`, training seed 101, and a new one-million-transition budget each. Their [IPPO manifest](../runs/dev-smac-3s5z-ippo-1m-threaded-101/manifest.json) and [DMS manifest](../runs/dev-smac-3s5z-dms-soft-1m-threaded-101/manifest.json) retain the new source/configuration. At 08:39 UTC they had completed **340,480** and **259,840** transitions respectively; both metric streams recorded **four scheduled process restarts** under the production 500,000-game-loop policy. This is evidence that the scheduled restart path executed in real training, not completion of the sustained one-million-transition check. These are fresh runs, not resumes of the failed pilots.

The pilot code caches unchanged successor policy inputs and records forecasts before auxiliary updates, including persistence and valid-target counts. History-only baselines now also use the visible past action mean for continuous controls and a visible-action histogram for discrete actions. Full-suite job **21196035** completed with exit `0:0`; its [test log](../logs/dms-release-source-tests_21196035.out) records **230 passed, 22 legacy-only skipped in 10.91 s**. This supersedes the earlier [191-test snapshot](../logs/dms-current-tests_21195870.out), which remains retained. Native legacy job **21195881** separately completed with exit `0:0` and [35 tests passed in 4.66 s](../logs/dms-current-legacy-tests_21195881.out). Focused checks also include 15 forecast/baseline tests and the 25 SMAC tests above. These results validate their tested source snapshots, not subsequent launcher or documentation edits. The earlier provenance update passed 39 focused report tests using fabricated evaluation fixtures.

Current manifests include source-file SHA-256 mappings and runtime job/device information. [report.py](../src/dms/report.py) rejects differing declared training snapshots across seeds. Missing or mismatched training/evaluation provenance leaves descriptive summaries available with warnings but blocks the release reference gate. Runtime job/device differences do not trigger a source mismatch. Matching hashes identify recorded source files; they do not independently certify execution or benchmark equivalence.

## Current matched development matrix

The [one-million-transition ledger](../reports/development-matrix-1m-seed101.json) records six fresh seed-101 historical runs. They share four environments, k=1, five evaluation episodes every 200,000 transitions, and a six-hour CPU allocation with four cores and 16 GB RAM. Within each task the JSON configs differ only in algorithm and output; both include `decoder_history=false` and `kl_beta=0.1`, which are unused by IPPO. All six startup source snapshots matched. The DMS setting is provisional, selected for promising predictive probes and its 500,000-step development return; it is not a frozen final choice.

| Task / method | Job | State at snapshot | Completed transitions |
| --- | --- | --- | ---: |
| HalfCheetah 2x3 IPPO | 21196012 | Completed, exit 0:0 | 1,000,000 |
| HalfCheetah 2x3 DMS soft | 21196013 | Running | 312,000 |
| Ant 4x2 IPPO | 21196014 | Running | 492,000 |
| Ant 4x2 DMS soft | 21196015 | Running | 108,000 |
| Walker2d 3x2 IPPO | 21196016 | Running | 684,000 |
| Walker2d 3x2 DMS soft | 21196017 | Running | 144,000 |

The completed HalfCheetah IPPO job took 10 min 34 s on milan1. Its five final diagnostic episodes give mean return **3,295.6647**, population episode SD **38.4682**; see [raw evaluation](../runs/dev-legacy-halfcheetah-2x3-ippo-1m-101/evaluation.jsonl). No matched DMS endpoint exists yet, and this SD is not training-seed uncertainty.

The [HalfCheetah replicate ledger](../reports/development-halfcheetah-seeds103-107.json) records four fresh **500,000-transition** jobs using development seeds 103 and 107, with the same episode diagnostics as the earlier 500,000-step pilots. Each requests four CPU cores, 16 GB RAM, and two hours. They use a shared archived source directory through `DMS_SOURCE_ROOT`; intention-only settings differ between methods but are unused by IPPO.

| Seed / method | Job | State / completed transitions | Final five-episode return mean / population SD |
| --- | --- | --- | --- |
| 103 IPPO | 21196090 | Completed 500,000, exit 0:0 | 3,114.3355 / 28.9997 |
| 103 DMS soft, KL 0.1 | 21196091 | Running, 200,000 | Pending |
| 107 IPPO | 21196092 | Completed 500,000, exit 0:0 | 2,708.2924 / 24.5729 |
| 107 DMS soft, KL 0.1 | 21196093 | Running, 196,000 | Pending |

The completed IPPO replicas took 6 min 11 s on milan4 and 6 min 10 s on milan5 respectively. Their [seed-103 evaluation](../runs/dev-legacy-halfcheetah-2x3-ippo-103/evaluation.jsonl) and [seed-107 evaluation](../runs/dev-legacy-halfcheetah-2x3-ippo-107/evaluation.jsonl) are preserved. The paired DMS runs are unfinished, so no across-seed method comparison is reported.

Two additional Ant 4x2 coordination variants were submitted: **all peers job 21196123** and **causal selection job 21196124**, each one million transitions, seed 101, four CPU cores, 16 GB RAM, and six hours. Their [ledger](../reports/development-ant-coordination-seed101.json) records configs differing from soft job 21196015 only in coordination mode and output, and an exact copy of that job's 16-file source snapshot. Both startups are verified against all 16 source files. These variants change policy input structure and parameter counts, so identical numerical seeds do not imply identical initialization or sampling streams; their comparisons are not isolated equal-architecture attention interventions.

## Intention mechanism diagnostic

Job **21192826** completed a fixed-checkpoint diagnostic at the modern HalfCheetah DMS pilot's 500,000-step checkpoint. The [diagnostic JSON](../runs/dev-mamujoco-halfcheetah-2x3-dms-soft-101/mechanism-diagnostic-01.json) records 512 additional environment transitions with zero parameter updates, separate diagnostic seeds, and separate checkpoint/diagnostic source mappings. It is a mechanism probe, not a final benchmark evaluation or a matching-source release artifact.

Peer latent temporal RMS coordinate standard deviations were **0.0121** and **0.0297**. Shuffling peer prior means while preserving the decoder's history input produced prediction changes of only **0.00057** and **0.00367** RMSE, and did not increase peer forecast MSE in either observed direction. Overall forecasts beat a persistence baseline in this sample, but the shuffle result does not attribute that accuracy to the latent representation. Small policy-distribution changes also establish sensitivity, not usefulness.

The controlled `decoder_history=false` ablation removes the direct encoded-history-to-decoder path while retaining the latent input, last visible action anchor, and autoregressive inputs. Its 500,000-transition historical pilot has now completed. Frozen-checkpoint diagnostics **21195732** (original decoder) and **21195733** (latent-only decoder) both completed with exit `0:0`, each in 30 s on milan1. Their [original-decoder JSON](../runs/dev-legacy-halfcheetah-2x3-dms-soft-101/mechanism-diagnostic-01.json) and [latent-only JSON](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-101/mechanism-diagnostic-01.json) preserve checkpoint hashes and separate training/diagnostic source mappings. The checkpoints were copied to distinct immutable `diagnostic-checkpoint-01.pt` files before collection.

Each diagnostic collected 512 fresh transitions (four environments × 128 steps) with seed 200001 and sampled 256 unique suffix windows. Parameters and normalizers remained frozen. Windows can overlap and are not independent evaluation episodes. Forecast comparisons use 4,064 matched time/peer targets (8,128 scalar components), covering 99.41% of the 4,088 valid targets. The history-mean baseline averages only locally visible past values per peer/component; targets without an observed past value and padded components are excluded.

| Diagnostic | Original decoder | Latent-only decoder |
| --- | ---: | ---: |
| Prior forecast RMSE | 0.50600 | 0.57344 |
| Last-visible-action persistence RMSE | 0.90047 | 0.95908 |
| Visible-history mean RMSE | 0.63867 | 0.67857 |
| Peer temporal RMS coordinate SD, observers 0 / 1 | 0.00713 / 0.02868 | 0.00258 / 0.00369 |
| Policy KL after peer-latent shuffle, observers 0 / 1 | 1.55e-5 / 2.22e-4 | 4.28e-8 / 2.90e-7 |
| Decoder prediction change RMSE after shuffle, observers 0 / 1 | 0.000369 / 0.002485 | 0.0000466 / 0.000267 |
| Forecast MSE increase after shuffle, observers 0 / 1 | +2.65e-6 / −5.20e-5 | −8.05e-7 / +3.41e-5 |

Shuffles pair different histories within the same observer/peer and match visibility masks. Actor comparisons preserve the observer's own query, observations, gates, and action masks; decoder comparisons preserve past inputs and the action anchor. Temporal variation is measured within each coordinate, excluding constant offsets between coordinates. Both models above predict better than persistence and the local history mean on their own collected trajectories, but peer-latent shuffling produces negligible policy/prediction changes and mixed-sign changes in forecast error. Removing decoder history at KL weight 1 therefore **did not restore useful latent dependence in this probe**, despite its higher pilot return.

The completed KL-weight pilots use the same latent-only decoder and 500,000-transition budget. Their frozen-checkpoint probes use the same 512-transition, seed-200001, 256-window protocol and matched forecast coverage as above. The [weight-0.1 result](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/mechanism-diagnostic-03.json), job **21196000**, and [weight-0.01 result](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-001-101/mechanism-diagnostic-01.json), job **21195926**, both completed with exit `0:0`. Parameters and normalization buffers remained unchanged.

| Diagnostic | KL weight 0.1 | KL weight 0.01 |
| --- | ---: | ---: |
| Prior forecast RMSE | 0.52761 | 0.57016 |
| Last-visible-action persistence RMSE | 0.97491 | 0.97564 |
| Visible-history mean RMSE | 0.68565 | 0.66957 |
| Peer temporal RMS coordinate SD, observers 0 / 1 | 0.14535 / 0.31335 | 0.42795 / 0.77321 |
| Policy KL after peer-latent shuffle, observers 0 / 1 | 0.002778 / 0.007995 | 0.039517 / 0.282473 |
| Decoder prediction change RMSE after shuffle, observers 0 / 1 | 0.12257 / 0.36313 | 0.52418 / 0.90575 |
| Forecast MSE increase after shuffle, observers 0 / 1 | +0.02989 / +0.10989 | +0.08633 / +0.55466 |

Each observer/peer decoder comparison uses 247 matched histories and 988 observed scalar targets. Actor shuffles compare 508 and 506 matched policy inputs for observers 0 and 1. Both lower weights show temporal variation and positive peer forecast-loss increases after shuffling, supporting useful predictive information in these sampled histories. Policy KL establishes use of that information, not a return benefit. Weight 0.1 has an observed return between the weight-1 latent-only and original-history pilots; weight 0.01 has stronger measured latent dependence but much lower return. These single-seed results do not establish that either setting improves coordination or benchmark performance. Cross-model forecast RMSE is not a matched-distribution accuracy ranking because each policy collected its own trajectories.

For weight 0.1, the [first probe](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/mechanism-diagnostic-01.json), job **21195975**, honestly retains `source_unchanged_during_diagnostic=false`: concurrent explanatory edits changed `intention.py` and `train.py`. The [second probe](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/mechanism-diagnostic-02.json), job **21195987**, had stable source and identical metrics. The third probe used a separate read-only [current diagnostic source snapshot](../runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101/diagnostic-source-03/source-manifest.json), selected through `DMS_DIAGNOSTIC_SOURCE_ROOT`; its recorded imported package path and hashes verify that selection. All three used the same immutable checkpoint and produced exactly identical forecast, temporal-variation, policy-shuffle, and decoder-shuffle metrics. Retained submission records explain the repeats; the repeated trajectories are not additional independent evidence.

## Runtime and recovered scheduler access

The [MuJoCo manifest](../runs/mamujoco-smoke-101/manifest.json) and [SMAC manifest](../runs/smac-smoke-101/manifest.json) record PyTorch **2.6.0+cu126**, NumPy **2.2.6**, Gymnasium **1.4.0**, Gymnasium Robotics **1.4.2**, MuJoCo **3.17.0**, PySC2 **3.0.0**, and SMAC **1.0.0**. The [StarCraft runtime log](../logs/dms-smoke_21192551.err) confirms **B75689 / SC2 4.10**, loaded from `/scratch/work/liz23/3rdparty/StarCraftII`.

The actual PyTorch wheel lists `sm_70` and successfully ran CUDA on V100 as well as H200. A blanket claim that this environment cannot run V100 is therefore incorrect. B300 remains excluded from the prepared templates because the recorded compiled architecture list ends at `sm_90`; Blackwell support has not been validated. Normal training templates still prefer the Ellis H200 queue with A100/H100 alternatives.

Default Slurm SRV/DNS discovery initially failed. Scheduler access recovered with this command-local setting:

```bash
export SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817
```

It changes no system configuration. Job submissions and accounting queries subsequently succeeded. Earlier documentation saying scheduler access is still blocked is superseded by this verified status.

## Exploratory runtime profiles

The profile used HalfCheetah 2x3 at k=1, four environments, two 128-step collection iterations per condition, hidden width 64, history length 32, and one PyTorch CPU thread. The following ranges are the two observed collection-plus-update throughputs, excluding final evaluation:

| Snapshot / job | Algorithm | CPU transitions/s | V100 CUDA transitions/s |
| --- | --- | ---: | ---: |
| Initial / 21192561 | IPPO | 915.6–939.1 | 366.5–555.3 |
| Initial / 21192561 | DMS | 145.6–145.9 | 184.8–188.7 |
| Cached successor inputs / 21192610 | IPPO | 579.2–1,000.1 | 119.2–565.8 |
| Cached successor inputs / 21192610 | DMS | 204.0–210.3 | 192.3–236.0 |

The [second profile](../runs/runtime-profile-21192610.json) completed successfully on V100 node dgx2 in 39 s; see its [log](../logs/dms-profile_21192610.out). The first used dgx18, and both contain only two iterations per condition with visible startup variance. These exploratory timings do not establish a universal CPU/GPU speed ranking, an isolated causal speedup, scaling to other agent counts, H200 throughput, or learning performance. Include final evaluation overhead when estimating complete experiment costs.

The separate [legacy CPU profile](../runs/legacy-profile-21192863.json), job **21192863**, completed on milan1 in 12 s. Its two observed IPPO throughputs were **1,254.3 and 1,281.7 transitions/s**; DMS recorded **262.3 and 269.2 transitions/s**. These are short collection-plus-update measurements, not matched learning results or full-budget runtime guarantees; see the [log](../logs/dms-legacy-profile_21192863.out).

The [SMAC serial/thread profile](../runs/smac-step-profile-21193014/profile.json), job **21193014**, completed with exit `0:0` on milan2 in 1 min 35 s. It used four 3s5z environments, 512 steps per environment per mode, and 32 warmup steps. Serial stepping measured **154.38 transitions/s** and four-thread stepping **273.50 transitions/s** over 1,920 timed transitions per mode, a **1.77× step-only ratio**. The instrumented-loop rates were 148.72 and 255.38 transitions/s. Paired scripted-action traces matched exactly, with zero mismatches across 553 recorded events per mode. Resets remained serial and each environment had at most one outstanding step. This single serial-then-threaded pair excludes optimizer updates and does not establish end-to-end training speed or long-run stability; measurement order and shared-node load remain confounders.

## Historical MuJoCo bootstrap

The isolated legacy environment now passes a **single-agent v2 smoke check** on a compute node. Job **21192670** completed with exit 0:0 in 1 min 13 s on milan3; its [output](../logs/dms-legacy-smoke_21192670.out) records deterministic seeded resets and timeout handling for Ant-v2, HalfCheetah-v2, and Walker2d-v2. The recorded environment uses Python 3.10.19, Gym 0.23.1, mujoco-py 2.1.2.14, MuJoCo engine 210, NumPy 1.26.4, Cython 0.29.37, and PyTorch 2.6.0+cu126, with a working NumPy/PyTorch bridge.

Earlier bootstrap attempts failed before this success: [job 21192605](../logs/dms-legacy-smoke_21192605.err) could not initialize `module`; [job 21192624](../logs/dms-legacy-smoke_21192624.err) could not find `gcc`. These are recorded failures, not successful runs.

The original-graph multi-agent compatibility adapter is now implemented, and job **21192737** passed its **35 tests** in 5.07 s on milan1, with Slurm exit 0:0. The [test log](../logs/dms-legacy-backend_21192737.out) and [JUnit report](../.cache/legacy/backend-tests.xml) cover all nine task partitions at k=0 and k=1, seeded controls/sensors, timeout handling, and a trainer update. The implementation session also reports a passing source preflight. An earlier adapter job, 21192721, exposed an invalid Ant termination fixture; [HISTORICAL_ENVIRONMENT.md](HISTORICAL_ENVIRONMENT.md) records the fixture correction and retained failure evidence.

Legacy manifests now record graph provenance, actuator routing, observation categories, simulator version, XML hashes, and reward settings. They explicitly retain `historical_binary_parity_verified=false`. This establishes a runnable historical compatibility track; the paper's unrecorded binary stack and empirical performance still need separate qualification. Modern v5 results remain a separate adaptation track.

## Prepared experiment matrix

[configs/generated/manifest.json](../configs/generated/manifest.json) lists **60 existing configs: 12 primary tasks × 5 variants** (`ippo`, `dms-soft`, `dms-causal`, `dms-all`, `dms-no-history-filter`). These are configuration files, not completed experiments. Each declares 10,000,000 environment transitions; the placeholder seed is 101. The manifest records development seeds **101, 103, 107** and final held-out training seeds **11, 23, 37, 53, 71, 89**.

The current MuJoCo track uses **k=1**, following the manuscript prose and allowing actual peer observations under the declared local-control sensor. Strict k=0 exposes only own controls and cannot exercise peer intention inference. The manuscript table's k=0 remains a separate sensitivity track; the historical sensing/observation interpretation needs clarification. The current modern adapter uses no global observation categories and gives IPPO the same locally sensed previous controls as DMS.

## Remaining requirements and immediate handoff

1. Complete the sustained SMAC process-lifetime runs and retain both failed runs and their saved source snapshots. Finish the active multi-peer pilots and seed-103/107 replicas before selecting a final protocol. Recheck numerical behavior and local visibility after further changes.
2. Establish stable learning and useful intention prediction beyond one two-agent pilot. The decoder-history ablation at KL weight 1 lacked useful peer-latent dependence; lower weights show predictive dependence in the sampled histories but have not established a return benefit. Inspect temporal latent variation, peer-selection rates, intervention effects, and forecasts against both persistence and visible-history mean/histogram baselines with matched coverage. Complete coordination, history-filter, and missing-action sensitivity experiments without revealing hidden commands. Pilot rewards and single-seed results do not satisfy these requirements by themselves.
3. Freeze code, settings, environment versions, checkpoint selection, and the final evaluation protocol before using the six held-out training seeds. Complete the matched 10-million-transition runs and separate final evaluations: 100 MuJoCo episodes or 1,000 SMAC games per trained seed.
4. Close benchmark comparability gaps. Modern MaMuJoCo v5 is a separate adaptation track from the original v2 experiments; exceeding a reference return on v5 does not establish original-environment improvement. Audit original SMAC maps/settings and retain the paper's unresolved win-rate aggregation uncertainty. Resolve or transparently report the k=0/k=1 conflict and the four-environment SMAC redesign.
5. Evaluate using each run's saved source snapshot, aggregate independent training seeds with [report.py](../src/dms/report.py), retain raw evaluations/provenance, and compare against [paper_targets.json](../configs/paper_targets.json). Older archived evaluators that omit actual evaluation environment/dependency metadata remain descriptive evidence and cannot pass the current release reference gate. A passed point-reference gate is not a full reproduction certificate. Complete the required ablations and close the [fresh-reader audit findings](READER_AUDIT.md) before release.

The durable method record is [PAPER_CONTRACT.md](PAPER_CONTRACT.md), current redesign choices are in [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md), and the full experimental protocol is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). No empirical acceptance gate is closed by the smoke or timing results above.
