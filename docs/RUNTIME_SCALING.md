# Runtime scaling on legacy Ant

H100 throughput was **1.13× CPU for Ant 4×2** and **1.37× for Ant 8×1** in this short, same-host DMS profile. Collecting experience dominated runtime; these measurements do not establish learning performance or a universal GPU speedup.

Job **21196516** completed successfully in **6m12s** on `gpu46`, partition `gpu-h100-80g`, with an NVIDIA H100 80GB HBM3 and Intel Xeon Platinum 8468. Allocation: one GPU, eight CPUs, 32 GiB, 30 minutes. Python 3.10.19, PyTorch 2.6.0+cu126, NumPy 1.26.4, Gym 0.23.1 and mujoco-py 2.1.2.14 were used with the MuJoCo210 compatibility backend. A real CUDA matrix kernel passed before measurement.

Each condition used four environments, 1,000 rollout steps, five PPO epochs, minibatches of 1,000, hidden size 64, latent size 8 and history length 32. DMS soft coordination was active from the start, with `decoder_history=false`, `kl_beta=0.1`, `warmup_episodes=0` and two auxiliary updates per round. Two warmup rounds preceded three timed rounds. All twenty rounds had finite updates and active peer gates: 80,000 transitions overall, including 48,000 timed transitions.

| Task | Device | Collect seconds/round | Update seconds/round | Transitions/s | GPU allocated / reserved MiB | 10M hours | With 50% margin |
|---|---|---:|---:|---:|---:|---:|---:|
| Ant 4x2 | cpu | 11.233288 | 0.676996 | 335.844 | — | 8.271 | 12.407 |
| Ant 4x2 | cuda | 10.189351 | 0.374456 | 378.651 | 156.29 / 202.00 | 7.336 | 11.004 |
| Ant 8x1 | cpu | 24.460887 | 1.890799 | 151.793 | — | 18.300 | 27.450 |
| Ant 8x1 | cuda | 18.542223 | 0.715460 | 207.709 | 241.05 / 328.00 | 13.373 | 20.060 |

The table averages only timed rounds; throughput is total timed transitions divided by total timed seconds. CUDA was synchronized before collection, after collection and after updates. Ant 4×2 has 206,180 parameters (81,172 policy/value; 125,008 intention); Ant 8×1 has 404,136 (145,432; 258,704). Counts match across CPU and CUDA. Slurm batch MaxRSS was 2.524 GiB. GPU figures are PyTorch allocation peaks and exclude driver/context allocations; the JSON also records cumulative host RSS peaks and current RSS per round.

The order was 4×2 CPU, 4×2 CUDA, 8×1 CPU, 8×1 CUDA. Every condition started from seed 101, with environment seeds 101, 1101, 2101, 3101. CPU/CUDA pairs differ only in device; backend metadata and parameter counts match. Common seed labels do not imply identical device RNG streams or trajectories. All 16 package-source hashes and the frozen profiler hash were verified unchanged.

**Existing four-CPU `batch-milan` rates are not interchangeable with this GPU-host CPU measurement.** This job used one Torch thread and serial environment stepping on a different CPU model. Three consecutive early-training rounds on one host do not quantify variation across machines or training stages. The 10M estimates exclude evaluation, checkpoint I/O and startup. Multiplying by 1.5 supplies a planning allowance, not a confidence interval or a completion guarantee.

[The portable JSON summary](../reports/runtime-scaling-summary.json) contains exact round timings, update metrics, configurations, backend metadata, parameter/memory counts, source hashes and the estimate arithmetic. Retained workspace artifacts are the [raw profile](../runs/ant-scaling-profile-seed101-01/profile.json), [submission and validation ledger](../runs/ant-scaling-profile-seed101-01/submission.json), and [frozen profiler](../runs/ant-scaling-profile-seed101-01/profile_runtime.py). Those `runs/` files are retained in this workspace and are not in the HalfCheetah export. The JSON summary embeds their measurement evidence for readers of a fresh checkout.

The measurement CLI is [scripts/profile_runtime.py](../scripts/profile_runtime.py). Its original defaults remain HalfCheetah 2×3, CPU/CUDA, IPPO/DMS, four environments, two rounds of 128 steps and no excluded warmups. This job explicitly selected `--backend legacy --scenario Ant --agent-conf 4x2 8x1 --devices cpu cuda --algorithms dms --warmup-iterations 2 --iterations 3 --rollout-steps 1000 --minibatch-size 1000 --no-decoder-history --kl-beta 0.1 --seed 101 --planning-margin 1.5`; the ledger preserves the full allocated launch and source override. Native profiling requires a compute allocation and a fresh output path.
