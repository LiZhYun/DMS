# HalfCheetah control-cost sensitivity

The new adapters project Gaussian policy commands to the native action bounds
before simulation and reward calculation. The old wrapper passed the raw
commands to Gym, whose control penalty can charge values outside those bounds.
This is a declared policy-interface change shared by IPPO and DMS; it affects
comparisons with the original implementation. See the
[environment audit](HISTORICAL_ENVIRONMENT.md).

Job **21219756** replayed all 100 development evaluation episodes of seven
completed 10M HalfCheetah 2x3 policies. Every original episode return and length
matched exactly, parameters and normalization buffers stayed unchanged, and
there were zero optimizer steps. Each step retained its native forward reward
and recorded these two costs:

```text
bounded cost = 0.1 * sum(clip(raw_command, -1, 1)^2)
raw cost     = 0.1 * sum(raw_command^2)
```

The second score is a reward-only sensitivity on the frozen bounded policy's
trajectories. It is not the performance of a policy trained with raw-command
costs or a recreation of the old observation/action pipeline. Projecting to legal
actions remains a valid policy design; the bounded score is retained under its
declared convention.

| Policy / training seed | Bounded-control return | Raw-command cost return |
| --- | ---: | ---: |
| CPU IPPO / 101 | 3,239.77 | 2,396.50 |
| CPU IPPO / 103 | 4,469.78 | 3,932.38 |
| CPU IPPO / 107 | 4,232.56 | 3,721.48 |
| CPU DMS / 101 | 4,897.10 | 4,427.96 |
| CPU DMS / 103 | 3,957.49 | 3,351.06 |
| CPU DMS / 107 | 4,574.82 | 3,965.23 |
| H100 history-32 IPPO / 101 | 5,844.91 | 4,918.54 |

Rows average 100 deterministic CPU episodes, seeds **400101–400200**, from each
fixed final checkpoint. The H100 memory baseline is a separate hardware and
architecture track. The six CPU policies form the matched three-seed comparison.

Across those three training seeds, DMS averages **4,476.47** under bounded costs
and **3,914.75** under raw costs; IPPO averages **3,980.70** and **3,350.12**.
The paired DMS-minus-IPPO mean is **+495.77** or **+564.63**, respectively, with
paired training-seed sample standard deviations **1,092.92** and **1,335.62**.
DMS leads on two of three seeds under either convention. Its raw-cost mean
exceeds the 3,800 engineering target, but seed 103 does not. These development
results do not close the final reproduction gate or establish an effect that
generalizes across training seeds.

The saved arrays independently reconstruct all **700,000** step rewards under
both conventions. A separate native physics probe applied 32 prescribed raw
and projected commands to paired Gym environments: observations, positions,
velocities and accelerations matched exactly, while native reward differences
matched the control-cost formulas. This probe uses MuJoCo **2.1**, not the
original 2.0 binary. All episode outcomes, checks, source hashes and trace hashes
are in the [audit ledger](../reports/halfcheetah-control-cost-audit.json).

## Repeat the diagnostic

Use a trusted completed run, its archived source, and the package versions in
its manifest. Within a Triton CPU allocation, from the repository root:

```bash
export DMS_SOURCE_ROOT="$PWD/runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-10m-101/source"
bash scripts/legacy/bootstrap.sh run scripts/diagnose_control_cost.py \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-10m-101 \
  --episodes 100 --seed 400101 \
  --output runs/reader-control-cost/result.json \
  --trace runs/reader-control-cost/controls.npz
```

The named run is in the [10M development artifact bundle](ARTIFACTS.md).
The optional trace stores controls in native actuator order and per-step rewards.
Choose new output paths; existing results are never replaced. This command
currently supports the historical HalfCheetah-v2 adapter and CPU evaluation.
It checks source, relevant package versions, environment metadata, completed
budget, model state and immutable inputs before writing a result.

Job **21220143** validated this public command on four archived checkpoints,
including both older CPU trainers and the newer history-32 trainer. Every
recorded field and all step arrays for two episodes per checkpoint exactly
matched the corresponding prefix of the full audit.
[Reader-command validation](../reports/control-cost-reader-validation.json).

A separate [partition check](../reports/control-cost-partition-validation.json),
job **21220365**, passed two episodes each for native 3x2 and 6x1 DMS smoke
policies. It checks the different action layouts and shared-reward reductions;
these 8,000-step policies provide no learning-performance evidence.
