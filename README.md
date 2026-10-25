# Deep Motor System: coordination through inferred intentions

This repository redesigns the implementation of **Coordination as Inference in Multi-Agent Reinforcement Learning**. Every agent predicts locally observed teammates' behavior, infers their intentions, and uses relevant intentions in its own actor and critic. IPPO uses the same environment interface and locally available sensors, providing a direct baseline.

The paper appeared in [Neural Networks, volume 172 (2024), article 106101](https://doi.org/10.1016/j.neunet.2024.106101). Use [CITATION.bib](CITATION.bib) to cite the method. The [original implementation](https://github.com/LiZhYun/DMS) is retained as an audit reference in this workspace.

The implementation is a work in progress. **Performance exceeding the paper has not been demonstrated.** Passing tests or completing the synthetic example does not establish benchmark reproduction. The exact targets and outstanding comparability issues are recorded in [the paper contract](docs/PAPER_CONTRACT.md) and [paper_targets.json](configs/paper_targets.json).

## Read the code in this order

[One training iteration](docs/METHOD_WALKTHROUGH.md) explains the data flow, what each agent observes, and which computations happen only during training.

1. [envs.py](src/dms/envs.py) defines the observation/action interface, a tiny synthetic task, and modern Multi-Agent MuJoCo. [legacy_env.py](src/dms/legacy_env.py) adapts the historical MuJoCo tasks, and [smac_env.py](src/dms/smac_env.py) implements original SMAC.
2. [intention.py](src/dms/intention.py) contains the shared prior/posterior encoder, decoder, variational objective, and history filtering.
3. [policy.py](src/dms/policy.py) contains independent actors and critics plus soft attention and causal selection.
4. [returns.py](src/dms/returns.py) handles advantage estimation across true terminations and time limits.
5. [train.py](src/dms/train.py) contains the configuration and the collect → PPO update → intention update → evaluate loop.

The code preserves observer, peer, time, and environment dimensions explicitly. Policies receive local observations and locally inferred intentions. Global-state reconstruction is disabled. [Implementation decisions](docs/IMPLEMENTATION_DECISIONS.md) explain the deliberate differences from the manuscript.

The optional [local policy memory](docs/LOCAL_POLICY_MEMORY.md) adds a bounded observation-history GRU to both IPPO and DMS. It keeps the same local information boundary and recomputes features during PPO; development experiments are testing whether it improves learning.

## Install

Use Python 3.10 or newer in an isolated environment. Install a PyTorch build compatible with the compute node first; then install this package and the required environment extras. The existing Triton environment is `dms/.venv` and currently uses PyTorch 2.6 with CUDA 12.6.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -c requirements/modern-constraints.txt -e '.[test,mujoco]'
.venv/bin/python -m pytest tests -q
```

The commands above select a CPU build for a workstation. For the measured Triton CUDA stack, select the [official PyTorch 2.6.0 CUDA 12.6 wheel](https://pytorch.org/get-started/previous-versions/#v260) instead. [modern-constraints.txt](requirements/modern-constraints.txt) pins the inspected direct scientific dependencies; it is not a complete transitive environment lock. Run manifests remain the record of actual versions and assets. The wheel contains the Python package; clone the repository for configurations, scripts, and tests.

SMAC requires the official environment package, its maps, and a separately installed StarCraft II binary. For a fresh checkout, install the pinned environment revision:

```bash
mkdir -p reference
git clone https://github.com/oxwhirl/smac.git reference/smac
git -C reference/smac checkout --detach d6aab33f76abc3849c50463a8592a84f59a5ef84
.venv/bin/python -m pip install -c requirements/modern-constraints.txt -e '.[smac]'
.venv/bin/python -m pip install -c requirements/modern-constraints.txt -e reference/smac
```

Skip cloning when that revision is already present. Install StarCraft II **4.10 / Base75689** using the [official Linux distribution instructions](https://github.com/Blizzard/s2client-proto#downloads), then set `SC2PATH` to your installation and copy the pinned maps into it:

```bash
export SC2PATH=/absolute/path/to/StarCraftII
mkdir -p "$SC2PATH/Maps"
cp -r reference/smac/smac/env/starcraft2/maps/SMAC_Maps "$SC2PATH/Maps/"
```

Use a dedicated installation or verify existing map hashes before copying over maps used by other experiments. This workspace already has the binary and maps at `/scratch/work/liz23/3rdparty/StarCraftII`. The adapter requests version 4.10; unit tests mock SMAC and do not launch the game. The [official SMAC setup guide](https://github.com/oxwhirl/smac#quick-start) describes the environment requirements.

The [SMAC environment audit](docs/SMAC_ENVIRONMENT.md) records the verified map hashes, game selection, observation/reward defaults, and remaining historical-version gaps. Each new run saves this environment metadata.

For the historical MuJoCo compatibility track, use a separate Python 3.10 environment with Gym 0.23.1, `mujoco-py` 2.1.2.14, and MuJoCo 2.1.0. The prepared Triton setup is `.venv-legacy`; it leaves the modern environment intact. Follow the [historical environment setup](docs/HISTORICAL_ENVIRONMENT.md#isolated-preparation-and-allocated-smoke) to prepare dependencies and compile the native extension on a compute node. The original graph, v2 rewards, and XML assets are preserved, with corrected actuator routing and explicit local control sensing. This is a documented compatibility track, not verified parity with the paper's unrecorded binaries.

The pinned native Ant simulator leaves contact-force buffers zero. The optional [computed-force variant](docs/ANT_CONTACT_FORCES.md) populates those existing local observations and the contact penalty. Its configuration, benchmark label, and results remain separate from native Ant; passing its execution checks does not establish historical comparability.

## Run the synthetic check

On a workstation or allocated compute node:

```bash
.venv/bin/python -m dms.train --config configs/toy_smoke.json \
  --device cpu --output runs/toy-smoke-first
```

This is a 512-transition integration check on a small delayed-signal task, not a paper benchmark. Choose a new output directory for every run. The runner refuses to append training to an existing metrics file; it does not currently provide exact environment-state resume.

## Use Triton

Submit from this directory and create the log directory before calling `sbatch`, since Slurm opens its logs before executing the script. Training, benchmark simulation, and CUDA smoke checks belong on compute nodes.

```bash
mkdir -p logs runs
export SLURM_CONF_SERVER=slurmctl:6817,slurmctl2:6817
sbatch scripts/triton/smoke.sbatch configs/toy_smoke.json runs/toy-smoke-gpu-101 101
sbatch scripts/triton/smoke.sbatch configs/mamujoco_smoke.json runs/mamujoco-smoke-101 101
sbatch scripts/triton/smoke.sbatch configs/smac_smoke.json runs/smac-smoke-101 101
```

Submit the next stage after the relevant smoke result is checked. The smoke allocation requests one GPU, four CPU cores, 16 GB RAM, and at most 30 minutes. The training template requests one GPU, eight CPU cores, 32 GB RAM, and 24 hours; this is an initial request, not a measured completion estimate. Measure throughput first and adjust `--time` before committing a full 10-million-step job.

```bash
sbatch scripts/triton/train.sbatch configs/halfcheetah_2x3_dms.json \
  runs/dev-halfcheetah-2x3-dms-101 101
sbatch scripts/triton/train.sbatch configs/halfcheetah_2x3_ippo.json \
  runs/dev-halfcheetah-2x3-ippo-101 101
```

The full configs use exactly 10,000,000 environment transitions. Initial shorter development configs should be created explicitly before submission; the job script reads their budgets unchanged. `--seed`, `--device`, `--output`, `--algorithm`, and `--total-steps` are supported direct training CLI overrides; other changes belong in JSON.

Scripts use `.venv/bin/python` by default; override `PYTHON` with an absolute interpreter path if needed. `DMS_ROOT` can identify this directory when submitting elsewhere, and `SC2PATH` overrides the shared installation. `PYTHONPATH` includes `src` so the module also runs without an editable package install.

For CPU training, use `scripts/triton/train_cpu.sbatch` with the same arguments. A short compute-node profile found CPU faster for the small IPPO networks; DMS should be profiled on the chosen task and configuration. The profiling command is `sbatch scripts/triton/profile.sbatch`. The [Ant scaling measurements](docs/RUNTIME_SCALING.md) include four- and eight-agent CPU/H100 timings, memory use, and full-budget planning estimates. Device choice does not change the registered environment-transition budget.

For independent SMAC processes, `parallel_env_steps=true` enables concurrent simulator steps with an ordered batch barrier. The policy and update order remain synchronous. The [bounded stepping profile](docs/SMAC_STEP_PROFILE.md) measured simulator throughput separately from model inference; it does not establish a full training speedup. SMAC also restarts long-lived game processes between episodes before their native clock limit, recording the restart count in training metrics.

Once the historical stack passes its smoke check, submit the matched 500,000-transition development pair with:

```bash
sbatch scripts/triton/train_legacy_cpu.sbatch \
  configs/development/legacy-halfcheetah-2x3-ippo.json \
  runs/reader-legacy-halfcheetah-ippo-101 101
sbatch scripts/triton/train_legacy_cpu.sbatch \
  configs/development/legacy-halfcheetah-2x3-dms-soft.json \
  runs/reader-legacy-halfcheetah-dms-101 101
```

These configs explicitly set `mamujoco_backend="legacy"`. The wrapper loads the native libraries and uses `.venv-legacy`; the ordinary CPU/GPU scripts use the modern environment.

Normal jobs target `gpu-h200-141g-ellis,gpu-a100-80g,gpu-h100-80g`. Smoke jobs also permit `gpu-debug`. B300 is excluded because the current PyTorch wheel lacks Blackwell support. Public H200 short queues are omitted because this project has Ellis access. The CUDA preflight executes a real kernel; mixed `gpu-debug` hardware that is incompatible with the environment fails before creating a run directory.

Logs are `logs/<job-name>_<job-id>.out` and `.err`. Scripts record dependencies and any readable Git revision without modifying Git metadata. Each run also saves the imported source files and configuration; these records remain necessary when the working tree differs from its recorded Git revision.

On the inspected Triton login environment (2026-10-25), default Slurm DNS discovery failed. The command-local `SLURM_CONF_SERVER` setting above selects the controllers registered in the cluster's `/etc/hosts`; a scheduler query succeeded with it. It changes no system configuration. Other installations should use their own scheduler settings.

The repository ships 60 configurations in `configs/generated`. To regenerate the complete 12-task matrix with matched baselines and ablations into a new directory:

```bash
.venv/bin/python -m dms.benchmark_configs --targets configs/paper_targets.json \
  --output-dir configs/regenerated --include-ablations
```

The command creates 60 configurations and a manifest listing development and held-out training seeds. The default is the modern v5 track. Add `--mamujoco-backend legacy` and choose another new output directory for historical v2 compatibility configs; these use distinct `mamujoco-legacy-` output names and require the legacy launcher. Generated settings are provisional templates, not a frozen final protocol. It refuses to replace existing files. Generated configurations use k=1; `--agent-obsk 0` generates the separately labelled table-setting sensitivity track into another directory.

The paper's supplementary parameter studies use historical ManyAgentSwimmer 10x2. [Its environment and configuration guide](docs/SWIMMER_ENVIRONMENT.md) documents the fixed packaged XML, native validation, and separate `dms.swimmer_configs` generator. Those study configurations remain unverified and separate from the twelve primary tasks.

## Evaluate and inspect a run

Runs contain `config.json`, `manifest.json`, `metrics.jsonl`, `evaluation.jsonl`, and `latest.pt`. Training metrics include policy/value losses, approximate KL, entropy, and the intention reconstruction/KL terms. Evaluation rows preserve individual episode returns, seeds, lengths, and SMAC wins. Check actual recorded steps rather than inferring progress from file names.

[Artifact export instructions](docs/ARTIFACTS.md) describe a verified local bundle of nine completed HalfCheetah development runs, including checkpoints and negative results. The exporter checks completed budgets, source hashes, and archive contents; no result bundle has been uploaded.

For a fresh final evaluation, run the checkpoint evaluator on an allocated compute node:

```bash
PYTHONPATH=runs/dev-halfcheetah-2x3-dms-101/source .venv/bin/python -m dms.evaluate \
  --checkpoint runs/dev-halfcheetah-2x3-dms-101/latest.pt \
  --episodes 100 --seed 500001 --device cuda \
  --output runs/dev-halfcheetah-2x3-dms-101/final-evaluation.json
```

Use 1,000 episodes per training seed for the registered final SMAC evaluation and 100 for MuJoCo. Training-time evaluations are diagnostics; final test seeds must not select hyperparameters or checkpoints. [The experiment plan](docs/EXPERIMENT_PLAN.md) defines development seeds, held-out training seeds, baselines, ablations, and the evidence needed to call this reproduction ready.

The `PYTHONPATH` above evaluates with that run's saved source snapshot. New evaluation files also record source hashes so changes between training and evaluation remain visible. Use the same environment and dependency versions as training.

Aggregate final evaluation files from independent training seeds with:

```bash
.venv/bin/python -m dms.report --target 3s5z --targets configs/paper_targets.json \
  --inputs runs/final-3s5z-dms-*/final-evaluation.json --output runs/3s5z-report.json
```

The report validates matching settings, budgets, and evaluation protocols, then computes uncertainty across training seeds. Exceeding a reference threshold does not alone certify environment comparability or completion of the required ablations.

## Benchmark tracks

| Track | What it establishes |
| --- | --- |
| Synthetic task | Interface, optimization, evaluation, and checkpoint plumbing |
| Original SMAC / StarCraft II 4.10 | Potential paper comparison after matching maps, difficulty, observation flags, and evaluation protocol |
| Historical MaMuJoCo v2 compatibility | Original observation graph, v2 rewards and XML assets on the documented Gym 0.23 / MuJoCo 2.1 stack; binary parity and empirical improvement remain unverified |
| Ant with explicitly computed contact forces | A separate legacy-backend adaptation that populates force observations and contact penalties; historical equivalence remains unverified and the reference release gate is blocked |
| Modern MaMuJoCo v5 | A declared modern adaptation; higher scores alone do not establish improvement over original MuJoCo v2 results |

For MuJoCo, native joint observations do not expose actuator commands. The adapter explicitly adds a locally sensed previous-control channel, restricted to joints in each observer's configured graph. IPPO receives the same channel. The manuscript also conflicts on observation radius: its task table says `agent_obsk=0`, while its prose says `1`. Initial configs follow the prose (k=1), because a strict k=0 action sensor exposes no peers. The plan also requires the table setting k=0 as a separate sensitivity run. Author clarification is pending. These distinctions must remain visible in any public result claim.

## Citation and license

The software follows the original repository’s [MIT license](LICENSE). [Third-party notices](THIRD_PARTY_NOTICES.md) identify the adapted graph and Swimmer source, packaged XML, and separately installed benchmark assets. Cite the published method using [CITATION.bib](CITATION.bib); no benchmark reproduction result is claimed by this development release.

The current copied public checkout passed **442 tests** (32 native-legacy tests skipped); a separate allocation passed **56 native legacy tests** on the same package source. Offline wheel and isolated import checks verified the modules, licenses, and packaged Swimmer XML. [The integrated validation](reports/current-source-validation.json) reused installed dependencies; it does not claim a fresh dependency installation or benchmark performance. Earlier real-SMAC CUDA smokes separately validated local policy memory and inactive-agent masking.

The [CPU CI workflow](.github/workflows/tests.yml) installs the pinned direct dependencies, runs the synthetic, mocked SMAC, and modern MuJoCo tests, and builds a wheel. Native legacy MuJoCo and real StarCraft checks run separately on allocated Triton nodes. The workflow has not yet run on GitHub because this repository has not been published.
