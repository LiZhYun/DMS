# Historical MuJoCo environment audit

The repository provides a tested **modern adaptation** and a separate **historical compatibility backend** using Gym v2 tasks and the original observation graph. The historical stack compiled successfully on an allocated Triton CPU node, and its complete 35-test adapter/trainer suite passed. This establishes runnable simulator and trainer integration, not historical performance or exact paper-binary parity.

The tested Python 3.10 / Gym 0.23.1 / MuJoCo 2.1.0 combination is a compatibility track. The original repository pins older versions, and the manuscript does not provide a complete environment lock. Exceeding a target under v5 alone cannot satisfy the historical-performance acceptance gate.

## Source provenance

The reference checkout is `LiZhYun/DMS` commit `ed94ab0795d794f8bf43cab54ec4737f33319af3`.

| Reference file | SHA-256 |
| --- | --- |
| `reference/DMS/requirements.txt` | `1502c5bd5e4d7c48501bc0c63dfafd9bc2db1ffafee5020d629503f1839d7db8` |
| `reference/DMS/envs/ma_mujoco/multiagent_mujoco/obsk.py` | `ae412f152d13be5df4cd0806d0531d2e9c82b2eb8725a2afd2166dc20aa42088` |
| `reference/DMS/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py` | `5c17e2086050f14b11a1c8016b5e130fd87016dc05b7c1c48e0ff3d29b2b9064` |

The original requirements specify `gym==0.17.2`, `mujoco-py==2.0.2.8`, `Cython==0.29.21`, `cffi==1.14.1`, `fasteners==0.15`, `glfw==1.12.0`, and `imageio==2.4.1`. NumPy and torch are unpinned. These are repository evidence, not proof of the precise binaries used to generate the accepted figures. A literal reconstruction would start with the MuJoCo 2.0 generation supported by those bindings and the old Gym source, rather than silently substituting 2.1.

Installed modern packages inspected on 2026-10-25 are Gymnasium 1.4.0, Gymnasium-Robotics 1.4.2, and MuJoCo 3.17.0. Record actual versions and XML hashes in every experiment manifest; these inspected versions are not a timeless description of the adapter.

## Differences that affect comparability

| Aspect | Original repository | Current modern adapter |
| --- | --- | --- |
| Physics and task | Gym `Ant-v2`, `HalfCheetah-v2`, `Walker2d-v2`, old `mujoco_py` | Gymnasium task v5, native `mujoco` bindings |
| Root observation features | Original launch script supplies no `global_categories`; wrapper defaults to an empty list | Explicit `global_categories=()` now preserves that absence; upstream Robotics default would add root qpos/qvel |
| Local graph radius | Paper table says 0, prose says 1; launch scripts also vary | Both 0 and 1 supported and tested; constructor defaults to table value 0; experiment configuration must name the selected track |
| Joint ownership | Partitions contain actuator IDs, but `step` concatenates actions without using those IDs | Commands route to each graph node's actuator ID |
| Peer commands | Wrapper returns complete `all_last_actions` to the runner | Only graph-visible prior control components reach each observer; both IPPO and DMS get identical declared sensor channels |
| Ant terminal reward | v2 gives the survival bonus on the failing step | v5 pays the health bonus only when healthy |
| Walker physical model | Feet have asymmetric friction, 1.9 and 0.9 | v5 sets both feet to 1.9 |
| Walker terminal reward | v2 adds the alive bonus on the failing step | v5 pays the health bonus only when healthy |
| Reset seed | Original wrapper's `seed` is a no-op and `reset` ignores keyword arguments | Seeds are passed explicitly to each environment reset |
| Time limits | Original wrapper infers `bad_transition` from step count | Explicit termination and truncation arrays preserve the final observation |

The local graph extraction itself matters in addition to simulator version. Legacy `get_joints_at_kdist` sorts joints by label before `build_obs` iterates categories; padding and one-hot public agent IDs are then appended. Its default local categories are `qpos,qvel,cfrc_ext` for Ant's own joints and `qpos` for neighboring joints; HalfCheetah and Walker use `qpos,qvel` locally and `qpos` for neighbors. These defaults are retained by the modern graph library, but its global-node implementation and several ordering bugs were corrected upstream.

For HalfCheetah 3x2 and Walker 3x2, the original named partition has actuator-ID groups `[2,1]`, `[0,5]`, `[4,3]`. Simply splitting the global control vector into contiguous pairs defines a different observation/control assignment. The original concatenation bug does exactly that; it must be disclosed rather than mistaken for the stated partition. The new backend routes these groups correctly. If numerical comparisons require measuring the old assignment's effect, use a separately named diagnostic mode with the original concatenation mapping and compare the same baseline under both mappings. Do not silently make the incorrect assignment the community default.

Control clipping is another declared implementation difference. The old wrapper concatenated raw policy outputs and applied an affine action-range transform without clipping; Gym's reward penalty could therefore use an out-of-bounds command even when the simulator limited its actuator. Both new backends clip each sampled command to its native action bounds **before simulation and reward calculation**, and the local sensor records that executed command. PPO retains the original Gaussian sample for its likelihood ratio. This bounded-control convention is intentional and matched between IPPO and DMS; it is not evidence of numerical equivalence to the old raw-command reward path. Legacy runtime metadata records it as `control_processing`.

A [fixed-policy control-cost audit](CONTROL_COST_AUDIT.md) now quantifies this
choice on seven completed 10M HalfCheetah policies. Across three matched CPU
training seeds, DMS averages 4,476.47 with bounded costs and 3,914.75 with raw
command costs; IPPO averages 3,980.70 and 3,350.12. All 700 original evaluation
episodes replayed exactly. This is a reward-only sensitivity on unchanged
bounded-policy trajectories, not retraining under the old interface or proof
of original binary parity. The audit includes a command readers can repeat.

The installed Robotics presets omit four manuscript configurations. `envs.py` now constructs Ant 8x1, HalfCheetah 3x2, and Walker 3x2/6x1 using current graph node objects and the original named memberships. All nine main tasks were exercised at both radii with real resets and steps; the 23 environment tests passed, including direct checks of simulator actuator routing. This establishes executability and the declared information boundary, not historical return equivalence.

The Ant and Walker reward/model differences are explicit in the installed `ant_v5.py` and `walker2d_v5.py` version histories. Gym's historical task implementations are independently available as [Ant v2 source](https://raw.githubusercontent.com/openai/gym/0.23.1/gym/envs/mujoco/ant.py), [HalfCheetah v2 source](https://raw.githubusercontent.com/openai/gym/0.23.1/gym/envs/mujoco/half_cheetah.py), and [Walker v2 source](https://raw.githubusercontent.com/openai/gym/0.23.1/gym/envs/mujoco/walker2d.py). Preserve their reward formulas and XML assets in the historical compatibility track.

## Isolated compatibility backend

Use a separate `dms/.venv-legacy` or a separately locked container/prefix. Do not downgrade NumPy or Gym in the modern environment or alter the shared base torch environment. A system-site-packages virtual environment can reuse the existing Python 3.10 / torch 2.6 installation, with a local NumPy override, but it is a convenience for this workspace rather than a distributable lock.

The allocated simulator smoke passed with these engineering pins:

```text
Python 3.10
gym==0.23.1
mujoco-py==2.1.2.14
numpy==1.26.4
Cython==0.29.37
MuJoCo binary release 2.1.0 (mujoco210)
```

The build uses `setuptools==68.2.2`, retaining the distutils API. Cython remains below 3 for the unmodified historical extension; NumPy 1.26 avoids importing the old extension against NumPy 2. Gym 0.23.1 avoids introducing the additional Gym 0.24 migration changes. The successful runtime is a compatibility stack, not the paper's older pinned Gym/binding stack.

`mujoco-py` is only the binding; supply the engine separately from the [official MuJoCo 2.1.0 release](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0), record its archive checksum, and set `MUJOCO_PY_MUJOCO_PATH` to its local directory. Version 2.1 removed the engine's license manager, which makes this track more practical than a literal reconstruction of older binary dependencies. This does not prove physics equivalence to MuJoCo 2.0.

The tagged binding's [requirements file](https://raw.githubusercontent.com/openai/mujoco-py/v2.1.2.14/requirements.txt) additionally requires GLFW, imageio, cffi, and fasteners. Exact tested dependency pins are in `scripts/legacy/bootstrap.sh`, and downloaded artifact hashes are recorded in `.cache/legacy/artifacts.json`. No TensorFlow, Ray, old DMS training code, or full legacy requirements file is needed for this backend.

`src/dms/legacy_env.py` now implements `LegacyMaMuJoCoEnv`, selected by `mamujoco_backend="legacy"` in the trainer configuration. `src/dms/legacy_graph.py` ships the required three robot graphs with the original repository's MIT notice and commit/hash provenance. The public package does not import the ignored reference checkout. The backend uses these components only:

1. Gym 0.23.1's `gym/envs/mujoco/{mujoco_env,ant,half_cheetah,walker2d}.py`, registered v2 task IDs, and matching `assets/{ant,half_cheetah,walker2d}.xml`.
2. Original DMS graph definitions and `build_obs` behavior from the pinned `obsk.py`, adapted into the small shipped graph module. It reads the legacy `sim.data` arrays without importing the original whole `MujocoMulti` wrapper or any runner. Synthetic-state checks matched all 72 per-agent observation vectors across the nine tasks at k=0 and k=1 exactly after float32 conversion. This includes sorted joint/category order, clipped and deduplicated Ant contact forces, and the original graph's unclipped Walker joint velocities.
3. New `EnvStep` conversion, deterministic seed forwarding, agent padding, actuator-index routing, and the same explicit local action sensor used by the modern backend.
4. Correct timeout decoding. Gym 0.23.1's [TimeLimit wrapper](https://raw.githubusercontent.com/openai/gym/0.23.1/gym/wrappers/time_limit.py) supplies `TimeLimit.truncated`; map that flag to truncation, and map a remaining `done` to termination. Do not infer timeout solely from elapsed steps: a genuine terminal event can happen at the limit.

## Native prerequisites inspected locally

Read-only checks found GCC/cc, the system linker, C standard-library headers, Python 3.10 headers in the shared base environment, `libgomp.so.1`, `libGL.so.1`, and `libEGL.so.1`. The inspected standard system locations and selected existing Conda prefixes did **not** contain OSMesa development headers/runtime, a linkable unversioned `libGL.so`, general GL/GLEW headers, or a `patchelf` executable. No existing `mujoco210` directory was found in the checked conventional home/scratch locations. This is a bounded inspection, not a search of every filesystem path or compute node.

Follow-up `module spider` resolved that gap: these dependencies exist behind `triton/2024.1-gcc`, even though the initial `module avail` did not expose them. The shell-local command is:

```bash
module load triton/2024.1-gcc gcc/12.3.0 osmesa/11.2.0 glew/2.2.0 patchelf/0.17.2
```

`osmesa/11.2.0` is a shim that depends on `mesa/23.0.3`. The actual Mesa prefix is `/appl/scibuilder-spack/aalto-rhel9-prod/2024-01/software/linux-rhel9-haswell/gcc-12.3.0/mesa-23.0.3-k7odbpd`; `include/GL/osmesa.h`, `lib/libOSMesa.so`, and `lib/libGL.so` were verified there. The GLEW module supplies `GL/glew.h`, and `patchelf/0.17.2` supplies the executable. `ldd` on this OSMesa library found its LLVM and other dependencies without missing entries. No shared package or module file was changed. This avoids downloading or resolving a second native Mesa stack; current unconstrained Mesa packages can also omit OSMesa, so blindly installing the latest Mesa is unsuitable.

The binding compiles a native extension during import even when no rendering is requested. Its [tagged builder](https://raw.githubusercontent.com/openai/mujoco-py/v2.1.2.14/mujoco_py/builder.py) links the Linux CPU build against `mujoco210`, `glewosmesa`, `OSMesa`, and `GL`, and invokes `patchelf`; the [OSMesa shim](https://raw.githubusercontent.com/openai/mujoco-py/v2.1.2.14/mujoco_py/gl/osmesashim.c) requires `GL/osmesa.h`. Supply missing native development files in an isolated prefix or a known cluster module/container. Include the engine's `bin` directory in `LD_LIBRARY_PATH` before import. `MUJOCO_PY_FORCE_CPU=1` chooses the CPU extension consistently and avoids depending on an import-time NVIDIA-library detection heuristic; it does not disable PyTorch GPU training. Setting only `MUJOCO_GL` is not a substitute for this binding's build configuration.

Prebuild once in that isolated writable environment, then smoke it on the actual Triton compute-node image before submitting a matrix. A successful login-container build alone does not establish ABI/library availability on compute nodes. Do not modify global linker paths or install system packages as part of ordinary experiment submission.

## Isolated preparation and allocated smoke

`scripts/legacy/bootstrap.sh prepare` creates only `.venv-legacy` and `.cache/legacy`, reusing the shared base Python/torch read-only through a system-site-packages virtual environment. Override `DMS_LEGACY_BASE_PYTHON` when using another Python 3.10 installation. Preparation downloads the official 4,385,135-byte MuJoCo 2.1.0 archive and its 96-byte checksum file, verifies the checksum, downloads pinned Python wheels and Gym source, and writes `.cache/legacy/artifacts.json` with artifact sizes and hashes. It neither imports MuJoCo nor builds its extension on the login node.

```bash
bash scripts/legacy/bootstrap.sh prepare
sbatch --parsable --job-name=dms-legacy-smoke \
  --partition=batch-milan,batch-csl,batch-skl \
  --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --chdir=/scratch/work/liz23/reproduce/dms \
  --output=logs/%x_%j.out --error=logs/%x_%j.err \
  --wrap='bash scripts/legacy/bootstrap.sh smoke'
```

The smoke mode refuses to run without `SLURM_JOB_ID`. On an allocated CPU node it loads the existing modules, installs only the cached pinned dependencies into `.venv-legacy`, compiles the extension on first import, checks the torch/NumPy bridge, and resets/steps all three v2 tasks. It tests deterministic seeded reset and a two-step time limit, then writes `.cache/legacy/smoke.json`. This is a simulator feasibility check: it does not yet implement the original multi-agent graph adapter or establish parity with the paper's unrecorded binary stack.

The first two allocated attempts exposed batch-shell prerequisites before a simulator could run. Job `21192605` failed because the site Lmod options script did not define `module`; the bootstrap now also sources the shell initializer. Job `21192624` installed the Python dependencies and translated the extension with Cython, then failed because the compute image had no default `gcc`. The bootstrap now loads `gcc/12.3.0` explicitly and verifies that the compiler is present. These failures did not modify shared packages. User-site packages are disabled so unrelated `~/.local` installations cannot satisfy the pinned dependency checks.

Job `21192670` then **passed** on `milan3` with Slurm exit `0:0`, 73 seconds elapsed, and approximately 1.05 GiB maximum resident memory. It built the CPU extension and successfully checked Ant-v2, HalfCheetah-v2, and Walker2d-v2. Each task produced finite observations/rewards, repeated seeded resets exactly, and returned the expected two-step timeout. The loaded engine reports version `210`; Python is 3.10.19, torch is 2.6.0+cu126, and the torch/NumPy bridge passed. The machine-readable evidence is `.cache/legacy/smoke.json`; the original job logs are retained in `logs/dms-legacy-smoke_21192670.{out,err}`. The initial download total was 36,275,058 bytes across 16 artifacts.

The allocated backend matrix is available through `bash scripts/legacy/bootstrap.sh test`, using the same CPU submission settings above. Preparation now also caches the five small test-runner wheels, bringing the verified artifact total to 37,907,021 bytes. The test command writes `.cache/legacy/backend-tests.xml`. Native tests require both `DMS_TEST_LEGACY=1` and a Slurm allocation; the normal test suite runs pure graph checks and skips native imports. The matrix covers all nine partitions at k=0/k=1, seeded resets, actual simulator actuator routing, local control visibility, timeouts, and genuine terminal events occurring exactly at the limit. A four-step DMS rollout/update check exercises the trainer integration and requires finite metrics and active peer inference.

Job `21192737` passed all **35 tests** in 5.07 seconds, with Slurm exit `0:0` and nine seconds total allocation time. The earlier backend job `21192721` passed 34 checks but exposed an invalid terminal fixture: placing Ant inside the floor allowed contact dynamics to return it to the healthy range during one step. The corrected fixture starts above the upper healthy-height bound and verifies the post-step height before checking the terminal flag; no adapter timeout logic changed. Logs for both attempts are retained.

For allocated training or evaluation, `bash scripts/legacy/bootstrap.sh run -m dms.train --config <config.json>` loads exactly the same native modules and launches `.venv-legacy/bin/python`. By default, its only `PYTHONPATH` entry is the repository's live `src` directory. Set `"mamujoco_backend": "legacy"` in that configuration. The `run` mode also requires a Slurm allocation and does not install dependencies. The modern `.venv` remains separate.

The initial matched development pair is `configs/development/legacy-halfcheetah-2x3-{ippo,dms-soft}.json`: CPU, seed 101, k=1, 500,000 environment transitions, four environments, 1,000-step rollouts, and 100 completed warmup episodes. They differ only in algorithm and output directory. These are development configurations, not held-out acceptance results. A short throughput check should explicitly override warmup to zero when measuring active DMS; the actual development pair retains the declared warmup.

## Final evaluation with the run's archived source

Set `DMS_SOURCE_ROOT` when using `bootstrap.sh run` to select an archived package. The value names the directory **containing** `dms/`, such as `runs/<run>/source`, rather than the `dms/` directory itself. An unset value keeps the current training behavior. A relative value resolves from the repository root, independent of the submission directory. The launcher validates that the selected directory contains `dms/__init__.py` and `dms/train.py`, canonicalizes the path, and replaces `PYTHONPATH` after loading native modules. Empty values, colon-separated path lists, missing directories, and incomplete packages fail without falling back to live code. `prepare`, `smoke`, and `test` retain their existing purposes; the source override applies to `run`.

Use the same isolated environment as training. From the repository root, set `legacy_run` to the absolute path of the completed held-out legacy run, then allocate the final evaluator as follows:

```bash
legacy_run="/absolute/path/to/the/completed-heldout-run"
srun --partition=batch-milan,batch-csl,batch-skl \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  env DMS_SOURCE_ROOT="$legacy_run/source" \
  bash scripts/legacy/bootstrap.sh run -m dms.evaluate \
  --checkpoint "$legacy_run/latest.pt" \
  --episodes 100 --seed 500001 --device cpu \
  --output "$legacy_run/final-evaluation.json"
```

If already inside the required Slurm allocation, run the command beginning with `env DMS_SOURCE_ROOT=...` directly. The archived package must include `dms/evaluate.py` and the legacy backend used by the checkpoint. The evaluator refuses an existing output filename, and it records hashes of the evaluation source. Source-directory validation checks package structure, not agreement with the training manifest: compare `evaluation_source_sha256` with the run's recorded `source_sha256` before accepting the result. Keep final evaluation seeds separate from development diagnostics as specified in `EXPERIMENT_PLAN.md`.

Nine isolated launcher tests verify live-source defaults, archived absolute/relative paths including spaces, replacement of inherited `PYTHONPATH`, rejection of invalid/incomplete overrides, and the allocation guard. They use sentinel packages and a mocked native runtime; no game, simulator extension, cluster job, or dependency installation is needed for these wrapper checks.

## Evidence needed before historical performance claims

Preserve package versions, engine and XML checksums, graph source hash, control routing, observation categories, sensor policy, radius, reward parameters, episode limit, and seed schedule with every run. Test all nine tasks for deterministic reset, finite stepping, correct control ownership, original graph-observation snapshots, and separate true-terminal/time-limit behavior. Compare fixed initial states and action sequences against an available original stack before describing simulator parity. Gym-version changes may alter the seeded random sequence even when reset distributions remain similar.

Under strict graph-local sensing, **k=0 exposes no peers**. DMS can still run but peer inference and coordination cannot explain any gain. The paper's conflicting k=0 table and k=1 prose therefore need an explicit experiment label, and any independently chosen action-sensing radius is a separate information assumption. Native joint state is not automatically an action-command sensor. Never resolve this ambiguity by broadcasting the runner's complete action tensor.

Historical acceptance requires the paper targets, matched training budgets, final independent-seed evaluations, and matched-information baselines in the declared compatible environment. Modern v5 results should be reported as a separate track. The original simulator lock, graph/control corrections, and sensor ambiguity remain part of the evidence ledger even after a v2 compatibility backend becomes runnable.
