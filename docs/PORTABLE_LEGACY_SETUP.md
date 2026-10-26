# Preparing the optional legacy environment

The historical compatibility backend uses Python 3.10, Gym 0.23.1, MuJoCo 2.1.0, and `mujoco-py==2.1.2.14`. It is separate from the modern community default. The actual simulator build and training integration have been tested on **Aalto Triton** with Python 3.10.19 and PyTorch 2.6.0+cu126. This is a compatibility track, not a claim that the paper used these exact binaries; see [HISTORICAL_ENVIRONMENT.md](HISTORICAL_ENVIRONMENT.md).

The bootstrap does not install PyTorch. It expects a Python 3.10 environment with a stable `torch>=2.6,<3` release; 2.6.0 is the tested release. Other accepted PyTorch versions need their own execution validation and must remain distinct in experiment provenance. Preparation checks the interpreter version, imports PyTorch, and reports its actual version **before any download**. It imports no MuJoCo bindings and compiles no extension during preparation.

Separate [CPU and CUDA Apptainer routes](../containers/README.md) package the native dependencies and bypass Triton library modules. Both images passed native tests, thirteen physics comparisons and four training/checkpoint checks on Triton; CUDA validation used A100. Other host platforms remain unverified. The instructions below describe the existing native bootstrap.

## Choose an isolated Python environment

For an external reader who already has `python3.10`, the most direct route is to create the legacy venv explicitly and install PyTorch there. Run these commands from this repository's root, choosing a fresh environment when `.venv-legacy` does not already belong to this setup:

```bash
python3.10 -m venv .venv-legacy
.venv-legacy/bin/python -m pip install 'torch==2.6.0' \
  --index-url https://download.pytorch.org/whl/cpu
DMS_LEGACY_BASE_PYTHON="$PWD/.venv-legacy/bin/python" \
  bash scripts/legacy/bootstrap.sh prepare
```

The CPU wheel command follows the [official PyTorch 2.6.0 instructions](https://pytorch.org/get-started/previous-versions/#v260). DMS needs `torch`; torchvision and torchaudio are unnecessary. A GPU build can be chosen from that same official version table, with a driver and GPU compatible with its CUDA wheel. The CPU recipe above is an isolated preparation route; it has not been tested as a complete native simulator deployment outside Triton. Installing the base PyTorch wheel is a separate download from the bootstrap's small legacy artifact cache and its 500 MB initial artifact limit.

Alternatively, provide an existing standalone or Conda Python 3.10 prefix that already has PyTorch installed:

```bash
DMS_LEGACY_BASE_PYTHON="/absolute/path/to/python310-prefix/bin/python" \
  bash scripts/legacy/bootstrap.sh prepare
```

When `.venv-legacy` is absent, the bootstrap creates it with `--system-site-packages`, reusing that base prefix's PyTorch read-only. It then checks that the newly created child can actually import PyTorch. A nested venv does not generally inherit packages installed only in the parent venv, so selecting an arbitrary existing venv as the base is insufficient. Use a standalone/Conda prefix, or install PyTorch directly into `.venv-legacy` as in the first recipe. The bootstrap does not repair this by changing shared packages.

An existing `.venv-legacy` is checked and reused. It is never deleted, recreated, or rebound to a different base. An explicit `DMS_LEGACY_BASE_PYTHON` is validated even when the legacy venv exists, but does not change that venv's interpreter. If the existing environment is incomplete or fails its prerequisite check, inspect or repair that isolated environment before retrying; the bootstrap stops before downloads. Empty, missing, or nonexecutable explicit interpreter overrides also fail instead of silently choosing another Python.

If no override is given, an existing legacy venv is sufficient. For a new environment the bootstrap first retains this workspace's tested Triton base if it exists, then falls back to `python3.10` on `PATH`. External readers do not need access to the personal Triton path. The `-I` prerequisite check ignores inherited `PYTHONPATH`, `PYTHONHOME`, and user-site packages, so an unrelated installation cannot hide a missing base dependency. Interpreter symlinks are preserved to retain venv identity.

## What preparation creates

`prepare` creates or reuses `.venv-legacy` and downloads into `.cache/legacy`. It verifies the official MuJoCo 2.1.0 archive checksum, caches pinned Python wheels and Gym source, and records artifact hashes in `.cache/legacy/artifacts.json`. NumPy 1.26.4, Cython 0.29.37, setuptools 68.2.2, Gym, and the historical binding are installed into the legacy venv by the allocated `smoke` step. They are not installed into the selected base prefix or the modern `.venv`.

Preparation requires the ordinary `bash`, `curl`, `tar`, and `sha256sum` commands plus network access. It does not establish that native libraries are available or that MuJoCo can execute on the eventual compute node. The pinned artifact cache is not a lock for the inherited base environment; preserve actual package versions, runtime metadata, and source hashes with each run.

## Tested native execution: Triton

`smoke`, `test`, and `run` retain the existing Triton module setup and require a Slurm allocation. The tested native stack loads:

```bash
module load triton/2024.1-gcc gcc/12.3.0 \
  osmesa/11.2.0 glew/2.2.0 patchelf/0.17.2
```

After preparation, submit the existing smoke from the repository root:

```bash
mkdir -p logs
sbatch --job-name=dms-legacy-smoke \
  --partition=batch-milan,batch-csl,batch-skl \
  --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output=logs/%x_%j.out --error=logs/%x_%j.err \
  --wrap='bash scripts/legacy/bootstrap.sh smoke'
```

This installs the cached legacy dependencies locally and compiles the binding on the allocated node, then checks finite simulation, seeded resets, timeouts, and the torch/NumPy bridge. Successful output is recorded in `.cache/legacy/smoke.json`. `bootstrap.sh test` runs the historical adapter/trainer integration tests in an allocation. Existing jobs and `.venv-legacy` are unaffected by the new preparation checks because native runtime modes do not recreate or revalidate the base environment.

## Other Linux systems: not yet validated

There is currently **no generic system-Linux runtime mode** in this bootstrap. Successful `prepare` on another machine does not make `smoke` or `run` portable: those commands still require Triton's module names, module initialization, library layout, and Slurm guard.

A native port needs Python 3.10 development headers, a compatible C compiler, `patchelf`, OpenGL and OSMesa headers/libraries, the binding's GLEW support, and the MuJoCo 2.1.0 engine in a writable isolated environment. The binding compiles during import and links native libraries even for non-rendering use. Distribution-specific `lib` versus `lib64`/multiarch paths and Mesa/GL dependencies need actual import, simulation, and ABI checks. Mocked paths alone would not validate that deployment. No global package installation or unverified library download is performed by this setup. The separately tested CPU and CUDA containers remove these module dependencies, but execution on other hosts still needs validation; the repository's modern backend remains the default public execution path.
