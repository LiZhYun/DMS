# Legacy MuJoCo containers

This optional Apptainer image packages Python 3.10.19, CPU PyTorch 2.6.0,
Gym 0.23.1, `mujoco-py` 2.1.2.14 and MuJoCo 2.1.0. It runs the compatibility
backend without a host Python environment or Triton library modules. It has
been tested with Apptainer 1.5.4 on Triton x86-64 Linux. The CUDA variant below
also passed native validation on A100; other hosts remain unverified. Neither
image contains StarCraft II.

## Build and check

From the repository root, on a workstation or allocated compute node:

```bash
bash scripts/legacy/container.sh build .cache/containers/legacy-cpu.sif
bash scripts/legacy/container.sh test .cache/containers/legacy-cpu.sif
```

The host needs [Apptainer](https://apptainer.org/docs/user/latest/build_a_container.html),
network access during the build, and working
[fakeroot support](https://apptainer.org/docs/user/latest/fakeroot.html).
On a cluster, request a compute allocation before building or simulating.
The verified build used two CPU cores, 8 GB RAM and 8m20s; allow extra time for
downloads. Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to suitable scratch
directories when the default cache or temporary filesystem is too small.

The definition pins the base image by digest and checks the MuJoCo archive
SHA-256. It pins the main Python dependencies; Debian packages and some Python
transitive dependencies are resolved at build time. Their actual versions are
saved inside `/opt/dms-environment/`. This is not a fully locked rebuild.
The wrapper refuses to overwrite an existing image and writes its checksum and
inspection metadata only after the native image test passes. A failed build or
test may leave an image without those success records; use a new path to retry.

`mujoco-py` opens an import lock in its installed package even after compilation.
Execution therefore uses `--writable-tmpfs`. The wrapper skips the automatic
read-only build test and immediately executes the same test with this overlay;
the build command fails if that native test fails. No host libraries are loaded
from the Triton legacy environment.

## CUDA image

The CUDA variant uses PyTorch **2.6.0+cu126** with the same pinned Python,
Gym and MuJoCo versions. [CUDA validation](../reports/legacy-cuda-container-validation.json)
passed on an NVIDIA A100-SXM4-80GB: 56 native tests without skips, thirteen physics
cases with all 130 arrays exactly matching the CPU image, four active DMS
training/checkpoint checks and a separate saved-source CUDA evaluation. The
wrapper preserved scheduler GPU visibility and correctly hid all devices when
`CUDA_VISIBLE_DEVICES` was explicitly empty. Other GPU models and hosts remain
unverified for this image.

Build on a CPU allocation, then test on an allocated NVIDIA GPU. The measured
CUDA build used two CPU cores, 16 GB RAM and 12m58s; the image is 3.24 GB. Its
separate GPU validation took 3m03s. Allow additional time for downloads:

```bash
bash scripts/legacy/container.sh build-cuda .cache/containers/legacy-cuda.sif
bash scripts/legacy/container.sh test-cuda .cache/containers/legacy-cuda.sif
bash scripts/legacy/container.sh run-cuda .cache/containers/legacy-cuda.sif \
  -m dms.train \
  --config configs/development/legacy-halfcheetah-2x3-dms-soft.json \
  --device cuda --total-steps 4000 --output runs/container-cuda-halfcheetah-smoke-101
```

CUDA modes use [Apptainer's `--nv` support](https://apptainer.org/docs/user/1.5/gpu.html)
to expose the host NVIDIA driver. The host GPU and driver must support this
PyTorch/CUDA build; a working host Python environment is not required.
The wrapper preserves the scheduler's `CUDA_VISIBLE_DEVICES`, including an
explicitly empty value. `test-cuda` requires a real CUDA matrix operation to
succeed, then checks native MuJoCo execution. The build's native test runs on
CPU, so building a CUDA image alone does not validate GPU execution.
MuJoCo physics and offscreen rendering remain on CPU; CUDA runs the networks.

Use `run-cuda` and `--device cuda` for saved-source CUDA checkpoint evaluation,
with the same `DMS_SOURCE_ROOT` override shown below. Keep the image checksum,
accelerator model and observed library versions with experiment artifacts.

## Run a short native check

```bash
bash scripts/legacy/container.sh run .cache/containers/legacy-cpu.sif \
  -m dms.train \
  --config configs/development/legacy-halfcheetah-2x3-dms-soft.json \
  --device cpu --total-steps 4000 --output runs/container-halfcheetah-smoke-101
```

This short run checks execution, not learned performance. It retains the
configuration's intention warmup; the full validation below separately exercises
active intention updates. Use a fresh output directory for each run. The wrapper
mounts this repository read-only at `/workspace` and only `runs/` writable.
Use repository-relative configuration paths and `runs/...` output paths.
Commands execute with the image's Python and `src/` on its import path.
For full training, omit the short-budget override and use the registered
configuration and resource allocation.

To capture the fixed-action physics probes for comparison with another installation:

```bash
bash scripts/legacy/container.sh run .cache/containers/legacy-cpu.sif \
  scripts/legacy/physics_probe.py --output runs/reader-physics.npz
```

The companion JSON records the cases, environment metadata and source hashes.
Compare corresponding NPZ arrays and metadata; equal compressed-file bytes are
not required. These probes do not train a policy.

Evaluate with the completed run's saved source rather than a later working tree:

```bash
DMS_SOURCE_ROOT="$PWD/runs/container-halfcheetah-smoke-101/source" \
  bash scripts/legacy/container.sh run .cache/containers/legacy-cpu.sif \
  -m dms.evaluate --checkpoint runs/container-halfcheetah-smoke-101/latest.pt \
  --device cpu --episodes 2 --seed 400801 --output runs/container-smoke-evaluation.json
```

This is an execution check with development episode seeds. Final benchmark runs
must use their registered evaluation budgets and seed blocks.
`DMS_SOURCE_ROOT` selects a host directory containing `dms/train.py`; the wrapper
mounts it read-only at `/dms-source` and uses that package for imports. By default
it selects this checkout's `src/`. Preserve the image checksum alongside each run:
run manifests record actual Python package and
environment versions, but do not automatically include the outer SIF checksum.
Training and evaluation devices must be CPU for the CPU-only image.

## Validation and limits

[The validation ledger](../reports/legacy-container-validation.json) preserves
the build attempts, final image checksum, source inventory and raw results.
Build **21213009** and validation **21213010** completed successfully:

- All **56** native adapter/contact-force/Swimmer tests passed, with zero skips.
- **13,312** deterministic simulator transitions across thirteen cases produced
  **130 exactly matching arrays** against the same-node Triton compatibility
  stack: observations, rewards, flags, sensed controls, visibility and native
  physics state. The cases cover all nine primary MuJoCo partitions, Swimmer
  10x2, and the three computed-contact Ant variants.
- Four 64-transition DMS checks exercised active intention learning, checkpoint
  saving and independent checkpoint evaluation for HalfCheetah, computed-contact
  Ant, Walker and Swimmer. All losses and results were finite, and the original
  checkpoint files were unchanged during evaluation.

These checks establish execution and agreement on the recorded physics probes.
They do not establish full training equivalence between CPU and CUDA PyTorch,
support for an untested host, or parity with the paper's original MuJoCo 2.0
binary. The compatibility benchmark and computed-contact Ant adaptation retain
their separate labels. The public physics probe also reproduced all 130 arrays
in job **21213435**. The reader wrapper separately passed its native test,
all 56 adapter tests and
the documented 4,000-transition CLI run in job **21213356**. Saved-source
checkpoint evaluation **21213471** verified the imported source, environment
and package versions and an unchanged checkpoint. No prebuilt image has been
published.

The documented CUDA 4,000-transition training command and saved-source evaluation
also passed in job **21215552**, using the current source with explicit accelerator
metadata. [The provenance validation](../reports/runtime-metadata-validation.json)
records that command check and exact before/after state comparisons for the
metadata-only change.
