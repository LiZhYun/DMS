# MuJoCo 2.0 comparison

The original repository's **Gym 0.17.2 / mujoco-py 2.0.2.8 / MuJoCo 2.0**
stack now runs in an isolated diagnostic environment. In the controlled probe,
all **129 dynamics arrays** exactly match the current compatibility stack.
Native same-seed resets differ between Gym versions. This supports the tested
physics comparison; it does not establish full training or accepted-paper parity.
[Machine-readable evidence](../reports/mujoco200-runtime-comparison.json).

## Runtime and controls

The engine came from the [official legacy download page](https://www.roboti.us/download.html).
The [official free license](https://www.roboti.us/license.html) supplies an unlocked
activation key for these versions. Normal binding activation was used. Engine,
binding and Gym downloads have recorded SHA-256 hashes; the Python artifacts
also match PyPI's published hashes. A recorded engine checksum identifies the
download; the download page did not supply an independent checksum.

Build job **21281422** completed on `milan2` in 73 seconds. Ant, HalfCheetah and
Walker each passed 32 finite transitions and an exactly repeatable seeded reset.
The isolated prefix is `.cache/legacy200-audit/venv`; existing experiment
environments were unchanged. This uses Python 3.10.19 and modern build dependencies,
including NumPy 1.26.4 and Cython 0.29.37, rather than the complete original
requirements. The paper did not record a complete binary lock.

Comparison job **21281466** completed on `milan32` in 26 seconds. Each capture
uses five variants—HalfCheetah, Walker, Swimmer10x2, native Ant and computed-force
Ant—with three initial states per variant and 128 fixed controls. It records
positions, velocities, contact buffers, native observations, rewards, terminal
flags and k=1 local observations for all corresponding partitions.

The robot XML assets are byte-identical across the two Gym versions. The three
standard robots' `step` and `_get_obs` functions also have identical Python
syntax trees. Each comparison checks exact XML, initial qpos/qvel, action
arrays, graph source and probe source before comparing outcomes.

## Results

| Comparison | Exact dynamics arrays | Exact native reset arrays |
| --- | ---: | ---: |
| Repeat original repository versions | 129 / 129 | 30 / 30 |
| Change Gym 0.17.2 to 0.23.1, keep engine 200 and bindings | 129 / 129 | 0 / 30 |
| Change engine/bindings 200 to 210, keep Gym 0.23.1 | 129 / 129 | 30 / 30 |
| Original repository versions versus compatibility stack | 129 / 129 | 0 / 30 |

Gym 0.17.2 hashes the seed and initializes a NumPy `RandomState`; Gym 0.23.1
uses a PCG64 generator. The probe records those native resets separately,
then explicitly sets matching physical states before applying controls.
Thus reset differences cannot explain the matched dynamics result.

Native Ant's contact arrays are zero throughout all three tested trajectories
under **both** engines. Explicit post-constraint computation gives nonzero
forces and matches exactly across engines. Computing those forces remains an
environment adaptation; it does not restore a demonstrated original behavior.

These are fixed-input simulator trajectories, not benchmark episodes. Unwrapped
dynamics continue after terminal states to preserve the control schedule.
Agreement here does not prove long-horizon closed-loop or learning equivalence.
It also does not resolve the original action-routing bug, raw-command control
costs, observation-radius ambiguity, or unrecorded paper environment details.
The production adapter keeps its validated 210 runtime guards.

## Repeat the probe

The public [probe](../scripts/legacy/engine_probe.py) accepts exactly the three
runtime combinations in the table. Run it separately inside each prepared
isolated environment, with the package on `PYTHONPATH` and native libraries
configured. Native compilation and execution belong on a compute allocation.

```bash
# In an allocated node, using the existing compatibility environment:
bash scripts/legacy/bootstrap.sh run scripts/legacy/engine_probe.py \
  --output runs/engine210-new.npz

# In a separately prepared original-version environment:
python scripts/legacy/engine_probe.py --output runs/engine200-new.npz
```

Use fresh paths; the probe refuses to overwrite outputs. Each NPZ has a JSON
companion with versions, hashes and case definitions. The local verified archive
`dist/dms-mujoco200-audit-20261029.zip` contains all four captures, comparison and
build scripts, source snapshots, logs and a payload manifest. Its Triton setup
scripts contain workspace paths that must be adapted elsewhere. It excludes
engine binaries, activation keys and Python environments, and has no public URL.
