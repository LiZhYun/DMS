# Explicit Ant contact-force variant

The historical backend offers two distinct Ant modes:

- `ant_contact_forces="native"` is the default and preserves the existing Gym
  Ant-v2 behavior. No additional force computation occurs.
- `ant_contact_forces="computed"` explicitly computes the force buffer before
  Gym calculates contact reward and before the adapter extracts local force
  observations. It is an opt-in environment variant, not verified paper-binary
  parity or evidence of improved learning.

Only legacy Ant supports this option. Selecting `computed` for another scenario
raises an error. The modern backend is outside this change.

## Why the distinction is necessary

All 24 local contact-force channels per agent were exactly zero in the saved
normalization statistics of the native Ant 4x2 IPPO 1M checkpoint and DMS-soft
800k checkpoint. The nonzero variance was solely the normalizer's initial
pseudocount, approximately 1e-10 and 1.25e-10 respectively. These were not rare
zero-contact observations: the force channels had never varied during training.

This matches a documented issue for Ant-v3 and earlier under newer mujoco-py
versions. [Gymnasium's Ant documentation](https://gymnasium.farama.org/environments/mujoco/ant/)
warns that contact forces can remain zero. MuJoCo's
[`mj_rnePostConstraint` documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rnepostconstraint)
identifies the function that populates `cfrc_ext` and explains that automatic
invocation depends on force- or acceleration-related sensors. The installed Gym
0.23.1 Ant XML has no such sensors, and its `do_simulation` does not explicitly
invoke that function.

The original DMS requirements pin mujoco-py 2.0.2.8. That generation may have the
same behavior. Consequently, computing forces must not silently replace the
historical track or be described as restoring an established paper result.

## Measured native diagnostic

Triton job **21197578** completed with exit `0:0` on `milan2` in **11 seconds**,
using a 4-CPU, 8-GiB, 15-minute allocation. It used the archived Ant adapter from
the DMS-soft development run and an immutable copy of its **800,000-transition**
checkpoint, with MuJoCo engine **210**, Gym **0.23.1**, and mujoco-py **2.1.2.14**.

It executed 200 transitions per mode in paired simulator instances. The probe
called post-constraint computation and then restored the derived force and
acceleration arrays. The ordinary instance supplied every checkpoint input;
computed forces never entered that actor. The test checked exact paired
positions, velocities, observations, and native rewards on every step, and
verified that post-constraint computation did not change position or velocity.

| Action mode | Steps with physical contacts | Native maximum absolute force | Computed maximum absolute force | Mean computed contact reward |
| --- | ---: | ---: | ---: | ---: |
| Zero | 195 / 200 | 0 | 24.073878 | -0.00578685 |
| Uniform random in [-1, 1] | 66 / 200 | 0 | 187.485099 | -0.00103111 |
| Frozen checkpoint, deterministic | 194 / 200 | 0 | 34.531764 | -0.00463864 |

Local observations changed on every contact-bearing probe step. The native
contact-reward term was exactly zero in all 600 transitions. The checkpoint's
mean forward reward was **0.00062567** per step, control reward **-0.04056544**,
and total native reward **0.96006023**, confirming standing behavior in this
short probe. Recomputing only the contact penalty at those same physical states
would give **0.95542159** mean reward; that is a counterfactual calculation, not
an evaluation of a policy trained with force observations.

The compact [diagnostic ledger](../reports/ant-contact-force-diagnostic.json)
records source and external Gym code hashes, exact submission command,
checkpoint hash, scheduler/log provenance, and the hash of the unchanged raw
report. That report retains all 600 raw force arrays and local observations,
reward terms, and paired state checks at
`runs/ant-contact-diagnostic-01/native-measurements.json`. The raw report,
standalone script, and copied source/checkpoint are workspace artifacts that
must accompany an evidence export; they are not bundled in a clean checkout.
The evidence remains tied to the unchanged archived adapter and was not
generated using the new computed mode.

## Implementation and information boundary

`legacy_ant.py` defines a lazy factory with a small subclass of Gym's existing
`AntEnv`. It overrides two methods:

1. `do_simulation` first performs the unchanged native substeps, then calls
   `mujoco_py.functions.mj_rnePostConstraint`. Gym's inherited `step` next reads
   the computed force buffer for the existing contact penalty.
2. `reset_model` performs the unchanged seeded native reset, computes forces,
   and returns an observation from the resulting buffer.

There is no global monkeypatch, modified XML, changed integrator, new actuator
mapping, or additional actor channel. Each agent continues to receive only its
existing graph-local `cfrc_ext` slots, clipped to [-1, 1], alongside its original
joint features and declared local action sensors. Root pose, root velocity,
other agents' private observations, and global simulator state are not added.

The reward formula remains

```text
forward_velocity + 1 - 0.5 * sum(executed_control**2)
    - 0.0005 * sum(clip(cfrc_ext, -1, 1)**2)
```

Its numerical contact term changes because `cfrc_ext` is now populated. Computing
forces only after Gym returned its reward would be inconsistent, and could make
the next reward read a stale buffer. The subclass therefore computes them
inside `do_simulation`, before that step's reward calculation.

Use the backend directly with:

```python
from dms.legacy_env import LegacyMaMuJoCoEnv

env = LegacyMaMuJoCoEnv(
    "Ant", "4x2", agent_obsk=1, ant_contact_forces="computed"
)
```

Every Ant manifest records `ant_contact_forces` and the structured
`contact_force_computation` contract, including timing, API, reward behavior,
implementation/Gym source hashes, and the historical-parity limitation. The
engine version and XML hash remain explicit. A native/computed mismatch must
remain a comparability mismatch even when all other task settings agree.
The computed mode uses the distinct benchmark track
`historical_gym_v2_ant_computed_forces_adaptation`; native mode retains
`historical_gym_v2_mujoco210_compatibility`. Results from the new mode must not
inherit reference-comparison eligibility merely from using Gym v2 physics.

## Validation and next experiment

The pure suite rejects unsupported values/scenarios before native imports. The
allocated native suite pairs identical seeds and actions for all three Ant
partitions, checks exact physical-state equality, compares computed observations
against an explicit independent force calculation, verifies the current-step
contact reward, preserves the local information boundary, and checks seeded
reset and timeouts. Default-native tests also require unchanged zero-force
behavior in this pinned stack.

Run inside an allocation with the isolated legacy dependencies:

```bash
DMS_TEST_LEGACY=1 bash scripts/legacy/bootstrap.sh run -m pytest \
  tests/test_ant_contact_forces.py tests/test_legacy_env.py \
  tests/test_legacy_swimmer.py -q
```

Allocated job **21197871** passed **56 tests with no skips**, including all 44
existing historical environment regressions, and then completed a
**128-transition active-DMS smoke** with one evaluation episode. Its exact
frozen source used the distinct computed-force adaptation track, and the smoke
verified the configuration hooks for computed forces and `initial_log_std=-1`.
PPO and auxiliary metrics and checkpoint tensors were finite; peer-gate fraction
was .96875. The [validation ledger](../reports/ant-computed-validation.json)
records the source/test hashes, imported source assertion, pytest path override,
native/computed metadata, metrics, and allocation provenance. The job finished
with exit `0:0` on `milan2` in 23 seconds. An earlier passing check, 21197864,
used the generic track label and is retained as a superseded workspace artifact.

A subsequent learning comparison should train IPPO and DMS from scratch with
the same explicit force mode, local information, seeds, and budget. Existing
native checkpoints and results remain labelled native. Restoring intended local
force sensing and passing a smoke test do not establish improved returns.
