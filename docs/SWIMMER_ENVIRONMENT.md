# Historical ManyAgentSwimmer 10x2

`ManyAgentSwimmer` adds the paper's ten-agent parameter-study task to the historical compatibility backend. Each agent controls two consecutive joints in a 20-actuator chain. This environment is separate from the nine primary MuJoCo comparison tasks; its implementation does not establish a reproduced learning curve or a performance improvement.

Select it with these fields in a training configuration:

```json
{
  "env": "mamujoco",
  "mamujoco_backend": "legacy",
  "scenario": "ManyAgentSwimmer",
  "agent_conf": "10x2",
  "agent_obsk": 1
}
```

Only the published `10x2` partition is supported. Run through the existing [legacy launcher](../scripts/triton/train_legacy_cpu.sbatch) after the [legacy environment preparation](PORTABLE_LEGACY_SETUP.md). Native execution uses the same isolated Gym 0.23.1, mujoco-py 2.1.2.14, and MuJoCo 210 stack as the other historical tasks. The package imports no native simulator until constructing an environment. It adds no Python dependency or runtime asset-generation step.

## Physics and local information

[legacy_swimmer.py](../src/dms/legacy_swimmer.py) implements the reference environment's frame skip of four and reward `root_x_velocity - 0.0001 * sum(control**2)`. The packaged [XML](../src/dms/assets/manyagent_swimmer_10x2.xml) retains timestep 0.005, RK4 integration, fluid density 4000, viscosity 0.1, motor gear 150, and control bounds [-1, 1]. A policy step therefore advances 0.02 simulated seconds. Resets independently perturb each initial position and velocity by uniform noise in [-0.1, 0.1]. There is no native terminal condition; the adapter truncates at 1,000 steps and preserves the final observation.

The graph contains joints `rot0` through `rot19`, with chain edges between consecutive joints. Agent `i` controls actuators `2*i` and `2*i+1`; their position and velocity indices are the corresponding final 20 native coordinates. Actors receive only their declared local joint observations, public identity, padding masks, and locally sensed previous executed controls. The native full observation is never an actor or critic input.

At `agent_obsk=0`, each agent observes its own two joint positions/velocities and controls: no peer intention can be inferred from the declared sensor. At `agent_obsk=1`, the two endpoint agents additionally observe one adjacent joint, and interior agents observe one joint from each adjacent agent. Neighbor features contain positions, not neighbor velocities; previous-control visibility follows exactly those joint boundaries. Both IPPO and DMS receive the same sensor channels. Reset clears all previous-control observations.

The manuscript's k=0 table conflicts with its k=1 prose. This adapter supports both explicitly and makes no claim to recover the paper's unrecorded sensing convention. The default paper-compatibility experiments use k=1; k=0 remains a distinct sensitivity condition.

## Asset and source provenance

The source is LiZhYun/DMS revision `ed94ab0795d794f8bf43cab54ec4737f33319af3`. The upstream MIT notice is retained in the new module and asset. The relevant source fingerprints are:

| Source or asset | SHA-256 |
| --- | --- |
| Original `manyagent_swimmer.py` | `c3e52d46867b4bed3063e0b572ba15060d105456a53c7d75d13edff45b835781` |
| Original `manyagent_swimmer.xml.template` | `e258a36037bd7dca12e563ed9c0986a6239b6394ea9ff76edaf49f65ebeb9831` |
| Original `obsk.py` | `ae412f152d13be5df4cd0806d0531d2e9c82b2eb8725a2afd2166dc20aa42088` |
| Reference 10x2 rendering before the added license comment | `636870e46fad00f8cc9957150c2e706c0caba5cb9b4a06ae8a12fe143b8ce388` |
| Packaged 10x2 XML including the license comment | `2793aa62f97be1bceed660c5ea0860716991f64d2f9a1824f66ca914e037e20b` |

The asset was generated once from inspected literal body/actuator templates. An independent extraction of those literal strings from the reference AST reproduced the same bytes before adding the license comment; no reference Python code was executed. Native constructors read this fixed asset, avoiding the original generator's shared writable path and Jinja2 dependency. Wheel package data includes the XML, and training source snapshots copy it alongside the Python modules.

Environment manifests record the actual XML hash, original environment/template hashes, adapted graph hash, actuator routing, observation sizes, reward, frame skip, timestep, fluid parameters, and native package versions. Python source hashes and XML hashes remain separate manifest fields. `historical_binary_parity_verified` remains false: MuJoCo 210 execution does not recover the paper's unspecified original binary stack.

## Verification

[test_legacy_swimmer.py](../tests/test_legacy_swimmer.py) covers the fixed asset, graph ordering, control visibility, and exclusion of root/distant state without importing native dependencies. Its native checks verify seeded reset values, clipped actuator routing, cleared history, the final timeout observation, and exactly 1,000 steps before truncation. A separate direct MuJoCo simulation is initialized from the same state and follows eight scripted actions; resulting positions, velocities, forward rewards, and control penalties match the adapter.

CPU validation job **21196579** completed with exit `0:0` on milan1 in eight seconds. It passed **44 tests** in 4.86 seconds: the existing 35 historical-adapter checks and all nine new swimmer cases. The [validation ledger](../runs/swimmer-validation-20261025-01/validation.json), [JUnit report](../runs/swimmer-validation-20261025-01/native-tests.xml), and [log](../logs/dms-swimmer-tests_21196579.out) retain the frozen source, asset, and test fingerprints. These results establish compatibility on the tested stack, not historical binary identity.

The bounded DMS smoke, job **21196586**, completed with exit `0:0` on milan1 in 52 seconds. It used exactly **128 training transitions**, one environment, ten agents, zero intention warmup, two PPO/auxiliary updates, and one separate 1,000-step evaluation episode. Both update records were finite, had nonempty observed forecast targets, and used peer intentions: gate fractions were 0.196875 and 0.2. The training budget was shorter than an episode; no completed training episode is claimed. The separate evaluation returned -75.6942, which is execution evidence rather than a performance result.

The [configuration](../runs/swimmer-validation-20261025-01/smoke-config.json), [raw metrics](../runs/swimmer-validation-20261025-01/smoke/metrics.jsonl), and [manifest](../runs/swimmer-validation-20261025-01/smoke/manifest.json) retain this limited scope. All 17 recorded Python files and the packaged XML match the frozen validation source; the manifest's actual XML hash also matches. The validation ledger records 13 smoke checks and an offline wheel build from an isolated source copy. The wheel contains `legacy_swimmer.py` and the exact validated XML; no package installation was performed.

## Remaining empirical coverage

The accepted manuscript includes swimmer curves for coordination thresholds 0.2, 0.6, and 0.8, and joint long/short settings `(K,l)=(100,6), (10000,4), (1000000,2)`, each reaching ten million training transitions. Its prose also lists delta 0.4, which is absent from that swimmer figure. No such sweeps have been run here. A coordination-threshold sweep must use the causal mode, since soft attention does not use that threshold; history-filter sensitivity needs measured rejection rates. The bundled standalone swimmer main-result asset is not an included manuscript main-results panel. The supplementary target in [paper_targets.json](../configs/paper_targets.json) remains unverified and separate from the primary configuration generator and reporting gate.

The separate [swimmer_configs.py](../src/dms/swimmer_configs.py) writes seven supplemental configurations and a study manifest. One reference condition, delta 0.6 with `(K,l)=(10000,4)`, belongs to both plotted studies. The other configs are delta 0.2 and 0.8, the explicitly prose-only delta 0.4, the two remaining joint `(K,l)` settings, and an IPPO control. The joint settings do not isolate the separate effects of K and l. All DMS variants use causal coordination and retain history filtering. Parameters inherited from the template, including the latent-initialized decoder and beta 0.1, are provisional engineering choices.

From the repository root, write either sensing track to a fresh directory:

```bash
PYTHONPATH=src .venv/bin/python -m dms.swimmer_configs --output-dir configs/swimmer-k1
PYTHONPATH=src .venv/bin/python -m dms.swimmer_configs --agent-obsk 0 --output-dir configs/swimmer-k0-sensitivity
```

These commands create files only. Existing destinations are protected from replacement, and no training or submission occurs. Every generated config declares ten million transitions and seed 101 as a development placeholder; the manifest keeps reserved final seeds separate and requires a common source/settings freeze before their use. k=0 outputs explicitly record that no peer controls are sensed. Neither matrix can satisfy the primary twelve-task reporting gate.

The matched [IPPO pilot](../configs/development/swimmer10x2-ippo-pilot.json) and [causal-DMS pilot](../configs/development/swimmer10x2-dms-causal-pilot.json) each declare 500,000 transitions, four environments, and five diagnostic episodes every 100,000 transitions. They differ only in algorithm and output. This budget extends past the configured 100-episode intention warmup and additional 100-episode causal warmup. They have not been submitted by the generator or as part of this adapter validation. The configuration tests passed **20 cases**, including the nine existing primary-matrix checks.
