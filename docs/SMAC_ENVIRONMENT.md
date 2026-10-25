# SMAC environment provenance

Audit date: 2026-10-25. The three installed paper maps are byte-identical to the local official SMAC checkout. The existing runtime logs verify StarCraft II 4.10 / build 75689. This audit establishes the present environment's identity; the accepted paper does not supply a complete simulator/map lock, so exact historical parity remains unverified.

## Installed maps and official source

The runtime installation is `/scratch/work/liz23/3rdparty/StarCraftII`, selected through `SC2PATH` in the Triton launch scripts. Its map files are under `Maps/SMAC_Maps/`.

| Map | Allies / enemies | Agent-step limit | Bytes | SHA-256 |
| --- | --- | --- | --- | --- |
| `3s5z` | 8 / 8 | 150 | 15,697 | `20ce045f7442eba0010548cb26c03f80db93a4b5a4d94eae1d2486404774e088` |
| `8m_vs_9m` | 8 / 9 | 120 | 14,927 | `15c9e65efefde407f8ae34a2bd9b70d46580db730accc15d3477fb281bb609d9` |
| `MMM2` | 10 / 12 | 180 | 17,419 | `797df232260bc50678cc5b48438b606056b2dd799469c872a2635e98d4f5c1d6` |

Each installed file was compared byte-for-byte with `reference/smac/smac/env/starcraft2/maps/SMAC_Maps/<name>.SC2Map`. That checkout has official remote `https://github.com/oxwhirl/smac.git`, clean working-tree status at inspection, and commit `d6aab33f76abc3849c50463a8592a84f59a5ef84` (2024-02-18). The installed `smac==1.0.0` distribution is an editable installation of this checkout. Public provenance is the [pinned official maps directory](https://github.com/oxwhirl/smac/tree/d6aab33f76abc3849c50463a8592a84f59a5ef84/smac/env/starcraft2/maps/SMAC_Maps), not the ignored checkout's availability on another machine.

The official README offers either its `v0.1-beta1/SMAC_Maps.zip` release or copying maps from the source checkout. No matching release archive was found in the inspected local cache/third-party paths, so this audit verifies the checkout files rather than claiming an independently verified release-archive download chain. File timestamps alone are not provenance.

The task registry and environment code inspected are:

| Official source | SHA-256 |
| --- | --- |
| [smac_maps.py](https://github.com/oxwhirl/smac/blob/d6aab33f76abc3849c50463a8592a84f59a5ef84/smac/env/starcraft2/maps/smac_maps.py) | `6c23b24c5d8e35f234e98cf9264baa4776e869cdcfcf3e17bbe0050f145edb83` |
| [starcraft2.py](https://github.com/oxwhirl/smac/blob/d6aab33f76abc3849c50463a8592a84f59a5ef84/smac/env/starcraft2/starcraft2.py) | `98588e57372c1d551b3ab4a2618f97e0a4f4ba33aa6f5daff27fb153f362e5db` |

## Runtime identity and options

`SMACEnv` requests game version `4.10`. PySC2 3.0.0 resolves this to game version `4.10.0`, build `75689`, and data version `B89B5D6FA7CBF6452E721311BFBC6CB2`. The actual launched binary is independently evidenced by `logs/dms-smoke_21192551.err`: its version line reports `B75689 (SC2.4.10)` and its command uses `Versions/Base75689/SC2_x64` with that data version. Subsequent development logs show the same selection. The approximately 62 MiB game executable was not read or hashed during environment construction or this audit.

Installed interface packages are `smac==1.0.0`, `pysc2==3.0.0`, `s2clientprotocol==5.0.17.98310.0`, and `protobuf==3.20.3`. Protocol package version is not the launched game's version. Existing successful execution establishes that the current combination runs; it does not establish historical package parity.

Current resolved options, shared by IPPO and DMS, are:

| Setting | Value / meaning |
| --- | --- |
| `difficulty` | `"7"`, the built-in `VeryHard` opponent |
| `step_mul` | 8 game steps per agent decision |
| `move_amount` | 2 |
| `continuing_episode` | `True`; native timeouts emit `info["episode_limit"]` |
| `obs_last_action` | `True`; visible ally action history comes from native local observations |
| `obs_all_health`, `obs_own_health` | `True`, `True` |
| `obs_pathing_grid`, `obs_terrain_height`, `obs_timestep_number` | All `False` |
| `obs_instead_of_state` | `False`; the adapter does not request or use privileged state |
| `state_last_action`, `state_timestep_number` | `True`, `False`; these native state settings do not make state an actor/critic input |
| `heuristic_ai`, `heuristic_rest` | Both `False` |
| Native shooting / sight ranges | 6 / 9 in the inspected SMAC implementation |

The adapter adds the agent's own previous command, a history-present bit, and public agent identity to its native local observation. Every compared algorithm receives these same channels. Peer actions are read only from each observer's visible ally blocks, not from the runner's complete joint-action buffer. Native legal-action masks remain unchanged.

## Reward and boundary conventions

| Reward option | Current value |
| --- | --- |
| `reward_sparse` | `False` |
| `reward_only_positive` | `True` |
| `reward_death_value` | 10 |
| `reward_win` | 200 |
| `reward_defeat` | 0 |
| `reward_negative_scale` | 0.5; inactive while `reward_only_positive=True` |
| `reward_scale` | `True` |
| `reward_scale_rate` | 20 |

The dense battle reward tracks enemy health/shield changes and enemy deaths. With positive-only rewards, native SMAC uses the absolute value of the enemy-change/death total, including its treatment of shield regeneration. A win adds the win bonus. Scaling divides by `max_reward / 20`; the nominal `max_reward` includes enemy kill bonuses, the win bonus, and enemy maximum health/shields populated when units are initialized. Do not substitute the pre-reset partial value for the final normalization constant.

The adapter preserves this native shared reward. It maps a native episode-limit event to truncation, preserves the final observation, and treats battle outcomes as termination. A genuine battle outcome at the limit takes precedence because native SMAC checks the outcome before its timeout branch. Individual unit death does not terminate the shared episode. Missing battle results after a protocol failure raise an infrastructure error instead of silently counting a loss.

## SC2 game lifetime and scheduled process replacement

The official [PySC2 environment source](https://github.com/google-deepmind/pysc2/blob/master/pysc2/env/sc2_env.py) defines `MAX_STEP_COUNT = 524000` and explains that SC2 fails beyond `2**19 = 524288` game loops. These are simulator loops, not agent decisions or aggregate vector-environment transitions. Native SMAC usually resets a battle by killing units and triggering their respawn in the same game; this keeps the cumulative game clock running.

Two development jobs failed with missing battle results at similar training counts: IPPO CPU job `21192903` last recorded 261,120 aggregate transitions, and DMS GPU job `21192904` last recorded 260,480. Their error logs are `logs/dms-cpu_21192903.err` and `logs/dms-train_21192904.err`; Slurm reported exit `1:0` after 35:20 and 55:12 respectively. With four games and `step_mul=8`, those transition counts imply about 522,240 and 520,960 loops per game before reset/initialization overhead. This supports game lifetime as a likely shared cause. The old runs did not record native game-loop telemetry at failure, so this is a qualified diagnosis, not direct proof that every failure arose from that limit.

The adapter now reads the actual native `observation.game_loop` at `reset()`. After a completed episode reaches 500,000 loops, its next reset closes the old process and creates a fresh native SMAC environment. It does not interrupt an active episode or replace its final observation, reward, or outcome. The default threshold leaves room for a complete episode on all three paper maps before 524,000 loops. Construction also rejects a threshold too close to that bound using `(episode_limit + 10) * step_mul` as its allowance. Native unit initialization can wait for respawn, so that allowance is a practical margin rather than a formal bound on every possible simulator stall.

The first process uses the explicitly requested seed. Scheduled replacements use generation `g = 1, 2, ...` and `int(SeedSequence([explicit_seed, g]).generate_state(1)[0]) % (2**31 - 1)`; an explicit reseed restarts that schedule. This is a reproducible process-seed policy shared by IPPO and DMS. The native game and unit histories restart; training weights and optimizer state do not. This process replacement changes the simulator random stream at declared episode boundaries, which should be retained in experiment provenance.

`metadata.process_lifetime` records the threshold, reason, and seed policy. `runtime_info` reports scheduled replacements, seed generation, process seed, observed game loop, and the current native backend's `force_restarts` count. The latter is not a lifetime total across replaced processes. Scheduled replacement is distinct from SMAC's protocol-recovery restart: the adapter continues to raise on a terminal return without native battle results, rather than inserting a lost battle into training statistics.

The 25 current mock adapter tests pass, including boundary-only replacement, final-transition preservation, repeatable process seeds, explicit reseeding, ordinary resets below the threshold, and threshold validation. A separate real-engine check is available:

```bash
sbatch scripts/triton/check_smac_lifetime.sbatch
```

It requests two CPUs, 8 GiB, and 15 minutes, runs one `3s5z` environment with the normal map episode limit and legal scripted actions, and lowers only the process threshold to 1,000 loops. It requires two scheduled replacements and a complete natural episode afterward, compares every adapter reward/boundary/info with the actual native return, verifies observed game-loop and process-ID changes, checks the deterministic seed sequence, checks empty native/adapter action histories after reset, and rejects any native forced recovery. It is bounded to 3,000 transitions, 16 episodes, and 600 seconds inside the allocation. `runs/smac-lifetime-<job_id>/report.json` and `events.jsonl` retain source hashes, environment metadata, transitions, native outcomes, process seeds, and clock values. This check validates process management; its scripted-policy battles are not benchmark evaluation results.

The native check **passed** as job `21195862` on `milan1`: 329 transitions, six naturally completed episodes, six clean-history checks, two scheduled process replacements, and zero native protocol-recovery restarts in 27.98 instrumented seconds. The observed clocks changed `1180 → 0` and `1090 → 0`, with distinct OS process IDs; process seeds were `101 → 114807835 → 401559344`. Every native transition's reward, end flag, and battle info was preserved, and the sixth episode completed in the third process. Evidence is retained in `runs/smac-lifetime-21195862/report.json` and `events.jsonl`; the tested adapter SHA-256 is `c497d3c1365c3f6a830f906ce6251c2b69fbac06199d272cb86c961d01b0af32`. This shortened-threshold check establishes real-engine restart behavior. A longer training run crossing the default 500,000-loop threshold remains a separate validation of the original failure mitigation.

## Metadata and validation

`SMACEnv.metadata` now records:

- `map_asset`: resolved path, map-relative path, byte count, and SHA-256, or an explicit unresolved status.
- `runtime_selection`: PySC2's selected game/build/data version, explicitly labeled as configuration selection rather than evidence that a binary launched.
- `packages`: SMAC, PySC2, protocol, and protobuf versions.
- `environment_options`: explicitly requested values, retained for compatibility.
- `resolved_environment_options`: actual backend attributes, including unpassed observation/reward defaults and the map's episode limit.

Resolution calls PySC2's map registry and run-configuration selection but never `start()` or `reset()`. Only the small map asset is hashed. If installation/version/map resolution is unavailable, metadata remains unresolved and the adapter's existing construction/launch behavior is preserved. Custom test factories do not import SMAC/PySC2 for provenance and are labeled `not_collected`.

All **19 mock adapter tests passed**. They cover map hashing without a process launcher, missing assets/runtime selection, resolved defaults, and the existing observation/action/boundary contract. An additional read-only probe constructed all three real adapters with `_launch` replaced by an assertion failure; each map hash and effective option set resolved successfully, and each native `_sc2_proc` remained `None`. No StarCraft process was launched for this audit.

## Remaining comparability gaps

The pinned DMS source defaults to `game_version=None`, so its wrapper does not identify the actual game build used for the paper. Its requirements list `s2clientprotocol==4.10.1.75800.0` and `protobuf==4.25.8`, which differ from this runtime and still do not prove the original executable's identity. The original wrapper also defaults to `continuing_episode=False` and `obs_instead_of_state=True`; the present adapter's timeout reporting and local-only learning are explicit implementation choices. Dense reward defaults and difficulty match the inspected original wrapper.

The [official SMAC README](https://github.com/oxwhirl/smac/blob/d6aab33f76abc3849c50463a8592a84f59a5ef84/README.md) explains that game versions can change benchmark performance and that the original SMAC publication used SC2 4.6.2.69232. That statement concerns SMAC's own publication, not evidence about which build generated the DMS paper's targets.

The paper supplies neither map hashes nor a complete package/engine lock; no original experiment artifact confirms that these exact map bytes and binaries generated its reported scores. Existing runs started before the metadata addition were not retroactively rewritten. The paper's seed identities and evaluation aggregation also remain incompletely specified in `PAPER_CONTRACT.md`. Future benchmark claims therefore need the recorded environment identity, matched baseline budgets/information, independent final-seed evaluations, and the declared acceptance protocol; map identity alone establishes no performance result.
