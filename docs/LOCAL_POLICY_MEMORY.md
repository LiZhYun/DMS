# Local policy memory

`policy_history_length` adds a bounded GRU encoder over each agent's own local
observations. It works with IPPO and DMS. The default, `0`, keeps the existing
feed-forward policy and checkpoint layout. A positive value includes the current
observation and up to that many total decision observations.

For example, add these fields to an existing SMAC configuration:

```json
{
  "env": "smac",
  "map_name": "3s5z",
  "policy_history_length": 16,
  "mask_inactive_agents": true
}
```

This is separate from `history_length`, which controls DMS's observed-action
history for intention inference. Enabling local policy memory does not establish
a performance improvement; matched experiments are still required.

## Information and data flow

The [environment wrapper](../src/dms/observation_history.py) stores the raw
observations already available to each observer. These may contain locally
visible peer features from the declared sensor. It never adds a teammate's full
observation, global simulator state, or future information.

Let `A` be agents, `H` policy history length, `D` base observation width, `T`
rollout decisions, and `E` environments:

| Stage | Shape | Meaning |
| --- | --- | --- |
| Base environment observation | `[A, D]` | Current local observations |
| Wrapped observation | `[A, H, D+1]` | Raw windows with one validity channel |
| Stored rollout observations | `[T, E, A, H, D+1]` | Whole normalized windows used for collection |
| One observer's PPO minibatch | `[batch, H, D+1]` | Selected decision windows, preserving history order |
| Encoded local feature | `[batch, hidden_dim]` | Final GRU state for each valid suffix |

The last window slot is the current observation. Unfilled slots are left padded;
the final channel is exactly `0` or `1` and is never normalized. `env.obs_dim`
continues to report `D`, the normalizer's base width. Environment metadata records
the actual window shape and reset convention.

The [policy](../src/dms/policy.py) has separate actor and critic
[encoders](../src/dms/recurrent.py). Each runs a unidirectional GRU from zero over
the valid suffix, using packed batches. An empty history produces a zero local
feature. DMS's current intention context is appended after this local encoding;
its cached gates and the available-action mask retain their existing roles.
Causal peer ablations reuse one encoded local feature while varying peer context.

## Rollouts, updates, and boundaries

The wrapper keeps **raw** windows between PPO updates. During collection, the
policy normalizes the whole window using observation moments that remain frozen
throughout that rollout. The runner caches these **normalized** windows, current
intention features, and gates. PPO recomputes the GRU from each cached window
under the current policy parameters. It does not reuse a learned feature or an
old hidden state from collection. Independent transition minibatches are valid
because every sample carries its full causal window; observations inside a window
must stay ordered.

After an update, a continuing episode retains its raw history. The next rollout
normalizes that history with the new moments. Observation statistics count only
the newest valid raw observation once, excluding duplicated older history and
padding. With SMAC's `mask_inactive_agents=true`, inactive decision rows are also
excluded from PPO, entropy, value/PopArt targets, and the KL stopping statistic.
Team rewards and episode boundaries still enter GAE before that filtering. A
fully inactive observer skips its optimizer and moment updates. History validity
excludes empty/dead windows from observation moments even if inactive-loss
filtering is disabled.

Episode reset clears every agent's window before appending the reset observation.
In SMAC, a no-op-only available-action mask clears that dead agent's window;
other agents retain their own histories. A subsequent live observation starts a
fresh suffix. The wrapper preserves a timeout's final window for value
bootstrapping, then clears history when the runner explicitly resets. Returned
snapshots own their memory, so stepping or resetting cannot change saved data.

Actor encoder parameters use `actor_lr`; critic encoder parameters join the
critic MLP under `critic_lr` through `critic_parameters()`. The critic's final
linear head remains the PopArt head. Action masking applies to the current
decision distribution, including when its history is empty.

## Checkpoints and scope

Checkpoint configuration records `policy_history_length`. Evaluation reconstructs
the wrapper and both encoders, initializes new episode histories, and loads the
saved weights and normalizers. An older checkpoint missing this option defaults
to `0` and needs no encoder keys. Checkpoints remain evaluation checkpoints:
simulator state and an in-progress observation window are not saved for exact
trajectory continuation.

This bounded encoder is a deliberate alternative to the original stateful RNN.
It forgets observations older than `H` and recomputes up to `H` recurrent steps
per decision and PPO sample. Packed execution reduces batching overhead, but
larger windows still increase computation and rollout storage. It avoids stale
hidden-state seeds across policy updates. The module's general time-major API
supports explicit reset masks and detached chunk-start states for testing or
future stateful use; the integrated bounded path does not use that interface.

## Validation status

The focused checks passed: 12 encoder tests, 14 recurrent-policy tests, seven
wrapper tests, seven training/evaluation integration tests, and the three existing
policy regression tests. They cover causal and per-observer isolation, padding,
episode/death gradient boundaries, collection/PPO likelihood agreement, timeout
bootstrap, raw-window renormalization, optimizer ownership, and checkpoint
compatibility. The integration tests use toy and in-memory mock environments.

An independent toy probe used active history length 3 for eight diagnostic
transitions and complete normal/held-peer/masked-peer intervention episodes of
32 steps each. All diagnostic paths completed, retained history metadata, and
left the model state hash unchanged. This verifies plumbing, not benchmark
performance. Native validation job **21197247** completed with exit `0:0` on
V100 node dgx21 in 1 min 49 s. The copied public checkout passed **386 tests**
(26 native legacy checks skipped), then trained active DMS on real SMAC 3s5z
for **512 transitions**, with history length 16, four finite PPO/auxiliary
updates, eight completed training battles, dead-agent samples, and one separate
evaluation episode. An offline wheel build and isolated import check verified
the new modules and exact packaged Swimmer XML. [The validation ledger](../reports/recurrent-validation.json)
records hashes and actual runtime metadata. These checks establish execution
and regression coverage, not learned performance.

Run the focused checks from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_recurrent.py tests/test_recurrent_policy.py \
  tests/test_observation_history.py tests/test_recurrent_training.py \
  tests/test_policy.py
```
