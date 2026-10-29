# Auxiliary updates before PPO

`intention_update_order="before_ppo"` is an experimental update schedule. The
default remains `"after_ppo"`. It requires `intention_rl_lr=0`; combining the new
schedule with PPO gradients through the prior has not been validated.

For each observer, the new schedule freezes its original past-history filter,
fits the auxiliary intention model, and recomputes detached prior means from
those same past histories. PPO then uses these updated means with the original
collected action log probabilities, normalized observations and peer gates.
Rows with no selected peer retain their cached context. The caller's rollout
arrays are never rewritten. Completed future actions supervise auxiliary
learning; the refreshed policy-input path reads past histories only.

With sequential PPO, only the observer whose turn has arrived receives its
auxiliary update. Other observers retain their behavior models until their own
turn. The predecessor likelihood correction includes both the updated encoder
and the updated actor. Updating every encoder before the sequential loop would
violate that ordering assumption.

`aux_updates` still means steps per observer per complete rollout update, not
steps per PPO minibatch. The new schedule draws auxiliary windows separately
for each observer; the old schedule shares selected windows across observers
within each auxiliary step. Natural random-number consumption also changes.
The paired diagnostic's restored random states are not used in production.
IPPO, zero auxiliary steps, and rollouts without valid future windows retain
the original update path exactly.

The motivation is a [one-batch counterfactual](../reports/development-walker-residual-auxiliary-order-probe.json):
updating the prior after PPO shifts the actual policy outside the surrogate
just optimized. Auxiliary-first PPO with refreshed inputs reduced measured
Walker policy KL by 14.96% and improved the fitted surrogate by 10.07% on one
late checkpoint. Merely moving auxiliary updates while keeping stale inputs
exactly matched the original update. This is diagnostic evidence, not a return
improvement or a claim that the old default was incorrectly implemented.

The full unit suite passed 543 tests, with 32 native-only skips. Tests cover
continuous and categorical policies, concatenating and residual inputs,
independent and sequential PPO, unchanged rollout data, detached inputs,
unprocessed behavior policies and encoder-inclusive predecessor corrections.
The [native GPU checks](../reports/auxiliary-first-native-validation.json)
passed for Walker and SMAC: each compares default model, optimizer, metric and
RNG states against the preceding trainer over 8,000 transitions, then exercises
8,000 transitions of active auxiliary-first training. Walker ran on H200;
SMAC ran on A100. Smoke warmup is zero; development learning retains the
original 100-episode warmup. No performance promotion is supported yet.

## Matched development learning

The [submitted ledger](../reports/development-auxiliary-order-3m-seed101.json)
declares three fresh arms on each of Walker 3x2 and SMAC 8m_vs_9m:

| Arm | Auxiliary schedule | Peer inputs |
| --- | --- | --- |
| `after` | Existing PPO then auxiliary update | Enabled after 100 completed training episodes |
| `before` | Auxiliary update then refreshed-input PPO per observer | Enabled after 100 completed training episodes |
| `no-input` | Same before-PPO schedule | Disabled throughout training |

All six use **A100**, development training seed **101**, and **3M transitions**.
Walker retains four environments, 1,000 steps per rollout, independent PPO and
local observation history 32. SMAC retains 20 environments, 160 steps per
rollout, sequential PPO, inactive-agent masking and local observation history
16. Residual fusion, auxiliary LR 1e-3, two auxiliary steps, prior RL LR zero,
discounts, native rewards and other settings remain unchanged.

The fresh after-PPO arm supplies the matched hardware control. Earlier H100
scores are contextual references, not a controlled update-order comparison.
The initial H200 preparation was changed before any learning submission:
13 H200 nodes were draining, and the remaining node was fully allocated with a
planned reboot. The ledger preserves that preparation and the selection reason.
Typed A100 requests retain the Ellis-first compatible partition list; public
H200 queues and incompatible B300 hardware are excluded. Each endpoint uses a separate frozen-policy
100-episode CPU evaluation, seeds 400101–400200, with source/checkpoint hashes
and zero optimizer steps checked. Native validation passed before submission.
Automatic collectors retain failures as well as successful endpoints.

| Task | After-PPO training / evaluation | Before-PPO training / evaluation | No-input training / evaluation |
| --- | --- | --- | --- |
| Walker 3x2 | 21288097 / 21288119 | 21288099 / 21288120 | 21288101 / 21288121 |
| SMAC 8m_vs_9m | 21288103 / 21288122 | 21288105 / 21288123 | 21288107 / 21288124 |

Before execution, the original dependent evaluators were replaced to correct
the expected byte hash of serialized `Config` fields. The prepared dict had
appended the new schedule field; `dataclasses.asdict` writes declaration order.
Field values and the scientific protocol are identical. The ledger retains
the cancelled evaluation submissions and their replacement specs. Training
jobs, frozen source, configs and validation evidence were not changed.

This initial learning test can reject the candidate on one development seed;
success would still require independent training seeds and the final budget.
Held-out training seeds and final evaluation seeds remain unused.

To launch a separate run with the checked-in configuration on Triton, first
prepare the [native historical environment](PORTABLE_LEGACY_SETUP.md), then
choose a fresh output directory:

```bash
sbatch --gpus=a100:1 --cpus-per-task=1 --mem=4G --time=06:00:00 \
  scripts/triton/train_legacy_gpu.sbatch \
  configs/development/walker-auxorder-before-3m-a100-101.json \
  runs/my-walker-auxiliary-first 101
```

The corresponding SMAC command uses the established StarCraft II setup and
requests resources for all 20 game processes:

```bash
sbatch --gpus=a100:1 --cpus-per-task=4 --mem=32G --time=07:00:00 \
  scripts/triton/train.sbatch \
  configs/development/8m-auxorder-before-3m-a100-101.json \
  runs/my-smac-auxiliary-first 101
```
