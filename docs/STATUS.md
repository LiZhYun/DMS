# Community release status

Scope revised **2026-10-29**: deliver clear, faithful software promptly.
Full-budget runs, the complete benchmark matrix and performance above the paper
are no longer required. They are not implied by software readiness.

## Preserved method

- Each observer infers intentions from locally observed past actions and masks.
- The auxiliary posterior can use completed future targets; acting uses only
  the prior. Hidden commands and future targets stay outside actor/critic inputs.
- Soft attention, all-peer conditioning and policy-KL peer selection are explicit
  alternatives. The last implements the paper’s coordination criterion, with
  its causal-interpretation limits stated.
- Independent PPO, finite histories, deterministic prior features and the loss
  reduction are disclosed redesign choices. See [the walkthrough](METHOD_WALKTHROUGH.md),
  [decisions](IMPLEMENTATION_DECISIONS.md) and [paper contract](PAPER_CONTRACT.md).

## Validation

[Community validation](../reports/community-release-validation.json) passed:
**545 tests**, with **32 optional native checks skipped**; a built wheel installed
into a separate target and executed outside the checkout; IPPO, soft DMS and
causal DMS each completed the documented 512-transition toy run; all three
saved-source checkpoint evaluations passed with unchanged training artifacts.
Job **21289857** performed the checks and collector **21289894** verified them.
The initial harness’s final assertion rejected only setuptools-regenerated
package metadata; the collector explicitly verified that source, configs,
scripts and tests remained unchanged. No learning implementation was changed.

Earlier native Walker and SMAC checks are retained in
[the validation report](../reports/auxiliary-first-native-validation.json), with
[separate saved-source checks](../reports/auxiliary-first-saved-source-evaluation.json).
Those native checks apply to their recorded source hashes, not arbitrary future edits.

## Experiments and limitations

The broad DMS performance-search campaigns were stopped on user instruction;
[the scheduler record](../reports/community-release-scope-change-20261029.json)
identifies the jobs. Completed and partial artifacts remain intact. The other
paper’s project and jobs were not changed.

[The development record](DEVELOPMENT_STATUS_20261029.md) preserves both positive
and negative results. Some individual targets were exceeded, while important
DMS comparisons failed. There is no claim of complete benchmark reproduction
or superior performance. Historical MuJoCo binary equivalence remains qualified,
and the manuscript’s observation-radius ambiguity remains explicit. Some raw
run and reference links are workspace-only; no checkpoint bundle is publicly hosted.

The current [release plan](EXPERIMENT_PLAN.md) concerns software correctness,
method transparency and reader usability. Published to [LiZhYun/DMS](https://github.com/LiZhYun/DMS), branch `main`.
The original paper code is preserved on `legacy-paper-code`. The [publication record](../reports/community-publication.json) verifies revision
`03e64b5` and the [successful GitHub CI run](https://github.com/LiZhYun/DMS/actions/runs/40918285093),
including dependency installation, tests and wheel building. No DMS Slurm jobs
remain active. This metadata update does not change the validated Python source.
