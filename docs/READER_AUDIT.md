# Fresh-reader reproduction audit

Reviewed 2026-10-25 against the current working tree. This was a read-only walkthrough of installation, environment setup, launchers, saved-source evaluation, and reporting. It did not perform a new network installation or submit training jobs. File/line references describe this snapshot; other contributors are actively improving it.

**Outcome.** The synthetic example and modern environment have a clear repository-based entry point. Legacy preparation now checks and documents its Python/Torch prerequisites before downloading, while native legacy execution is explicitly supported only on the tested Triton stack. Final release still needs accessible result artifacts and completed empirical acceptance evidence. Passing unit tests and installing a wheel do not close those items.

**Issues found and resolved during this review.**

| Finding | Verified resolution |
| --- | --- |
| The SMAC command installed `reference/smac`, which is ignored and absent from a fresh checkout. | [README.md:30](../README.md#L30) now gives clone/install commands at pinned revision `d6aab33f76abc3849c50463a8592a84f59a5ef84`, a StarCraft 4.10 installation link, an explicit `SC2PATH`, and map-copy commands. The map hashes and upstream provenance are in [SMAC_ENVIRONMENT.md:15](SMAC_ENVIRONMENT.md#L15). External game installation is still required and clearly stated. |
| The generator example targeted the already shipped `configs/generated` files and would fail its overwrite guard. | [README.md:113](../README.md#L113) now identifies the shipped matrix and regenerates into `configs/regenerated`. The intentional guard remains in [benchmark_configs.py:149](../src/dms/benchmark_configs.py#L149). |
| The historical final-evaluation example used seed 20001 despite the experiment plan specifying 500001. | [HISTORICAL_ENVIRONMENT.md:131](HISTORICAL_ENVIRONMENT.md#L131) now uses 500001, agreeing with [EXPERIMENT_PLAN.md:33](EXPERIMENT_PLAN.md#L33). |
| Training wrappers could override an intended frozen package by prepending live `src` to `PYTHONPATH`. | The modern wrapper now validates `DMS_SOURCE_ROOT`, replaces inherited `PYTHONPATH`, and checks imported package/trainer origins before creating the output: [train.sbatch:35](../scripts/triton/train.sbatch#L35). The legacy launcher already resolves the same override before resetting native-library paths: [bootstrap.sh:111](../scripts/legacy/bootstrap.sh#L111), [bootstrap.sh:189](../scripts/legacy/bootstrap.sh#L189). The implementing agent reports 26 modern/legacy sentinel-wrapper checks passing; these checks do not run simulators. |
| The initial tree had no explicit root license or package license metadata. | The owner added [LICENSE:1](../LICENSE#L1), preserving the upstream MIT notice, and [THIRD_PARTY_NOTICES.md:3](../THIRD_PARTY_NOTICES.md#L3), identifying the graph adaptation and separately installed benchmark assets. [pyproject.toml:8](../pyproject.toml#L8) now declares README/license metadata and line 23 includes both notice files in packaging. [CITATION.bib](../CITATION.bib) records the paper citation. This review does not independently audit third-party legal terms. |
| The historical bootstrap silently depended on an author-local Python/Torch base; an arbitrary Python 3.10 override could fail later. | [PORTABLE_LEGACY_SETUP.md:5](PORTABLE_LEGACY_SETUP.md#L5) states the dependency, gives an explicit isolated CPU-Torch recipe at line 12, explains standalone/Conda versus nested-venv inheritance at line 28, and limits native support to Triton. [bootstrap.sh:26](../scripts/legacy/bootstrap.sh#L26) resolves an explicit base or a `python3.10` fallback; lines 51–94 check Python 3.10 and a stable Torch 2.6–2.x before downloads, validate child inheritance, and preserve existing environments. The implementing agent reports 40 mocked wrapper checks passing and a real preflight on existing Python 3.10.19 / Torch 2.6.0+cu126; no generic Linux simulator run is claimed. |
| Install instructions did not specify an inspected dependency set or distinguish wheel and repository contents. | [README.md:26](../README.md#L26) now selects the official CPU Torch 2.6.0 wheel, applies [modern-constraints.txt:1](../requirements/modern-constraints.txt#L1), and links the measured CUDA alternative. README line 31 explicitly distinguishes package contents from repository configs/scripts and labels the constraints as direct dependencies, not a full transitive lock. |

**Open setup and release requirements.**

1. **Native historical execution outside Triton remains unverified.** The limitation is now explicit in [PORTABLE_LEGACY_SETUP.md:62](PORTABLE_LEGACY_SETUP.md#L62). Portable artifact preparation does not supply a generic native runtime: `smoke`, `test`, and `run` still use Triton modules and a Slurm guard. A separately tested container or native-library adapter is needed before advertising general Linux support for this optional track. This does not block the documented modern workstation path.

2. **The final measured environment still needs an archived recreation route.** [pyproject.toml:12](../pyproject.toml#L12) retains useful broad support bounds; the new [modern-constraints.txt:5](../requirements/modern-constraints.txt#L5) pins inspected direct scientific dependencies, including the SMAC protocol version at line 13. The CPU recipe is a different Torch build from the measured Triton CUDA stack, as the README states. These constraints improve repeatability but do not lock every transitive dependency or simulator binary. Preserve a tested final environment specification and downloadable run manifests before claiming exact recreation of the measured environment. Current reports correctly require matching actual comparison versions.

3. **Local evidence links need a public artifact route.** [.gitignore:1](../.gitignore#L1) excludes environments/cache, and lines 11–14 exclude `runs`, `logs`, `reference`, and extracted manuscript sources. These are intentionally external resources. A new checkout cannot open most raw-run links in [STATUS.md](STATUS.md), replay the checkpoint example in [README.md:129](../README.md#L129) before training, or inspect the manuscript paths in [paper_targets.json:4](../configs/paper_targets.json#L4). The checked-in pilot-series JSON files preserve their plotted episode records, but do not replace all checkpoints and manifests. For the release, provide an artifact index/download with raw evaluations, manifests, selected checkpoints, and source evidence, or explicitly label links that are only workspace evidence. The source checkout itself should remain usable without those historical outputs.

4. **Experimental readiness remains open.** The registered requirements in [EXPERIMENT_PLAN.md:31](EXPERIMENT_PLAN.md#L31) and [EXPERIMENT_PLAN.md:56](EXPERIMENT_PLAN.md#L56) require frozen settings, complete budgets, held-out seeds, comparative performance, and mechanism evidence. Current development jobs and smoke checks do not satisfy them. The historical binary stack and observation-radius ambiguity remain declared comparability gaps. These are release blockers for a claim of successful paper reproduction, separate from whether the software is useful as an explicitly labeled development release.


**Subsequent installation evidence, 2026-10-25.** Commit `56394f6` passed a
fresh Python 3.10 CPU installation from an initially empty pip cache in job
**21198078**: 442 tests passed, 32 native-only tests skipped, and `pip check`
passed. Fresh SMAC installation job **21198390** followed the documented
pinned upstream checkout and install commands, then tested an installed wheel
outside the source checkout with real 3s5z training, evaluation, checkpoint
saving, and a separately loaded-checkpoint evaluation. The existing game
binary/maps were reused read-only; installing those assets from scratch and
cross-platform support were not tested. See the [CPU ledger](../reports/fresh-install-validation.json)
and [SMAC ledger](../reports/fresh-smac-install-validation.json). This closes
the missing fresh-dependency-install check for those tested workflows; public
artifact hosting and experimental acceptance remain open.

**Saved-source evaluation compatibility.** The modern command in [README.md:129](../README.md#L129) runs the evaluator from a completed run's saved `source` directory. Historical evaluation must use the same native environment through `DMS_SOURCE_ROOT` and `bootstrap.sh run`, as shown in [HISTORICAL_ENVIRONMENT.md:124](HISTORICAL_ENVIRONMENT.md#L124); merely selecting `.venv/bin/python` or exporting an inherited `PYTHONPATH` does not load those native prerequisites.

Current [evaluate.py:43](../src/dms/evaluate.py#L43) records the environment and dependency versions actually used for evaluation, plus evaluation-source hashes. The new one-million-step HalfCheetah beta-0.1 snapshot contains these fields. The earlier completed 500,000-step beta-0.1 snapshot's evaluator does not. Executing that older saved evaluator remains valid for a descriptive result, but cannot manufacture the newer provenance fields. [report.py:162](../src/dms/report.py#L162) and [report.py:206](../src/dms/report.py#L206) preserve summaries with explicit warnings while blocking the release reference gate when source, environment, or dependency evidence is missing or inconsistent. Do not edit archived source or copy training metadata into a result as if it had been observed during evaluation. Freeze a sufficiently instrumented source version before final runs.

**Optional reader improvements.** Add a small support matrix distinguishing modern workstation, original SMAC with an external game, and Triton-only automated legacy setup. Link the current development ledger from the README so the older 500,000-step example is not mistaken for the selected final method. These are usability improvements; neither should be used to imply that final benchmark acceptance has already passed.

Validation evidence available during this review: full compute job 21196035 reported 230 passed and 22 native-legacy skips; separate native legacy job 21195881 reported 35 passed. Later wrapper changes have the focused evidence above, rather than inheriting coverage from the earlier full-suite snapshot. The new CPU CI workflow has not run on GitHub because the repository is unpublished. No broad tests were rerun for this documentation-only audit.

**Development artifacts, 2026-10-26.** A verified local archive now contains all
seven completed 10M HalfCheetah development runs, including all three matched
training seeds and their weaker outcomes. The archive includes configs, saved
source, training diagnostics, per-episode diagnostic evaluations and checkpoints;
separate 100-episode evaluations remain in the checked-in reports. The
[archive ledger](../reports/development-halfcheetah-10m-artifacts.json) records
its SHA-256, 147 payload members and exact regeneration command. No public URL
has been created, so hosting remains open. The README's support matrix and
current-status link are now present; the earlier optional usability items have
been addressed.
