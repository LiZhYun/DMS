# Local development artifacts

## Full-budget HalfCheetah development

The [seven-run 10M bundle](../reports/development-halfcheetah-10m-artifacts.json)
is available locally as
[`dist/dms-halfcheetah-10m-development-20261026.zip`](../dist/dms-halfcheetah-10m-development-20261026.zip).
It contains all IPPO/DMS auxiliary-LR-1e-3 runs for development seeds 101, 103
and 107, plus the selected auxiliary-LR-1e-4 DMS run for seed 101. All seven
fixed final checkpoints are included; weaker outcomes have not been removed.

The verified ZIP has 147 payload files and is 7,230,792 bytes. Its SHA-256 is:

```text
5ca1ce582c8a6ee514b9962d781b1aca33c92d0a37b2206d6180409fe5efb4ce
```

The bundle's episode records are the ten training-time diagnostic games at each
checkpoint. Separate 100-episode endpoint results are retained in the checked-in
[evaluation queue](../reports/development-halfcheetah-10m-evaluation-queue.json)
and [IPPO seed-101 evaluation](../reports/development-halfcheetah-ippo-10m-evaluation.json).
[Current status](STATUS.md) reports the mixed three-seed results and their limits.
The archive ledger records the exact export command and manifest/exporter hashes.
The verifier below also accepts this ZIP by changing its `path` value.
**Neither bundle has a public download URL yet.** Both are development evidence,
not completed paper reproduction.

## Matched local-memory controls

The four completed 3M CUDA HalfCheetah controls (IPPO/DMS, history 0/32) are
packaged in `dist/development-halfcheetah-memory-3m-v1.zip`. The
[memory artifact ledger](../reports/development-halfcheetah-memory-3m-artifacts.json)
records all payload hashes, completion checks, checkpoints and the exact rebuild
command. Both positive and negative matched comparisons are included. Archive
`evaluation.jsonl` files contain the ten-episode diagnostics; separate
100-episode endpoint results remain in [the evaluation ledger](../reports/development-fixed3m-evaluation-queue.json).
This bundle has no public download URL and is development evidence from one
reused training seed.

## SMAC objective controls

All six completed 3M SMAC 3s5z controls (IPPO/DMS × gamma 0.99,
terminal-deadline gamma 0.99, and gamma 0.95) are packaged in
`dist/development-smac-3m-v1.zip`. The [SMAC artifact ledger](../reports/development-smac-3m-artifacts.json)
records the verified archive, all payload hashes and the exact rebuild command.
Every fixed final checkpoint is retained, including the zero-win DMS run.
Archive episode records are the ten-game training diagnostics; all six separate
100-game endpoints remain in [the evaluation ledger](../reports/development-fixed3m-evaluation-queue.json).
This one-training-seed development bundle has no public download URL and does
not contain the StarCraft II installation or maps.

## Earlier 500k pilots

The nine completed historical HalfCheetah 2x3 pilots are packaged locally in
[`dist/development-halfcheetah-500k-v1.zip`](../dist/development-halfcheetah-500k-v1.zip).
The [artifact index](../reports/development-artifact-index.json) records its
SHA-256, byte count, selected runs, checkpoints, and verification evidence.
**This archive has not been uploaded or published; it is currently available only
in this workspace.** These development runs do
not establish paper reproduction or improved benchmark performance.

The ZIP contains 184 payload files plus `ARTIFACT_MANIFEST.json`. Every payload
has a relative path, SHA-256, and byte count in that manifest. Each run entry
records its completion status, 500,000-transition budget, training seed, and five
final evaluation seeds. The archive is 7,518,270 bytes and includes all nine small
`latest.pt` checkpoints. SHA-256:

```text
b2dd2e8284bc037deda3713844029c6178d64b9847815d1bdc6f0bd5562a3b44
```

## Included runs and current evidence

- IPPO: training seeds 101, 103, and 107.
- Original-history DMS soft: seed 101.
- Latent-only DMS soft, KL weight 1: seed 101.
- Latent-only DMS soft, KL weight 0.1: seeds 101, 103, and 107.
- Latent-only DMS soft, KL weight 0.01: seed 101.

The three-seed comparison is preserved, including the negative result:

| Training seed | IPPO mean return | Latent-only DMS, KL 0.1 | DMS minus IPPO |
| --- | ---: | ---: | ---: |
| 101 | 1,928.2011 | 3,109.8340 | +1,181.6329 |
| 103 | 3,114.3355 | 1,612.3823 | −1,501.9532 |
| 107 | 2,708.2924 | 2,239.9707 | −468.3217 |
| Mean over training seeds | 2,583.6097 | 2,320.7290 | −262.8807 |

Each endpoint averages five diagnostic episodes with seeds 10001–10005. Seed
101 was used for development selection; the three seeds are not a confirmatory
test set. Every saved source snapshot matches its own manifest, but source
snapshots differ across some runs, including the seed-101 method pair. The
[paired development report](../reports/historical-halfcheetah-beta01-paired.json)
records those differences. No confidence interval, release reference gate, or
claim of superiority is attached to this export.

## Rebuild and verify

Run from the repository root with the selected completed run directories present.
The exporter uses only the Python standard library; the workspace interpreter is
shown below. It refuses an existing destination, so choose a new filename for
another export instead of overwriting an artifact.

```bash
.venv/bin/python scripts/export_artifacts.py \
  --output dist/development-halfcheetah-500k-v1.zip \
  --include-checkpoints --verify \
  --run runs/dev-legacy-halfcheetah-2x3-ippo-101 \
  --run runs/dev-legacy-halfcheetah-2x3-ippo-103 \
  --run runs/dev-legacy-halfcheetah-2x3-ippo-107 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-soft-101 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-101 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-101 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-103 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-01-107 \
  --run runs/dev-legacy-halfcheetah-2x3-dms-latent-decoder-beta-001-101
```

Every export reads the completed ZIP back and checks all payload hashes and
sizes before making the new destination visible. To verify an existing archive
without exporting again:

```bash
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from export_artifacts import verify_archive
path = Path('dist/development-halfcheetah-500k-v1.zip')
manifest = verify_archive(path)
print(hashlib.sha256(path.read_bytes()).hexdigest())
print(len(manifest['runs']), 'runs;', len(manifest['files']), 'verified payload files')
PY
```

## Export contract

The explicit whitelist is `config.json`, `manifest.json`, `metrics.jsonl`,
`evaluation.jsonl`, and `source/**/*.py`; `--include-checkpoints` adds `latest.pt`.
For historical `ManyAgentSwimmer` 10x2 runs, the exporter also requires exactly
`source/dms/assets/manyagent_swimmer_10x2.xml`. Its bytes must match the training
manifest's `environment.xml_sha256`, and the scenario, partition, and asset name
must agree with the configuration. The archive records this asset hash separately
from the Python source hashes. Missing or changed XML and missing asset metadata
are rejected; other XML files remain outside the whitelist. This support was
added after the nine-run HalfCheetah archive above was created; that archive and
its original execution evidence remain unchanged.
The exporter does not copy environments, caches, private papers, general run
directories, or unrelated files. It never deserializes checkpoint contents.
The content manifest excludes its own hash; the external artifact index hashes
both that manifest and the complete ZIP.

Runs must declare completion, reach their configured budget in both manifest and
metrics, contain the configured final evaluation episodes/seeds, and have exactly
matching declared and saved source hashes. Missing files, symlinked selected
paths, colliding archive names, and changing inputs are rejected. A temporary
archive is validated first, then linked atomically to a new destination; an
existing artifact is never replaced. Per-run integrity checks do not certify
historical binary parity, equivalent source across runs, or scientific success.

The source files and runtime manifests travel with each checkpoint. Environment
setup remains documented in [HISTORICAL_ENVIRONMENT.md](HISTORICAL_ENVIRONMENT.md);
the archive does not bundle simulator binaries or Python environments. It is a
local review artifact while reproduction work continues.

## SMAC development progress snapshot

The [SMAC objective comparison](../reports/smac-objectives-development-20261026.png)
plots win rates and native shaped returns from one preserved, checked-in
[progress snapshot](../reports/development-progress-20261026T110133Z.json).
It contains all six registered continuing-value/terminal-deadline and discount
comparisons, with unequal progress, ten diagnostic games per checkpoint and no
confidence bands. These are development observations from training seed 101.

Regenerate it from the repository without access to ignored run directories:

```bash
.venv/bin/python scripts/plot_smac_development.py \
  --snapshot reports/development-progress-20261026T110133Z.json \
  --output /tmp/dms-smac-progress.png
```

The output path must be unused. The adjacent JSON records the exact source
and figure hashes and the included per-episode outcomes. Rendering job
21211946 completed successfully; the saved figure was visually inspected.
