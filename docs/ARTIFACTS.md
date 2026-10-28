# Local development artifacts

## Walker and SMAC conditioning controls

The [verified eight-run archive](../reports/development-conditioning-artifacts.json)
is available locally at `dist/dms-conditioning-development-20261029-v1.zip`. It includes the H100 3M
Walker3x2 and SMAC 8m_vs_9m IPPO/DMS controls, plus all four joint-gradient
and trained no-input trials. Zero-win and weaker results remain included.

The archive is **68,180,285 bytes**, with **208 payload files**.
SHA-256: `15725e45c75b9f1e4c7d9ead8c92ff0dba01118adf7e0554bc2e76e6d88f01df`.
Each run includes its final checkpoint, saved source, configuration and training
diagnostics. The checked-in endpoint ledgers retain the separate 100-episode
evaluations; the archive's evaluation JSONL files are training diagnostics.
These are one-training-seed development results with no public download URL.

## HalfCheetah control-cost audit

Two [verified companion archives](../reports/halfcheetah-control-cost-artifacts.json)
preserve the seven 10M policies and all 700,000 per-step control/reward records
used in the [control-cost audit](CONTROL_COST_AUDIT.md):

| Local archive | Size | Payload files |
| --- | ---: | ---: |
| `dist/dms-halfcheetah-control-cost-policies-20261026-v1.zip` | 7,185,785 bytes | 152 |
| `dist/dms-halfcheetah-control-cost-traces-20261026-v1.zip` | 25,147,024 bytes | 26 |

The policy archive contains the six matched CPU policies and the H100 history-32
baseline, including checkpoints, saved source, configurations and diagnostics.
The trace archive includes the audit code, per-episode records, step arrays,
public diagnostic command, notices and a checksum manifest. Its manifest also
identifies the companion policy archive. Every payload was read back and
verified. These are local development artifacts with no public download URL.
The ledger records both archive SHA-256 values and the policy export command.

## Latest nine-run HalfCheetah bundle

The [nine-run 10M bundle](../reports/development-halfcheetah-10m-artifacts-v2.json)
is available locally at
`dist/dms-halfcheetah-10m-development-20261026-v2.zip`. It includes IPPO and
DMS with auxiliary LR 1e-3/1e-4, each at training seeds 101, 103 and 107.
Both poor lower-LR replications are retained. The earlier seven-run archive
below is also preserved.

The verified archive contains 189 payload files and is 10,084,330 bytes. SHA-256:

```text
5da4991b824786d70ab3e7b2e16caf550d557c2d133c505bda97d2129aab9e58
```

The manifest records every payload, checkpoint and saved source. Fixed100-episode
endpoint records are in the checked-in ledgers; archive `evaluation.jsonl` files
hold the training diagnostics. The bundle has no public download URL and is
three-seed development evidence, not a completed reproduction.

## Earlier seven-run HalfCheetah bundle

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
**These bundles have no public download URLs yet.** They are development evidence,
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

## Coordination thresholds and hardware controls

The six completed 3M HalfCheetah coordination runs are packaged in
`dist/dms-halfcheetah-coordination-development-20261026-v1.zip`: A100 all-peer
and causal 0.6, plus H100 all-peer and causal 0.6/0.2/0.1. Every endpoint is
included. The [artifact ledger](../reports/development-halfcheetah-coordination-artifacts.json)
records the verified 156-payload archive, all checkpoints, saved sources and
rebuild command. It is 7,143,982 bytes; SHA-256:

```text
afc87b3212968ebb1d576635097bf48fc038500136677c3050e1eb6ce72d7029
```

The [episode-distribution figure](../reports/halfcheetah-coordination-development-3m.png)
keeps hardware families separate and shows all 100 fixed-checkpoint evaluation
episodes per policy. Its [data and hashes](../reports/halfcheetah-coordination-development-3m.json)
and [SVG](../reports/halfcheetah-coordination-development-3m.svg) are also retained.
Archive `evaluation.jsonl` files contain training diagnostics; the checked-in
source ledgers hold the separate 100-episode endpoints. This is one-training-seed
development evidence, with no public download URL.

Rebuild the coordination figure from the checked-in ledgers alone:

```bash
.venv/bin/python -m pip install -e '.[plots]'
.venv/bin/python scripts/plot_coordination_endpoints.py --output /tmp/dms-coordination.png
```

The plot extra pins the tested Matplotlib 3.10.8. Choose an unused output path;
the script also writes an SVG and the selected episodes with source hashes.

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
