# DMS: coordination through inferred intentions

A readable implementation of **Coordination as Inference in Multi-Agent
Reinforcement Learning**, published in *Neural Networks* (2024).
[Citation](CITATION.bib) · [Original paper code](https://github.com/LiZhYun/DMS/tree/legacy-paper-code)

Each agent infers teammates’ intentions from locally observed past actions,
learns to predict their future behavior, and conditions its own policy on
relevant inferred intentions. The actor and critic use local information.
Future actions enter auxiliary training only.

This is a redesign preserving the method’s core ideas, with explicit
[implementation differences](docs/IMPLEMENTATION_DECISIONS.md). It is not a
claim of exact historical training replication or performance above the paper.
The release focuses on clear code, faithful information flow and runnable examples;
full-budget runs and the complete experiment matrix are not required.
[Validation and status](docs/STATUS.md).

## Start here

Use Python 3.10. The synthetic example needs no external simulator.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m dms.train --config configs/toy_smoke.json \
  --device cpu --output runs/my-first-dms
```

This runs 512 transitions and saves configuration, source, environment metadata,
training metrics, evaluation episodes and a checkpoint. Use a new output directory
for each run. The toy task checks execution; its score is not benchmark evidence.

Compare the same task with IPPO or exercise policy-KL peer selection:

```bash
.venv/bin/python -m dms.train --config configs/toy_smoke.json \
  --algorithm ippo --device cpu --output runs/my-first-ippo
.venv/bin/python -m dms.train --config configs/toy_causal_smoke.json \
  --device cpu --output runs/my-first-causal-dms
```

The causal smoke computes the selection rule immediately, without the benchmark
warmup. A short smoke can select no peers; it does not establish a coordination gain.
Evaluate the saved DMS policy without updating it:

```bash
PYTHONPATH=runs/my-first-dms/source .venv/bin/python -m dms.evaluate \
  --checkpoint runs/my-first-dms/latest.pt --episodes 10 --seed 20001 \
  --device cpu --output runs/my-first-dms/checkpoint-evaluation.json
```

## Follow one training iteration

[The walkthrough](docs/METHOD_WALKTHROUGH.md) explains the tensor axes, equations
and collect → PPO → intention-learning loop. Read these files in order:

| File | Responsibility |
| --- | --- |
| [envs.py](src/dms/envs.py) | Local observations, observed action sensors, visibility masks and the toy task |
| [intention.py](src/dms/intention.py) | Past-only prior, training-only posterior, decoder and masked variational loss |
| [policy.py](src/dms/policy.py) | Independent actor/critic, soft attention and policy-KL peer selection |
| [returns.py](src/dms/returns.py) | Advantages with distinct terminal and timeout handling |
| [train.py](src/dms/train.py) | Configuration, collection, PPO, auxiliary updates and checkpointing |

The three coordination choices are `soft` (attention), `all` (all available
peers), and `causal` (the paper’s policy-sensitivity criterion). The last uses
`KL(full policy || policy without peer)`; it does not identify a causal effect
on return. Hidden actions stay masked in every variant. The default learning
base is independent PPO; [sequential updates](docs/SEQUENTIAL_POLICY_UPDATE.md)
provide the disclosed HAPPO-style alternative.

## Checks and benchmarks

```bash
.venv/bin/python -m pytest tests -q
```

Optional simulator checks skip when their dependencies are unavailable.
For the tested modern MuJoCo setup, install
`.venv/bin/python -m pip install -c requirements/modern-constraints.txt -e '.[test,mujoco]'`.
The pinned setup, separate benchmark tracks and scheduler commands are in:

- [Original SMAC setup](docs/SMAC_ENVIRONMENT.md): official SMAC, maps and a separately installed StarCraft II 4.10.
- [Historical MuJoCo setup](docs/PORTABLE_LEGACY_SETUP.md) and [containers](containers/README.md): v2 compatibility backend; historical binary parity is qualified.
- [Extended reader reference](docs/READER_REFERENCE.md): CUDA installation, Triton, config generation, optional research variants and result aggregation.
- [Validation and scope](docs/STATUS.md): tested behavior, limitations and preserved development evidence.

On Triton, submit simulation and tests to compute nodes; for example,
`sbatch scripts/triton/smoke.sbatch configs/toy_smoke.json runs/triton-toy-101 101`
after creating `logs/`. Use a compatible PyTorch CUDA build for GPU jobs.

Optional policy memory, residual intention inputs, joint prior training and
auxiliary-first updates are experimental extensions. They are disabled in the
introductory configs. Their presence does not imply a measured improvement.

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
Simulator binaries and private manuscript sources are not distributed.
