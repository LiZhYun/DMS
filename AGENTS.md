# DMS ownership and scientific contract

This directory is owned by the DMS reproduction session. Other Codex sessions
own the sibling `../bpta-agentmixer/`; do not modify it. Do not change workspace
root files or merge shared infrastructure without coordinating ownership.

The manuscript in `paper/source/` is the method source. Legacy code in
`reference/DMS/` is for comparison, not an authoritative implementation.
Keep actor and critic inputs local. Infer peer intentions from locally observed
past actions; future actions are training targets only. Never feed global state
to the default independent-learning method.

Keep code straightforward, document tensor axes, and test scientific invariants.
Performance above the paper is a release target, never a result until measured.
Record environment versions and information-access assumptions. Modern MuJoCo
results are a separate benchmark variant until historical equivalence is shown.

Use the `triton-gpu` skill for Slurm work. Do not run sustained training on login
nodes. Smoke-test before a long run and never cancel another session's jobs.
