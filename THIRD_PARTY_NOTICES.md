# Third-party source and benchmark assets

`src/dms/legacy_graph.py` adapts the robot graph definitions and observation ordering from `envs/ma_mujoco/multiagent_mujoco/obsk.py` in [LiZhYun/DMS](https://github.com/LiZhYun/DMS), revision `ed94ab0795d794f8bf43cab54ec4737f33319af3`. The original file SHA-256 is `ae412f152d13be5df4cd0806d0531d2e9c82b2eb8725a2afd2166dc20aa42088`. The upstream repository supplies the MIT license with the notice `Copyright (c) 2020 Tianshou contributors`; that notice and license are retained in the adapted module and root LICENSE. The wording is preserved from upstream and does not identify the paper's authorship.

SMAC, StarCraft II, Gym/Gymnasium, Gymnasium Robotics, MuJoCo, and mujoco-py are separately installed dependencies or benchmark assets. They retain their own licenses and distribution terms. The root license covers this repository's software; it does not relicense those installations or the supplied manuscript. Reference checkouts, simulator binaries, manuscript sources, downloaded dependencies, and trained checkpoints are not bundled in the Python distribution.

The published method is credited in [CITATION.bib](CITATION.bib). Dependency revisions are recorded in [references.json](references.json), and experiment manifests record the packages and benchmark assets actually used.
