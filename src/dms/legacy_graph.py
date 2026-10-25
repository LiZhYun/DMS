"""The paper robot graphs and their original local observation ordering.

Adapted from LiZhYun/DMS, commit ed94ab0795d794f8bf43cab54ec4737f33319af3,
envs/ma_mujoco/multiagent_mujoco/obsk.py (SHA256 below). This small adaptation
keeps the paper task definitions and removes unrelated robots and runner code.

Original repository license: MIT License, Copyright (c) 2020 Tianshou contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ORIGINAL_GRAPH_COMMIT = "ed94ab0795d794f8bf43cab54ec4737f33319af3"
ORIGINAL_GRAPH_SHA256 = "ae412f152d13be5df4cd0806d0531d2e9c82b2eb8725a2afd2166dc20aa42088"


@dataclass(frozen=True)
class Joint:
    label: str
    qpos_ids: int
    qvel_ids: int
    act_ids: int
    bodies: tuple[int, ...] = ()


def graph_for(scenario: str, agent_conf: str) -> tuple[list[tuple[Joint, ...]], list[frozenset[Joint]]]:
    """Return original ordered partitions and undirected joint hyperedges."""
    if scenario == "ManyAgentSwimmer":
        if agent_conf != "10x2":
            raise ValueError("unsupported historical partition: ManyAgentSwimmer requires '10x2'")
        nodes = [Joint(f"rot{index}", index - 20, index - 20, index) for index in range(20)]
        edges = [frozenset(nodes[index:index + 2]) for index in range(19)]
        return [tuple(nodes[index:index + 2]) for index in range(0, 20, 2)], edges
    if scenario == "Ant":
        nodes = []
        for leg in range(4):
            limb_body = 2 + 3 * leg
            nodes.extend((
                Joint(f"hip{leg + 1}", -8 + 2 * leg, -8 + 2 * leg, (2 + 2 * leg) % 8,
                      (1, limb_body)),
                Joint(f"ankle{leg + 1}", -7 + 2 * leg, -7 + 2 * leg, (3 + 2 * leg) % 8,
                      (limb_body, limb_body + 1, limb_body + 2)),
            ))
        edges = [frozenset(nodes[start:start + 2]) for start in range(0, 8, 2)]
        edges.append(frozenset(nodes[::2]))
        if agent_conf == "2x4d":
            return [tuple(nodes[i] for i in (0, 1, 4, 5)), tuple(nodes[i] for i in (2, 3, 6, 7))], edges
        sizes = {"2x4": 4, "4x2": 2, "8x1": 1}
    elif scenario in {"HalfCheetah", "Walker2d"}:
        labels = (
            ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
            if scenario == "HalfCheetah"
            else ("thigh_joint", "leg_joint", "foot_joint", "thigh_left_joint", "leg_left_joint", "foot_left_joint")
        )
        native = [Joint(label, index - 6, index - 6, index) for index, label in enumerate(labels)]
        edges = [frozenset((native[left], native[right])) for left, right in ((2, 1), (1, 0), (0, 3), (3, 4), (4, 5))]
        # This order is intentional: the middle 3x2 agent spans the two legs.
        nodes = [native[index] for index in (2, 1, 0, 5, 4, 3)]
        sizes = {"2x3": 3, "3x2": 2, "6x1": 1}
    else:
        raise ValueError(f"unsupported historical scenario: {scenario!r}")
    if agent_conf not in sizes:
        raise ValueError(f"unsupported historical partition {agent_conf!r} for {scenario}")
    size = sizes[agent_conf]
    return [tuple(nodes[start:start + size]) for start in range(0, len(nodes), size)], edges


def distance_layers(
    partition: tuple[Joint, ...], edges: list[frozenset[Joint]], radius: int
) -> dict[int, list[Joint]]:
    """Original breadth-first layers, with joints sorted by label within each."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    frontier = set(partition)
    seen: set[Joint] = set()
    layers = {}
    for distance in range(radius + 1):
        layers[distance] = sorted(frontier, key=lambda joint: joint.label)
        seen.update(frontier)
        adjacent = set().union(*(edge for edge in edges if edge & frontier))
        frontier = adjacent - seen
    return layers


def observation_categories(scenario: str, radius: int) -> list[tuple[str, ...]]:
    """Original defaults: full own-joint features, only qpos beyond distance 0."""
    own = ("qpos", "qvel", "cfrc_ext") if scenario == "Ant" else ("qpos", "qvel")
    return [own] + [("qpos",)] * radius


def local_observation(
    data: Any, layers: dict[int, list[Joint]], categories: list[tuple[str, ...]]
) -> np.ndarray:
    """Read local native data in original distance/joint/category order.

    Ant contact forces are clipped and each body is included once per observer.
    Walker joint velocities remain unclipped, matching original graph features
    (which differed from the single-agent Gym observation).
    """
    values: list[float] = []
    seen_bodies: set[int] = set()
    for distance, joints in sorted(layers.items()):
        for joint in joints:
            for category in categories[distance]:
                if category == "qpos":
                    values.append(float(data.qpos[joint.qpos_ids]))
                elif category == "qvel":
                    values.append(float(data.qvel[joint.qvel_ids]))
                elif category == "cfrc_ext":
                    for body in joint.bodies:
                        if body not in seen_bodies:
                            values.extend(np.clip(data.cfrc_ext[body], -1.0, 1.0).tolist())
                            seen_bodies.add(body)
                else:
                    raise ValueError(f"unsupported historical observation category: {category}")
    return np.asarray(values, dtype=np.float32)
