"""Fixed 10x2 ManyAgentSwimmer physics from the original DMS environment.

The asset is generated once and packaged; constructing an environment never
writes into the package and does not require Jinja2. Native dependencies are
imported only by make_swimmer(), inside an allocated legacy execution.

Adapted from LiZhYun/DMS, commit ed94ab0795d794f8bf43cab54ec4737f33319af3,
envs/ma_mujoco/multiagent_mujoco/manyagent_swimmer.py and its XML template.

MIT License, Copyright (c) 2020 Tianshou contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
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

from pathlib import Path

import numpy as np


ASSET_PATH = Path(__file__).parent / "assets" / "manyagent_swimmer_10x2.xml"
ORIGINAL_SOURCE_SHA256 = "c3e52d46867b4bed3063e0b572ba15060d105456a53c7d75d13edff45b835781"
ORIGINAL_TEMPLATE_SHA256 = "e258a36037bd7dca12e563ed9c0986a6239b6394ea9ff76edaf49f65ebeb9831"


def make_swimmer():
    """Construct the native 20-actuator swimmer; the adapter owns its timeout."""
    from gym import utils
    from gym.envs.mujoco.mujoco_env import MujocoEnv

    class ManyAgentSwimmerEnv(MujocoEnv, utils.EzPickle):
        def __init__(self):
            MujocoEnv.__init__(self, str(ASSET_PATH.resolve()), 4)
            utils.EzPickle.__init__(self)

        def step(self, action):
            before = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            forward = (self.sim.data.qpos[0] - before) / self.dt
            control = -0.0001 * np.square(action).sum()
            return self._get_obs(), forward + control, False, {
                "reward_fwd": forward, "reward_ctrl": control,
            }

        def _get_obs(self):
            return np.concatenate([self.sim.data.qpos[2:], self.sim.data.qvel])

        def reset_model(self):
            self.set_state(
                self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(-0.1, 0.1, size=self.model.nv),
            )
            return self._get_obs()

    return ManyAgentSwimmerEnv()
