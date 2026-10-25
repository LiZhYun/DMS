"""Optional computed-force variant of Gym 0.23.1 Ant-v2.

The native Gym implementation remains untouched. This small subclass computes
the existing force buffer before Gym reads it for observations and reward.
Importing this module does not import Gym or build a native extension.
"""


def make_computed_ant():
    """Construct Ant with explicit post-constraint forces; caller owns timeout."""
    from gym.envs.mujoco.ant import AntEnv
    from mujoco_py import functions

    class ComputedContactAntEnv(AntEnv):
        def do_simulation(self, control, frames):
            super().do_simulation(control, frames)
            # AntEnv.step computes its contact penalty immediately after this
            # call. Computing only in the multi-agent observation adapter would
            # leave reward stale and potentially one transition behind.
            functions.mj_rnePostConstraint(self.sim.model, self.sim.data)

        def reset_model(self):
            super().reset_model()
            functions.mj_rnePostConstraint(self.sim.model, self.sim.data)
            return self._get_obs()

    return ComputedContactAntEnv()
