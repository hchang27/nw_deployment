"""
Same as traj_gen, but sample from a set of checkpoints instead of adding noise. 
"""
import os
from copy import deepcopy

import isaacgym

assert isaacgym
import random


def diverse_trajectory_factory(cls):
    class DiverseTrajWrapper(cls):

        def __post_init__(self, **deps):
            from ml_logger import logger
            super().__post_init__(**deps)

            # grab checks
            with logger.Prefix(self.checkpoint):
                checkpoint_list = logger.glob("checkpoints/*.pt")

            stamped_timestamps = [path for path in checkpoint_list if "last" not in path]
            checkpoint_timestamps = [int(x.split("/")[-1].split(".")[0].split('_')[-1]) for x in stamped_timestamps]

            self.checkpoint_paths = {t: path for t, path in zip(checkpoint_timestamps, stamped_timestamps)}
            self.checkpoint_keys = sorted(list(self.checkpoint_paths.keys()))

        def get_sampling_policy(self, *args, **kwargs):
            """
            Sample from the set of checkpoints according to the sampling function (uniform for now)
            """
            key = random.choice(self.checkpoint_keys)
            print(f'Sampling new sampling policy for this rollout with key {key}...')

            logger_ckpt = os.path.join(self.checkpoint, f"checkpoints/model_{key}.pt")

            # we'll only use the ppo runner for loading, not for stepping the environment. 
            # however, we just need to be careful to deepcopy things properly. 
            self.ppo_runner.load(logger_ckpt, load_from_logger=True)
            self.sampling_policy = deepcopy(self.ppo_runner.alg.actor_critic.actor)
            self.sampling_policy.eval().to(self.env.device)

            self.sampling_estimator = deepcopy(self.ppo_runner.get_estimator_inference_policy(device=self.env.device))

            return self.sampling_policy, self.sampling_estimator

    return DiverseTrajWrapper


if __name__ == '__main__':
    from lucidsim_old.ball_pit.ball_traj_gen import BallTrajGenerator
    from lucidsim_old.traj_generation.traj_gen import TrajGenerator

    # don't optimize imports
    assert TrajGenerator and BallTrajGenerator

    traj = diverse_trajectory_factory(BallTrajGenerator)()
    traj()
