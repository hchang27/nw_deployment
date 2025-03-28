from pathlib import Path

import isaacgym

assert isaacgym
import copy
import os
from collections import defaultdict
from copy import deepcopy

import torch

from lucidsim_old.traj_generation.traj_gen import TrajGenerator


class BallTrajGenerator(TrajGenerator):
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"

    # For the ball, this just sets the logging prefix name -- doesn't affect the actual terrain
    terrain_type = "ball"

    dataset_prefix = "debug/scene_00001"

    rollout_range = (0, 20)
    num_steps = 50 * 10

    # checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-02-20/02.57.22/02.57.22/1"
    checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-03-08/14.37.26/14.37.26/1"

    use_estimated_states = True
    add_delay = True

    add_noise_prob = 0.25  # percentage of rollouts to add noise to
    sampling_noise = 0.1  # how much noise to add to the teacher weights when sampling, in std dev 

    seed = 42

    log = True

    def __post_init__(self, **deps):
        """
        Set up the env and runner.
        """
        from ml_logger import ML_Logger

        from main_street.config import RunArgs
        from main_street.envs.go1.ball.ball_config import Go1BallCfg
        from main_street.envs.go1.ball.ball_sampling import BallSampling
        from main_street.utils import task_registry

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        RunArgs._update(deps)

        Go1BallCfg.env.num_envs = 1
        Go1BallCfg.terrain.terrain_width = 16
        Go1BallCfg.terrain.terrain_length = 16
        Go1BallCfg.terrain.border_size = 15

        Go1BallCfg.env.episode_length_s = 2 * self.num_steps / 50
        Go1BallCfg.ball.resampling_time = 2 * self.num_steps / 50

        RunArgs.seed = self.seed

        RunArgs.task = "go1_ball_sampling"
        RunArgs.delay = self.add_delay
        RunArgs.headless = True

        env_cfg, train_cfg = task_registry.get_cfgs(RunArgs.task)
        env_cfg.env.num_envs = 1
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.height = [0.02, 0.02]
        # turn off domain rand
        env_cfg.terrain.terrain_dict = {
            "smooth slope": 0.0,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.0,
            "rough stairs down": 0.0,
            "discrete": 0.0,
            "stepping stones": 0.0,
            "gaps": 0.0,
            "smooth flat": 0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.0,
            "large stairs up": 0.0,
            "large stairs down": 0.0,
            "parkour": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_flat": 1.0,
            "parkour_step": 0.0,
            "parkour_gap": 0.0,
            "demo": 0.0,
        }
        env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = False

        env_cfg.depth.angle = [0, 1]
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_interval_s = 6
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False

        env: BallSampling
        env, _ = task_registry.make_env(name=RunArgs.task, args=RunArgs, env_cfg=env_cfg)

        # load policy
        train_cfg.runner.resume = True

        ppo_runner, train_cfg = task_registry.make_alg_runner(
            load_checkpoint=self.checkpoint, env=env, name=RunArgs.task, args=RunArgs, train_cfg=train_cfg
        )
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        if env.cfg.depth.use_camera:
            depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        else:
            depth_encoder = None

        if env_cfg.env.center_robot:
            x_min, x_max = env.terrain.vertices[:, 0].min(), env.terrain.vertices[:, 0].max()
            y_min, y_max = env.terrain.vertices[:, 1].min(), env.terrain.vertices[:, 1].max()

            width = x_max - x_min
            length = y_max - y_min

            env.env_origins[:] = torch.tensor([x_min + width / 2, y_min + length / 2, 0], device=env.device).repeat(
                env_cfg.env.num_envs, 1)

        actor = ppo_runner.alg.actor_critic.actor
        actor = actor.eval().to(env.device)

        self.env = env
        self.policy = deepcopy(actor)
        self.estimator = copy.deepcopy(estimator)

        self.sampling_policy = deepcopy(actor)
        self.sampling_estimator = deepcopy(estimator)

        self.depth_encoder = deepcopy(depth_encoder)
        self.ppo_runner = ppo_runner

        self.log_info = defaultdict(list)

        self.logging_prefix = os.path.join(self.terrain_type, str(Path(self.dataset_prefix)))

    def append_log(self, obs, teacher_actions):
        super().append_log(obs, teacher_actions)
        self.log_info["ball_location"].append(self.env.root_states[1, :3].cpu().numpy())
        self.log_info["dyaw"].append(obs[0][6].item())

    def set_goals(self, original_goals):
        return


if __name__ == "__main__":
    traj_gen = BallTrajGenerator()
    traj_gen()
