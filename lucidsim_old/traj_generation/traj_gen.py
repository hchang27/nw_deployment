"""
Use isaacgym terrains to generate trajectories. Can automatically sample the labels here. 

Note: only one at a time, since we need vary the terrain difficulty
"""
import warnings

import isaacgym

assert isaacgym

import copy
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Literal

import torch
from params_proto import PrefixProto
from tqdm import trange


class TrajGenerator(PrefixProto):
    root = "http://luma01.csail.mit.edu:4000"
    terrain_type: Literal["parkour_flat", "parkour_hurdle", "parkour_gap", "parkour_step"] = "parkour_hurdle"
    prefix = f"lucidsim/scenes/{terrain_type}"

    rollout_range = (0, 20)
    num_steps = 50 * 10

    checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-03-08/14.37.26/14.37.26/1"

    use_estimated_states = True
    add_delay = True

    difficulty = 0.5

    add_noise_prob = 0.5  # percentage of rollouts to add noise to
    sampling_noise = 0.1  # how much noise to add to the teacher weights when sampling, in std dev 

    seed = 42

    log = True
    save_video = True

    def __post_init__(self, **deps):
        """
        Set up the env and runner.
        """
        from ml_logger import ML_Logger

        from main_street.config import RunArgs
        from main_street.envs.base.legged_robot import LeggedRobot
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.utils import task_registry

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)
        self.logger.job_started()
        print(self.logger)
        print("logging to:", self.logger.get_dash_url())
        self.logger.log_text(f"""
        charts:
        """, ".charts.yml", dedent=True)

        RunArgs._update(deps)

        LeggedRobotCfg.commands.max_ranges.lin_vel_x = [0.5, 0.8]

        LeggedRobotCfg.terrain.terrain_dict = {
            "smooth slope": 0.,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.,
            "rough stairs down": 0.,
            "discrete": 0.,
            "stepping stones": 0.0,
            "gaps": 0.,
            "smooth flat": 0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.,
            "large stairs up": 0.,
            "large stairs down": 0.,
            "parkour": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_flat": 0.0,
            "parkour_step": 0.0,
            "parkour_gap": 0.0,
            "demo": 0.0,
            "parkour_stairs": 0.0,
        }

        LeggedRobotCfg.terrain.terrain_dict[self.terrain_type] = 1.0
        LeggedRobotCfg.terrain.terrain_proportions = list(LeggedRobotCfg.terrain.terrain_dict.values())

        LeggedRobotCfg.terrain.num_rows = 1
        LeggedRobotCfg.terrain.num_cols = 1

        LeggedRobotCfg.terrain.terrain_width = 24
        LeggedRobotCfg.terrain.terrain_length = 24
        if self.terrain_type == "stairs":
            LeggedRobotCfg.terrain.terrain_width = 4
            LeggedRobotCfg.terrain.terrain_length = 10

        LeggedRobotCfg.terrain.border_size = 15
        LeggedRobotCfg.terrain.simplify_grid = True
        LeggedRobotCfg.terrain.half_valid_width = [4.0, 5.0]  # to make things a lot wider
        LeggedRobotCfg.terrain.no_roughness = False
        LeggedRobotCfg.terrain.flat_roughness_only = False
        LeggedRobotCfg.terrain.horizontal_scale = 0.025

        LeggedRobotCfg.terrain.max_difficulty = False
        LeggedRobotCfg.terrain.curriculum = False
        LeggedRobotCfg.terrain.difficulty = self.difficulty

        RunArgs.task = "go1"
        RunArgs.seed = self.seed
        RunArgs.delay = self.add_delay
        RunArgs.headless = False

        env_cfg, train_cfg = task_registry.get_cfgs(RunArgs.task)

        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 2 * self.num_steps / 50  # just making sure we don't resample within the rollout
        env_cfg.commands.resampling_time = 2 * self.num_steps / 50

        # turn off domain rand
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = False

        env_cfg.depth.angle = [0, 1]
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_interval_s = 6
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False

        env: LeggedRobot
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

        actor = ppo_runner.alg.actor_critic.actor
        actor = actor.eval().to(env.device)

        self.y_noise = 0.25

        self.env = env
        self.policy = deepcopy(actor)
        self.estimator = deepcopy(estimator)

        self.sampling_policy = deepcopy(actor)
        self.sampling_estimator = deepcopy(estimator)

        self.depth_encoder = deepcopy(depth_encoder)
        self.ppo_runner = ppo_runner

        self.log_info = defaultdict(list)


    def get_sampling_policy(self, policy, noise_amt):
        if noise_amt is not None:
            sampling_policy = copy.deepcopy(policy)
            with torch.no_grad():
                for param in sampling_policy.parameters():
                    std_dev = param.std()
                    noise = torch.randn(param.size(), device=param.device) * std_dev * noise_amt
                    param.add_(noise)
            return sampling_policy, self.estimator
        else:
            return policy, self.estimator

    def handle_step(self, obs, infos):
        if self.env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, : self.env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = self.depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
        else:
            depth_latent = None

        if self.use_estimated_states:
            priv_states_estimated = self.sampling_estimator(obs[:, : self.env.cfg.env.n_proprio])
            obs[
            :,
            self.ppo_runner.alg.num_prop
            + self.ppo_runner.alg.num_scan: self.ppo_runner.alg.num_prop
                                            + self.ppo_runner.alg.num_scan
                                            + self.ppo_runner.alg.priv_states_dim,
            ] = priv_states_estimated

        actions = self.sampling_policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        if self.use_estimated_states:
            teacher_priv_states_estimated = self.estimator(obs[:, : self.env.cfg.env.n_proprio])

            obs[
            :,
            self.ppo_runner.alg.num_prop
            + self.ppo_runner.alg.num_scan: self.ppo_runner.alg.num_prop
                                            + self.ppo_runner.alg.num_scan
                                            + self.ppo_runner.alg.priv_states_dim,
            ] = teacher_priv_states_estimated

        teacher_actions = self.policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        if self.log:
            self.append_log(obs, teacher_actions)

        next_obs, _, rews, dones, infos = self.env.step(actions.detach())

        return next_obs, infos

    def append_log(self, obs, teacher_actions):
        self.log_info["obs"].append(obs.cpu().numpy())
        self.log_info["actions"].append(teacher_actions.detach().cpu().numpy())
        self.log_info["states"].append(self.env.root_states[0].cpu().numpy())
        self.log_info["dofs"].append(self.env.dof_pos[0, :].cpu().numpy().tolist())

    def set_goals(self, original_goals):
        new_goals = self.env.terrain.resample_hurdle_goals(0, 0, original_goals, self.y_noise)
        self.env.set_terrain_goal(new_goals, 0, 0, shift_corners=False)

        return new_goals

    def __call__(self):

        if not self.log:
            warnings.warn("Logging is turned off for this run.")

        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.ppo_runner.device)[:,
                         -1] if self.ppo_runner.if_depth else None

        original_goals = self.env.terrain.goals[0, 0].copy()
        for rollout_id in trange(*self.rollout_range):
            new_goals = self.set_goals(original_goals)
            obs, _ = self.env.reset()

            frames = []

            use_bad_teacher = random.random() < self.add_noise_prob
            if use_bad_teacher:
                print("Using bad policy to sample trajectories!")
                noise_amount = self.sampling_noise

                self.sampling_policy, self.sampling_estimator = self.get_sampling_policy(self.policy, noise_amount)
            else:
                self.sampling_policy = self.policy
                self.sampling_estimator = self.estimator

            if self.save_video:
                frame = self.env.render(mode="logger")
                frames.append(frame)

            for step in range(self.num_steps):
                obs, infos = self.handle_step(obs, infos)

                if self.save_video:
                    frame = self.env.render(mode="logger")
                    frames.append(frame)

            # save and clear log info
            if self.log:
                if use_bad_teacher:
                    path = f"trajectories/videos/noisy_{rollout_id:04d}.mp4"
                else:
                    path = f"trajectories/videos/teacher_{rollout_id:04d}.mp4"
                if frames:
                    self.logger.save_video(frames, path, fps=50)
                    self.logger.log_text(f"""
                    - type: video
                      glob: {path}
                    """, ".charts.yml", dedent=True)
                    frames.clear()

                self.logger.save_pkl(self.log_info, f"trajectories/trajectory_{rollout_id:04d}.pkl")
                if new_goals is not None:
                    self.logger.save_pkl(new_goals, f"assets/goals_{rollout_id:04d}.pkl")

                self.log_info = defaultdict(list)

        # save the terrain
        if self.log:
            with self.logger.Prefix("assets"):
                # self.logger.save_pkl(self.env.terrain.vertices, "vertices.npy")
                # self.logger.save_pkl(self.env.terrain.triangles, "triangles.npy")
                self.logger.log_params(terrain=dict(border_size=self.env.terrain.cfg.border_size))
                self.logger.save_pkl(self.env.terrain.height_field_raw, "height_field_raw.npy")
                self.logger.save_pkl(self.env.terrain.smooth_height_field_raw, "smooth_height_field_raw.npy")
                self.logger.save_pkl(original_goals, "original_goals.npy")

        print("Done!")


if __name__ == "__main__":
    traj_gen = TrajGenerator(
        dataset_prefix="step",
        terrain_type="parkour_step",
        rollout_range=[0, 2],
    )
    traj_gen()
    # import gc
    #
    # Returns the number of
    # objects it has collected
    # and deallocated
    # "parkour_hurdle": 0.0,
    # "parkour_flat": 0.0,
    # "parkour_step": 0.0,
    # "parkour_gap": 0.0,
    # "demo": 0.0,
    # "parkour_stairs": 0.0,
    # for ttype in [ "flat", "gap", "step", "hurdle"]:
    #     traj_gen = TrajGenerator(
    #         dataset_prefix=ttype,
    #         terrain_type=f"parkour_{ttype}",
    #         rollout_range=[0, 2],
    #     )
    #     traj_gen()
    #     del traj_gen
    #     gc.collect()

    # traj_gen = TrajGenerator()
    # traj_gen = TrajGenerator()
