import isaacgym
assert isaacgym
import copy
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from params_proto import PrefixProto, Proto
from tqdm import trange


class TrajGenerator(PrefixProto):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/experiments/real2real")

    dataset_prefix = "hurdle/scene_00006"

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes/experiments/real2real"

    checkpoint = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"
    # checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-21/17.29.34/17.29.34/1/"

    use_estimated_states = True
    add_delay = True

    add_noise_prob = 0.25  # percentage of rollouts to add noise to
    sampling_noise = 0.1  # how much noise to add to the teacher weights when sampling, in std dev 

    seed = 42

    log = False

    def __post_init__(self, deps):
        """
        Set up the env and runner.
        """
        from ml_logger import ML_Logger

        from main_street.config import RunArgs
        from main_street.envs.base.legged_robot import LeggedRobot
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.utils import task_registry

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        RunArgs._update(deps)

        with self.logger.Prefix(self.dataset_prefix):
            self.params = self.logger.read_params("labels")
            self.labels = self.logger.load_yaml("labels.yaml")

        LeggedRobotCfg.commands.max_ranges.lin_vel_x = [0.5, 0.8]

        LeggedRobotCfg.terrain.mesh_src = os.path.join(self.local_prefix, self.dataset_prefix, "assets",
                                                       self.params["mesh_src"])
        LeggedRobotCfg.terrain.mesh_tf_pos = self.params["mesh_position"]
        LeggedRobotCfg.terrain.mesh_tf_rot = self.params["mesh_rotation"]
        LeggedRobotCfg.terrain.num_goals = 2

        LeggedRobotCfg.terrain.mesh_from_heightmap = False
        LeggedRobotCfg.terrain.n_sample_pcd = 1_000_000
        LeggedRobotCfg.terrain.horizontal_scale = 0.025

        LeggedRobotCfg.terrain.num_rows = 1
        LeggedRobotCfg.terrain.num_cols = 1
        LeggedRobotCfg.terrain.terrain_width = 24
        LeggedRobotCfg.terrain.terrain_length = 24
        LeggedRobotCfg.terrain.border_size = 15
        LeggedRobotCfg.terrain.simplify_grid = False  # True

        RunArgs.task = "go1"
        RunArgs.seed = self.seed
        RunArgs.delay = self.add_delay
        RunArgs.headless = True

        env_cfg, train_cfg = task_registry.get_cfgs(RunArgs.task)
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 20
        env_cfg.commands.resampling_time = 20

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
        if env.cfg.depth.use_camera:
            policy = ppo_runner.get_depth_actor_inference_policy(device=env.device)
        else:
            policy = ppo_runner.get_inference_policy(device=env.device)

        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        if env.cfg.depth.use_camera:
            depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        else:
            depth_encoder = None

        actor = ppo_runner.alg.actor_critic.actor
        actor = actor.eval().to(env.device)

        self.y_noise = 0.25

        self.env = env
        self.sampling_policy = self.policy = actor
        self.estimator = estimator
        self.depth_encoder = depth_encoder
        self.ppo_runner = ppo_runner

        self.log_info = defaultdict(list)

        with self.logger.Prefix(self.dataset_prefix):
            self.gym_tf_pos = (-self.env.terrain.corner_shift).tolist() + [0]
            self.logger.log_params(gym_tf_pos=dict(value=self.gym_tf_pos))
            
    def get_sampling_policy(self, policy, noise_amt):
        if noise_amt is not None:
            sampling_policy = copy.deepcopy(policy)
            with torch.no_grad():
                for param in sampling_policy.parameters():
                    std_dev = param.std()
                    noise = torch.randn(param.size(), device=param.device) * std_dev * noise_amt
                    param.add_(noise)
            return sampling_policy
        else:
            return policy

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
            priv_states_estimated = self.estimator(obs[:, : self.env.cfg.env.n_proprio])
            obs[
            :,
            self.ppo_runner.alg.num_prop
            + self.ppo_runner.alg.num_scan: self.ppo_runner.alg.num_prop
                                            + self.ppo_runner.alg.num_scan
                                            + self.ppo_runner.alg.priv_states_dim,
            ] = priv_states_estimated

        actions = self.sampling_policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        teacher_actions = self.policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        robot_state = self.env.root_states[0].cpu().numpy()

        if self.log:
            self.log_info["obs"].append(obs.cpu().numpy())
            self.log_info["actions"].append(teacher_actions.detach().cpu().numpy())
            self.log_info["states"].append(robot_state)
            self.log_info["dofs"].append(self.env.dof_pos[0, :].cpu().numpy().tolist())

        next_obs, _, rews, dones, infos = self.env.step(actions.detach())

        return next_obs, infos

    def __call__(self):

        num_labels = len(self.labels)
        labels_list = [(marker["start"], marker["goal"]) for marker in self.labels]

        # Set in terrain
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.ppo_runner.device)[:,
                         -1] if self.ppo_runner.if_depth else None

        sample_done = False
        for sample_num in trange(num_labels, desc="sample_num"):
            self.env.set_terrain_goal(labels_list[sample_num], 0, 0, shift_corners=True)
            obs, _ = self.env.reset()

            noise_amount = None
            if random.random() < self.add_noise_prob:
                print("Using bad policy to sample trajectories!")
                noise_amount = self.sampling_noise

            self.sampling_policy = self.get_sampling_policy(self.policy, noise_amount)

            while not sample_done:
                obs, infos = self.handle_step(obs, infos)
                sample_done = self.env.time_out_buf[0]  # includes reaching goal

            sample_done = False

            # save and clear log info
            if self.log:
                with self.logger.Prefix(str(Path(self.dataset_prefix) / "trajectories")):
                    self.logger.save_pkl(self.log_info, f"trajectory_{sample_num:04d}.pkl")
            self.log_info = defaultdict(list)


if __name__ == "__main__":
    traj_gen = TrajGenerator()
    traj_gen()
