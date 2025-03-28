import isaacgym

assert isaacgym

import os
from typing import Literal

import numpy as np
from params_proto import PrefixProto, Proto

from lucidsim_old.dataset import Dataset


def process_image(img, device, transform=None):
    data = img.permute(0, 3, 1, 2).contiguous()
    data = data.to(device, non_blocking=True).contiguous()

    if transform is not None:
        data = transform(data)

    return data


class CameraArgs(PrefixProto, cli=False):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.05
    far = 2000.0  # 15.0 # 7.0 
    key = "ego"
    showFrustum = False
    downsample = 2
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False
    monitor = True


class EvalMesh(PrefixProto):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/experiments/real2real")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes/experiments/real2real"
    dataset_prefix = "stairs/scene_00002"

    # which samples to run for this terrain
    sample_range = None
    mesh_src = "textured.obj"
    mesh_material = "textured.mtl"

    use_estimated_states = True
    add_delay = True

    task: Literal["go1", "go1_flat"] = "go1_flat"  # switch to go1_flat for blind 
    use_camera = False  # turn off if you want to either use teacher or blind policy 

    # checkpoint = "/lucid-sim/lucid-sim/baselines/launch_distill_grav_realsense/2024-01-06/02.17.41/go1/200/"
    # checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-25/22.28.15/22.28.15/1"
    checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-19/23.48.06/go1_flat/200/"
    checkpoint_meta = "blind"
    background_img = "desert.jpeg"

    # metadata
    experiment_log = "/lucid-sim/lucid-sim/experiments/eval/stairs/"

    trial_max_timesteps = 10 * 50

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        print('use camera', self.use_camera)
        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        with self.logger.Prefix(self.dataset_prefix):
            self.params = self.logger.read_params("labels")

            labels = self.logger.load_yaml("labels.yaml")

        self.labels_list = [(label["start"], label["goal"]) for label in labels]

        if self.sample_range is None:
            self.sample_range = (0, len(self.labels_list))

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.done = False

        self._get_env()

        self.success_count = 0
        

    def _get_env(self):
        from main_street.config import RunArgs
        from main_street.envs.base.legged_robot import LeggedRobot
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.utils import task_registry
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

        RunArgs.task = self.task
        RunArgs.use_camera = self.use_camera
        RunArgs.resume = True
        RunArgs.delay = self.add_delay
        RunArgs.headless = not self.use_camera

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
        self.env, _ = task_registry.make_env(name=RunArgs.task, args=RunArgs, env_cfg=env_cfg)

        self.ppo_runner, train_cfg = task_registry.make_alg_runner(load_checkpoint=self.checkpoint, env=self.env,
                                                                   name=RunArgs.task, args=RunArgs,
                                                                   train_cfg=train_cfg, )
        if self.use_camera:
            self.policy = self.ppo_runner.get_depth_actor_inference_policy(device=self.env.device)
        else:
            self.policy = self.ppo_runner.get_inference_policy(device=self.env.device)
        self.estimator = self.ppo_runner.get_estimator_inference_policy(device=self.env.device)

        if self.use_camera:
            self.depth_encoder = self.ppo_runner.get_depth_encoder_inference_policy(device=self.env.device)

        self.total_progress = 0

    def __call__(self):
        from ml_logger import logger

        for sample_number in range(*self.sample_range):

            print(f"Running sample {sample_number + 1}")

            self.env.set_terrain_goal(self.labels_list[sample_number], 0, 0, shift_corners=True)
            obs, _ = self.env.reset()

            infos = {}
            if self.use_camera:
                infos["depth"] = self.env.depth_buffer.clone().to(self.env.device)[:, -1]
            else:
                infos["depth"] = None

            depth_latent = None

            trial_done = False

            while not trial_done:
                # set cam position and render
                if self.use_camera and infos['depth'] is not None:
                    obs_student = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                else:
                    depth_latent = None

                if self.use_estimated_states:
                    priv_states_estimated = self.estimator(obs[:, :self.env.cfg.env.n_proprio])
                    obs[:,
                    self.ppo_runner.alg.num_prop + self.ppo_runner.alg.num_scan: self.ppo_runner.alg.num_prop + self.ppo_runner.alg.num_scan + self.ppo_runner.alg.priv_states_dim] = priv_states_estimated

                actions = self.policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

                # step the environment . 
                obs, _, rews, dones, infos = self.env.step(actions.detach())

                reset = dones[0].item() == 1
                if reset:
                    success = 0
                    # check what type of reset 
                    if self.env.reach_goal_cutoff[0].item() == 1:
                        print("Reached goal")
                        success = 1
                        self.success_count += 1
                    elif self.env.time_out_buf[0].item() == 1:
                        print("Time out")
                    else:
                        print("Failed ")

                    starting_xy = self.env.env_goals[0, 0, :2].cpu().numpy()
                    target_xy = self.env.env_goals[0, -1, :2].cpu().numpy()
                    current_xy = self.env.last_root_pos[0, :2].cpu().numpy()

                    goal_dist = np.linalg.norm(target_xy - starting_xy)
                    current_dist = np.dot(target_xy - starting_xy, current_xy - starting_xy) / goal_dist

                    if self.env.reach_goal_cutoff[0].item() == 1:
                        progress = 1.0
                    else:
                        progress = min(current_dist / goal_dist, 1.0)

                    with logger.Prefix(self.experiment_log):
                        logger.log_metrics(
                            {self.checkpoint_meta: success, f"{self.checkpoint_meta}_progress": progress})
                        self.total_progress += progress
                        # logger.log_metrics_summary()
                        # logger.print(f"Current success rate {self.success_count / (sample_number + 1)}")

                    break

        with logger.Prefix(self.experiment_log):
            print(f"Dashboard: {logger.get_dash_url()}")
            logger.print(
                f"{self.checkpoint_meta} || {self.checkpoint} || {self.dataset_prefix} || {self.success_count / (self.sample_range[1] - self.sample_range[0])} || {self.total_progress / (self.sample_range[1] - self.sample_range[0])}")


if __name__ == "__main__":
    eval = EvalMesh()
    eval()
