import isaacgym;
from torchtyping import TensorType

from lucidsim_old import ROBOT_DIR

assert isaacgym

import asyncio
import os
from asyncio import sleep
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.events import GrabRender
from vuer.schemas import Obj, TriMesh, CameraView, DefaultScene, group, Sphere

from lucidsim_old.dataset import Dataset
import open3d as o3d

import torch
import sys

from lucidsim_old.job_queue import JobQueue
from lucidsim_old.utils import ISAAC_DOF_NAMES, euler_from_quaternion, get_three_mat, Go1, quat_rotate
import PIL.Image as PImage
import torchvision.transforms.v2 as T


def process_image(img, device, transform=None):
    data = img.permute(0, 3, 1, 2).contiguous()
    data = data.to(device, non_blocking=True).contiguous()

    if transform is not None:
        data = transform(data)

    return data


class CameraArgs(PrefixProto):
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
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "mit_stairs/stairs_0001_v1"

    # which samples to run for this terrain
    sample_range = (0, 3)
    mesh_src = "textured.obj"
    mesh_material = "textured.mtl"

    render_type: Literal["rgb", "depth"] = "rgb"
    serve_root = Proto(env="$HOME")

    use_estimated_states = True
    add_delay = True

    task: Literal["go1", "go1_flat"] = "go1_flat"  # switch to go1_flat for blind 
    use_camera = False  # turn off if you want to either use teacher or blind policy 

    # checkpoint = "/lucid-sim/lucid-sim/baselines/launch_distill_grav_realsense/2024-01-06/02.17.41/go1/200/"
    # checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-25/22.28.15/22.28.15/1"
    checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-19/23.48.06/go1_flat/200/"
    checkpoint_meta = "blind"
    background_img = "snowy.jpeg"

    # metadata
    experiment_log = "/lucid-sim/lucid-sim/analysis/eval/stairs/"

    trial_max_timesteps = 10 * 50

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        with self.logger.Prefix(self.dataset_prefix):
            gym_tf_pos = self.logger.read_params("gym_tf_pos")["value"]
            self.params = self.logger.read_params("labels")
            labels = self.logger.load_yaml("labels.yaml")

        self.mesh_position = self.params["mesh_position"]
        self.mesh_rotation = self.params["mesh_rotation"]
        self.gym_tf_pos = gym_tf_pos

        self.labels_list = [(label["start"], label["goal"]) for label in labels]

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.done = False

        self.loaded = False

        pipeline = [
            T.Resize((45, 80), interpolation=PImage.BILINEAR),
            T.ToImage(),
            T.ToDtype(torch.float32),
            T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        ]

        self.transform = T.Compose(pipeline)

        self._get_env()

        self.success_count = 0

    def _get_env(self):
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.config import RunArgs
        from main_street.utils import task_registry
        from main_street.envs.base.legged_robot import LeggedRobot
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

    def _get_mesh(self):
        mesh = None
        src_path = os.path.join(self.serve_root, self.local_prefix, self.dataset_prefix, "assets", self.mesh_src)
        mat_path = os.path.join(self.serve_root, self.local_prefix, self.dataset_prefix, "assets", self.mesh_material)

        src_rel_path = os.path.relpath(src_path, self.serve_root)
        mat_rel_path = os.path.relpath(mat_path, self.serve_root)
        if self.render_type == "rgb":

            print(src_rel_path, mat_rel_path)

            mesh = Obj(
                key="terrain",
                src="http://localhost:8012/static/" + src_rel_path,
                mtl="http://localhost:8012/static/" + mat_rel_path,
                position=self.mesh_position,
                rotation=np.deg2rad(self.mesh_rotation),
                onLoad="loaded mesh"
            )
        elif self.render_type == "depth":
            trimesh = o3d.io.read_triangle_mesh(str(src_path))
            mesh = TriMesh(
                key="terrain",
                vertices=np.array(trimesh.vertices),
                faces=np.array(trimesh.triangles),
                position=self.mesh_position,
                rotation=np.deg2rad(self.mesh_rotation),
                materialType="depth",
                onLoad="loaded mesh"
            )

            self.loaded = True

        else:
            raise NotImplementedError

        return mesh

    def _set_robot_pose(self, robot_state, joint_angles, sess: VuerSession):

        joint_values = {name: angle.item() for name, angle in zip(ISAAC_DOF_NAMES, joint_angles)}

        quat_t = robot_state[:, 3:7]
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = [angle.item() for angle in global_rot]

        position = robot_state[:, :3].cpu().numpy()
        cam_position = position + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base], device=self.env.device))[
                           0].cpu().numpy()

        cam_position = cam_position.tolist()

        mat = get_three_mat(cam_position[0], [r, p, y])

        sess.update @ [CameraView(**vars(CameraArgs), matrix=mat),
                       Go1(f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                           joint_values, global_rotation=(r, p, y), position=position[0].tolist())]

    async def main(self, sess: VuerSession):
        from ml_logger import logger

        mesh = self._get_mesh()
        sess.set @ DefaultScene(
            Go1(f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                joints={name: 0.0 for name in ISAAC_DOF_NAMES}),
            group(
                mesh,
                position=self.gym_tf_pos,
            ),
            rawChildren=[
                CameraView(
                    **vars(CameraArgs),
                ),
                Sphere(
                    key="background",
                    args=[50, 32, 32],
                    position=[0, 0, 0],
                    rotation=[np.pi / 2, 0, 0],
                    material=dict(side=2, map=f"http://localhost:8012/static/{self.background_img}"),
                    materialType="standard",
                )
            ],
            up=[0, 0, 1],
        )
        await sleep(2.0)

        while True:
            try:
                await sess.grab_render(key=CameraArgs.key, quality=0.9)
                print("Ready")
                await sleep(0.01)
                break
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        while not self.loaded:
            print("waiting on load")
            await sleep(0.1)

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
                robot_state = self.env.root_states
                joint_angles = self.env.dof_pos[0, :]
                self._set_robot_pose(robot_state, joint_angles, sess)

                # prep observations ?? 

                # process img, send to the policy
                await sleep(0.001)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                buff = event.value["frame"]

                if self.use_camera:
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
                        logger.log_metrics_summary()
                        logger.print(f"Current success rate {self.success_count / (sample_number + 1)}")

                    break

        with logger.Prefix(self.experiment_log):
            print(f"Dashboard: {logger.get_dash_url()}")

        sys.exit()

    async def onLoad(self, event, sess):
        self.loaded = True
        print(f"mesh has been loaded: {event.value}")

    def __call__(self, *args, **kwargs):
        print("starting")
        self.app.add_handler("LOAD", self.onLoad)
        self.app.spawn(self.main)
        self.app.run()


if __name__ == "__main__":
    eval = EvalMesh()
    eval()
