import isaacgym
from torchtyping import TensorType

from lucidsim_old import ROBOT_DIR

assert isaacgym

import asyncio
import os
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import PIL.Image as PImage
import torch
import torchvision.transforms.v2 as T
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, DefaultScene, Sphere, TriMesh, group

from lucidsim_old.dataset import Dataset
from lucidsim_old.utils import ISAAC_DOF_NAMES, Go1, euler_from_quaternion, get_three_mat, quat_rotate


def process_image(img, device, transform=None):
    data = img.permute(0, 3, 1, 2).contiguous()
    data = data.to(device, non_blocking=True).contiguous()

    if transform is not None:
        data = transform(data)

    return data


class CameraArgs(PrefixProto):
    """
    This is the evaluation script that does inference online.
    """
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
    # this is important. should default to False but currently defaults to True
    renderDepth = False
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False
    monitor = True

    serve_root = Proto(env="$PWD")
    port = 8015


class Eval(PrefixProto):
    """
    Evaluate the robot with stacked RGB images
    """
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "hurdle/scene_00001"

    sample_number = 0

    render_type: Literal["rgb", "depth"] = "rgb"
    save_video: bool = False

    port = 8015
    serve_root = Proto(env="$PWD")

    use_estimated_states = True
    add_delay = True

    # rgb_checkpoint = "/alanyu/scratch/2024/01-27/233931/checkpoints/net_1000.pt"
    rgb_checkpoint = "/alanyu/scratch/2024/01-30/160107/checkpoints/net_1000.pt"
    background_img = "assets/snowy.jpeg"

    stack_size = 3
    compute_deltas = True

    num_steps = 500

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries, port=self.port)

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.done = False

        pipeline = [
            T.Resize((45, 80), interpolation=PImage.BILINEAR),
            T.ToImage(),
            T.ToDtype(torch.float32),
            T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        ]

        self.transform = T.Compose(pipeline)

        from ml_logger import logger

        self.policy = logger.torch_load(self.rgb_checkpoint)

        # FIXME: this is a hack to get the terrain type
        self.terrain_type = self.dataset_prefix.split("/")[0]

        self._get_env()

    def _get_env(self):
        from main_street.config import RunArgs
        from main_street.envs.base.legged_robot import LeggedRobot
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.utils import task_registry
        LeggedRobotCfg.commands.max_ranges.lin_vel_x = [0.5, 0.8]

        with self.logger.Prefix(os.path.join(self.dataset_prefix, "assets")):
            height_field_raw, = self.logger.load_pkl("height_field_raw.npy")
            LeggedRobotCfg.terrain.heightmap_src = height_field_raw

            self.goals, = self.logger.load_pkl(f"goals_{self.sample_number:04d}.pkl")

        LeggedRobotCfg.terrain.num_rows = 1
        LeggedRobotCfg.terrain.num_cols = 1
        LeggedRobotCfg.terrain.terrain_width = 24
        LeggedRobotCfg.terrain.terrain_length = 24
        LeggedRobotCfg.terrain.border_size = 15
        LeggedRobotCfg.terrain.simplify_grid = False  # True
        LeggedRobotCfg.terrain.horizontal_scale = 0.025

        RunArgs.task = "go1"
        RunArgs.delay = self.add_delay

        env_cfg, train_cfg = task_registry.get_cfgs(RunArgs.task)
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 20
        env_cfg.commands.resampling_time = 20

        # turn off domain rand
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = False
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
            "stairs": 0.0
        }

        LeggedRobotCfg.terrain.terrain_dict[f"parkour_{self.terrain_type}"] = 1.0
        LeggedRobotCfg.terrain.terrain_proportions = list(LeggedRobotCfg.terrain.terrain_dict.values())

        env_cfg.depth.angle = [0, 1]
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_interval_s = 6
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False

        env: LeggedRobot
        self.env, _ = task_registry.make_env(name=RunArgs.task, args=RunArgs, env_cfg=env_cfg)

    def _get_mesh(self):
        mesh = None

        vertices = self.env.terrain.vertices
        triangles = self.env.terrain.triangles

        mesh = TriMesh(
            key="terrain",
            vertices=vertices,
            faces=triangles,
            position=[-self.env.terrain.cfg.border_size, -self.env.terrain.cfg.border_size, 0],
            materialType="standard",
            onLoad="loaded mesh",
        )

        return mesh

    def _set_robot_pose(self, robot_state, joint_angles, sess: VuerSession):

        joint_values = {name: angle.item() for name, angle in zip(ISAAC_DOF_NAMES, joint_angles)}

        quat_t = robot_state[:, 3:7]
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = (angle.item() for angle in global_rot)

        position = robot_state[:, :3].cpu().numpy()
        cam_position = position + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base], device=self.env.device))[
                           0].cpu().numpy()

        cam_position = cam_position.tolist()

        mat = get_three_mat(cam_position[0], [r, p, y])

        sess.update @ [
            CameraView(**vars(CameraArgs), matrix=mat),
            Go1(f"http://localhost:{self.port}/static/{ROBOT_DIR}/gabe_go1/urdf/go1.urdf",
                joint_values, global_rotation=(r, p, y), position=position[0].tolist())
        ]

    async def main(self, sess: VuerSession):
        mesh = self._get_mesh()
        sess.set @ DefaultScene(
            Go1(f"http://localhost:{self.port}/static/{ROBOT_DIR}/gabe_go1/urdf/go1.urdf",
                joints={name: 0.0 for name in ISAAC_DOF_NAMES}),
            group(
                mesh,
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
                    material=dict(side=2, map=f"http://localhost:{self.port}/static/{self.background_img}"),
                    materialType="standard",
                )
            ],
            up=[0, 0, 1],
        )
        await sleep(2.0)

        while True:
            try:
                await sess.grab_render(key=CameraArgs.key, quality=0.9, ttl=10)
                print("Ready")
                await sleep(0.01)
                break
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        rgb_buffer = torch.zeros((self.stack_size, CameraArgs.height, CameraArgs.width, 3), device=self.env.device)

        self.env.set_terrain_goal(self.goals, 0, 0, shift_corners=False)
        obs, _ = self.env.reset()

        for step in range(self.num_steps):
            # set cam position and render
            robot_state = self.env.root_states
            joint_angles = self.env.dof_pos[0, :]
            self._set_robot_pose(robot_state, joint_angles, sess)

            # process img, send to the policy
            await sleep(0.001)
            event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
            buff = event.value["frame"]
            img: TensorType["height", "width", "channel"] = np.array(PImage.open(BytesIO(buff)))

            img_t = torch.from_numpy(img).to(self.env.device)
            # shift
            # Note: most recent frame should be at the end of the buffer 
            rgb_buffer = torch.cat([rgb_buffer[1:], img_t[None, ...]], dim=0)

            processed_input: TensorType["batch", "channel", "height", "width"] = process_image(
                rgb_buffer, self.env.device, self.transform)

            # img_t = torch.rand_like(img_t, device=img_t.device)

            if self.compute_deltas:
                # compute the difference between the most recent frame and the previous frames
                processed_input[:-1] = processed_input[-1:] - processed_input[:-1]

            pred_actions, _ = self.policy(processed_input.reshape(-1, 45, 80)[None, ...], obs)

            # step the environment . 
            obs, _, rews, dones, infos = self.env.step(pred_actions.detach())

    async def onLoad(self, event, sess):
        self.loaded = True
        print(f"mesh has been loaded: {event.value}")

    def __call__(self, *args, **kwargs):
        print("starting")
        self.app.add_handler("LOAD", self.onLoad)
        self.app.spawn(self.main)
        self.app.run()


if __name__ == "__main__":
    eval = Eval()
    eval()
