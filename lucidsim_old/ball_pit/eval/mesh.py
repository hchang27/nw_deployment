import sys

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
from vuer.schemas import CameraView, DefaultScene, Plane, Sphere, group

from lucidsim_old.utils import ISAAC_DOF_NAMES, Go1, euler_from_quaternion, get_three_mat, quat_rotate


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
    far = 50.0

    key = "ego"
    showFrustum = True
    downsample = 2
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = True


class EvalMesh(PrefixProto):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"

    render_type: Literal["rgb", "depth", None] = "depth"
    # if none, won't grab render

    normalize_depth = True

    serve_root = Proto(env="$HOME")

    use_estimated_states = True
    add_delay = True

    mesh_texture = None  # "sidewalk.jpeg" # "grass.png"

    # metadata
    checkpoint_meta = "rgb_frame_diff"

    experiment_log = "/lucid-sim/lucid-sim/analysis/eval/ball/"
    # rgb_checkpoint = "/alanyu/scratch/2024/01-31/175815/checkpoints/net_1000.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-01/212701/checkpoints/net_1000.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-01/225601/checkpoints/net_1000.pt"
    rgb_checkpoint = "/alanyu/scratch/2024/02-02/165430/checkpoints/net_1000.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-03/040044/checkpoints/net_1000.pt"
    compute_deltas = False
    drop_last = False
    stack_size = 1

    plane_color = "#008080"
    plane_texture = "blue_carpet.jpeg"

    background_img = "stata_openspace.jpg"

    ball_color = "red"
    ball_texture = "soccer_sph_s.png"  # if this is True, it will override the color

    seed = 42

    trial_max_timesteps = 10 * 50
    ball_radius = 0.15

    episode_length_s = 20
    resampling_period_s = 1

    num_samples = 5

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        self.done = False

        pipeline = [
            T.Resize((45, 80), interpolation=PImage.BILINEAR),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        self.transform = T.Compose(pipeline)

        from ml_logger import logger

        self.policy = logger.torch_load(self.rgb_checkpoint)

        self._get_env()

        self.success_count = 0
        self.total_count = 0

    def _get_env(self):
        from ml_logger import ML_Logger

        from main_street.config import RunArgs
        from main_street.envs.go1.ball.ball_config import Go1BallCfg
        from main_street.envs.go1.ball.ball_sampling import BallSampling
        from main_street.utils import task_registry

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        Go1BallCfg.env.num_envs = 1
        Go1BallCfg.terrain.terrain_width = 16
        Go1BallCfg.terrain.terrain_length = 16
        Go1BallCfg.terrain.border_size = 15

        Go1BallCfg.env.episode_length_s = self.episode_length_s
        Go1BallCfg.ball.resampling_time = int(self.resampling_period_s * 50)

        Go1BallCfg.ball.view.horizontal_fov = 90
        Go1BallCfg.ball.view.width = 640
        Go1BallCfg.ball.view.height = 180
        Go1BallCfg.ball.view.near = 0.5
        Go1BallCfg.ball.view.far = 1.0

        Go1BallCfg.commands.max_ranges.lin_vel_x = [0.0, 0.1]

        Go1BallCfg.ball.stopping_distance = 0.4

        RunArgs.seed = self.seed

        RunArgs.task = "go1_ball_sampling"
        RunArgs.delay = self.add_delay
        RunArgs.headless = False

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

        if env_cfg.env.center_robot:
            x_min, x_max = env.terrain.vertices[:, 0].min(), env.terrain.vertices[:, 0].max()
            y_min, y_max = env.terrain.vertices[:, 1].min(), env.terrain.vertices[:, 1].max()

            width = x_max - x_min
            length = y_max - y_min

            env.env_origins[:] = torch.tensor([x_min + width / 2, y_min + length / 2, 0], device=env.device).repeat(
                env_cfg.env.num_envs, 1)

        self.env = env

    def _get_material_kwargs(self, color, texture=None, repeatTexture=False):
        if self.render_type == "depth":
            return dict(materialType="depth")
        elif self.render_type == "rgb":
            if texture:
                mat = dict(materialType="standard", material=dict(
                    map=f"http://localhost:8012/static/{texture}"))
                if repeatTexture:
                    mat["material"]["repeat"] = [1, 1]
                return mat
            else:
                return dict(materialType="standard", material=dict(color=color))
        else:
            return dict()
        #     raise ValueError(f"render_type {self.render_type} not supported")

    def _set_robot_pose(self, robot_state, joint_angles, sess: VuerSession):

        joint_values = {name: angle.item() for name, angle in zip(ISAAC_DOF_NAMES, joint_angles)}

        quat_t = robot_state[0:1, 3:7]
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = (angle.item() for angle in global_rot)

        position = robot_state[:, :3].cpu().numpy()
        cam_position = position + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base], device=self.env.device))[
                           0].cpu().numpy()

        cam_position = cam_position.tolist()

        mat = get_three_mat(cam_position[0], [r, p, y])

        ball_location = robot_state[1:2, :3].cpu().numpy().tolist()

        sess.update @ [CameraView(**vars(CameraArgs), matrix=mat),
                       Sphere(key="ball",
                              position=ball_location[0],
                              args=[self.ball_radius, 20, 20],
                              **self._get_material_kwargs(self.ball_color, self.ball_texture)),
                       Go1(f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                           joint_values, global_rotation=(r, p, y), position=position[0].tolist()
                           )
                       ]

    async def main(self, sess: VuerSession):
        from ml_logger import logger

        print("hi")
        mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color, self.plane_texture))

        material_kwargs = self._get_material_kwargs(self.ball_color, self.ball_texture)

        raw_children = [CameraView(**vars(CameraArgs))]

        if self.render_type == "rgb" and self.background_img is not None:
            raw_children.append(Sphere(
                key="background",
                args=[200, 32, 32],
                position=[0, 0, 0],
                rotation=[np.pi / 2, 0, 0],
                material=dict(side=2, map=f"http://localhost:8012/static/{self.background_img}"),
            ))

        sess.set @ DefaultScene(
            Go1(f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                joints={name: 0.0 for name in ISAAC_DOF_NAMES}),
            group(
                group(
                    mesh,
                ),
                Sphere(
                    key="ball",
                    args=[self.ball_radius, 20, 20],
                    **material_kwargs,
                )
            ),
            rawChildren=raw_children,
            up=[0, 0, 1],
        )
        await sleep(2.0)

        while True:
            try:
                await sess.grab_render(key="DEFAULT", quality=0.9, ttl=10)
                print("Ready")
                await sleep(0.5)
                break
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        for sample_number in range(self.num_samples):

            print(f"Running sample {sample_number + 1}")

            rgb_buffer = torch.zeros((self.stack_size, CameraArgs.height, CameraArgs.width, 3), dtype=torch.uint8,
                                     device=self.env.device)

            obs, _ = self.env.reset()

            trial_done = False

            while not trial_done:
                # set cam position and render
                robot_state = self.env.root_states
                joint_angles = self.env.dof_pos[0, :]
                self._set_robot_pose(robot_state, joint_angles, sess)

                # process img, send to the policy
                await sleep(0.001)
                if self.render_type in ["rgb", "depth"]:
                    event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                    buff = event.value["frame"]
                    img: TensorType["height", "width", "channel"] = np.array(PImage.open(BytesIO(buff)))

                    if self.render_type == "depth" and self.normalize_depth:
                        img = (img - img.min()) / (img.max() - img.min() + 1e-3) * 255
                        img = img.astype(np.uint8)

                    img_t = torch.from_numpy(img).to(self.env.device)

                    # Note: most recent frame should be at the end of the buffer 
                    rgb_buffer = torch.cat([rgb_buffer[1:], img_t[None, ...]], dim=0)
                    processed_input: TensorType["batch", "channel", "height", "width"] = process_image(rgb_buffer,
                                                                                                       self.env.device,
                                                                                                       self.transform)
                    if self.compute_deltas:
                        # compute the difference between the most recent frame and the previous frames
                        processed_input[:-1] = processed_input[-1:] - processed_input[:-1]

                    if self.drop_last:
                        processed_input = processed_input[:-1]

                    processed_input = processed_input.reshape(-1, 45, 80)[None, ...]

                else:
                    processed_input = None

                pred_actions, _, _, yaw = self.policy(processed_input, obs)

                # step the environment . 
                obs, _, rews, dones, infos = self.env.step(pred_actions.detach())

                reset = dones[0].item() == 1
                if reset:
                    self.success_count += self.env.reach_goal_count
                    self.total_count += self.env.sample_count

                    print(f"Trial is complete with success rate {self.env.reach_goal_count / self.env.sample_count}")

                    trial_done = True

        with logger.Prefix(self.experiment_log):
            print(f"Dashboard: {logger.get_dash_url()}")
            logger.print(
                f"{self.checkpoint_meta} || {self.rgb_checkpoint} || {self.success_count / self.total_count}")

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
