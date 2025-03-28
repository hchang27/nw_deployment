import time
from collections import defaultdict

import isaacgym
from torchtyping import TensorType

from lucidsim_old import ROBOT_DIR
from lucidsim_old.terraria.textures.hurdle import Hurdle

assert isaacgym

import asyncio
import os
import sys
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
from params_proto import PrefixProto, Proto
from PIL import Image as PImage
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, DefaultScene, Sphere, group

from lucidsim_old.dataset import Dataset
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

    # these are going to be modified depending on the render type
    near = 0.05
    far = 50.0

    key = "ego"
    showFrustum = False
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = True


class HurdleEval(PrefixProto):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "hurdle/scene_99999"

    # ------------ RGB MODELS ------------
    # segmented:
    # teacher_checkpoint = "/alanyu/scratch/2024/02-23/165010/checkpoints/net_0.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-23/165010/checkpoints/net_100.pt"

    # textured:
    teacher_checkpoint = "/alanyu/scratch/2024/02-28/231622/checkpoints/net_0.pt"
    rgb_checkpoint = "/alanyu/scratch/2024/02-28/231622/checkpoints/net_100.pt"

    use_teacher = False
    # which samples to run for this terrain
    sample_range = (0, 5)

    render_type: Literal["rgb", "depth"] = "rgb"
    normalize_depth = True

    serve_root = Proto(env="$HOME")

    use_estimated_states = True
    add_delay = True

    background_color = "0000ff"
    plane_color = "green"
    hurdle_color = "red"
    # sky_color = "blue"

    # ground_texture = None
    # hurdle_texture = None
    # sky_texture = None

    ground_texture = "grass.png"
    hurdle_texture = "bricks.jpeg"
    sky_texture = "snowy.jpeg"

    resize_shape = (45, 80)

    # metadata
    checkpoint_meta = "rgb_frame_diff"

    experiment_log = "/lucid-sim/lucid-sim/analysis/eval/stairs/"

    # for recurrent models, set this to be one 
    stack_size = 10

    compute_deltas = False  # True
    drop_last = False

    imagenet_pipe = True

    trial_max_timesteps_s = 10

    log = False

    record_video = True

    use_smooth_terrain = True

    port = 8017

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)

        if self.render_type in ["depth"]:
            queries["background"] = "000000"
        elif self.render_type in ["rgb"]:
            queries["background"] = "0000ff"
        elif self.render_type in ["object_mask", "obstacle_mask", "marker_mask"]:
            queries["background"] = "000000"
        elif self.render_type == "background_mask":
            queries["background"] = "ffffff"
        else:
            raise ValueError(f"Unknown render type {self.render_type}")

        self.app = Vuer(static_root=self.serve_root, queries=queries, port=self.port, uri=f"ws://localhost:{self.port}")

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.done = False

        if self.render_type == "depth":
            CameraArgs.near = 0.1
            CameraArgs.far = 3
        else:
            CameraArgs.near = 0.05
            CameraArgs.far = 200.0

        if self.imagenet_pipe:
            pipeline = [
                T.Resize(self.resize_shape, interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            pipeline = [
                T.Resize(self.resize_shape, interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=False),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
            ]

        self.transform = T.Compose(pipeline)

        from ml_logger import logger

        self.policy = logger.torch_load(self.rgb_checkpoint)
        if hasattr(self.policy.vision_head, "hidden_states"):
            self.policy.vision_head.hidden_states = None
        self.policy.eval()

        self.teacher_policy = logger.torch_load(self.teacher_checkpoint)
        if hasattr(self.teacher_policy.vision_head, "hidden_states"):
            self.teacher_policy.vision_head.hidden_states = None
        self.teacher_policy.eval()

        # FIXME: this is a hack to get the terrain type
        self.terrain_type = self.dataset_prefix.split("/")[0]

        self.env = self._get_env()

        self.success_count = 0

    def _get_env(self):
        from main_street.config import RunArgs
        from main_street.envs.base.legged_robot import LeggedRobot
        from main_street.envs.base.legged_robot_config import LeggedRobotCfg
        from main_street.utils import task_registry
        LeggedRobotCfg.commands.max_ranges.lin_vel_x = [0.5, 0.8]

        with self.logger.Prefix(os.path.join(self.dataset_prefix, "assets")):
            height_field_raw, = self.logger.load_pkl("height_field_raw.npy")
            LeggedRobotCfg.terrain.heightmap_src = height_field_raw

        # turn off domain rand
        LeggedRobotCfg.terrain.terrain_dict = {"smooth slope": 0.,
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
                                               "parkour_stairs": 0.0}

        LeggedRobotCfg.terrain.terrain_dict[f"parkour_{self.terrain_type}"] = 1.0
        LeggedRobotCfg.terrain.terrain_proportions = list(LeggedRobotCfg.terrain.terrain_dict.values())

        LeggedRobotCfg.env.episode_length_s = self.trial_max_timesteps_s

        LeggedRobotCfg.terrain.num_rows = 1
        LeggedRobotCfg.terrain.num_cols = 1
        LeggedRobotCfg.terrain.terrain_width = 24
        LeggedRobotCfg.terrain.terrain_length = 24
        LeggedRobotCfg.terrain.border_size = 15
        LeggedRobotCfg.terrain.simplify_grid = True  # True
        LeggedRobotCfg.terrain.horizontal_scale = 0.025

        RunArgs.task = "go1"
        RunArgs.seed = 42
        RunArgs.delay = self.add_delay

        env_cfg, train_cfg = task_registry.get_cfgs(RunArgs.task)
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 20
        env_cfg.commands.resampling_time = 20

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

        return env

    def _get_mesh(self):

        if self.use_smooth_terrain:
            hurdle = Hurdle(dataset_prefix=self.dataset_prefix,
                            root=self.root,
                            prefix=self.prefix,
                            serve_root=self.serve_root,
                            )

            if self.render_type == "rgb":
                # whatever works best for OF 
                if self.ground_texture is not None and self.hurdle_texture is not None:
                    ground_material = dict(materialType="standard",
                                           material=dict(
                                               map=f"http://localhost:{self.port}/static/{self.ground_texture}",
                                               mapRepeat=[100, 100]))
                    curb_material = dict(materialType="standard",
                                         material=dict(map=f"http://localhost:{self.port}/static/{self.hurdle_texture}",
                                                       mapRepeat=[0.1, 1]))
                else:
                    curb_material = dict(materialType="standard", material=dict(color=self.hurdle_color))
                    ground_material = dict(materialType="standard", material=dict(color=self.plane_color))
            elif self.render_type == "depth":
                curb_material = dict(materialType="depth", material=dict())
                ground_material = dict(materialType="depth", material=dict())
            elif self.render_type == "obstacle_mask":
                curb_material = dict(materialType="basic", material=dict(color="white"))
                ground_material = dict(materialType="basic", material=dict(color="black"))
            elif self.render_type == "background_mask":
                curb_material = dict(materialType="basic", material=dict(color="black"))
                ground_material = dict(materialType="basic", material=dict(color="white"))
            elif self.render_type == "marker_mask":
                curb_material = dict(materialType="basic", material=dict(color="black"))
                ground_material = dict(materialType="basic", material=dict(color="black"))
            else:
                raise NotImplementedError

            return hurdle(curb_material, ground_material)

        else:
            raise NotImplementedError

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

        sess.update @ [CameraView(**vars(CameraArgs), matrix=mat),
                       Go1(f"http://localhost:{self.port}/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                           joint_values, global_rotation=(r, p, y), position=position[0].tolist())]

    def _set_goals(self, sample_number):
        with self.logger.Prefix(os.path.join(self.dataset_prefix, "assets")):
            goals, = self.logger.load_pkl(f"goals_{sample_number:04d}.pkl")

        # remove the faulty first goal
        goals[1] = goals[2]
        self.env.set_terrain_goal(goals, 0, 0, shift_corners=False)

    async def main(self, sess: VuerSession):
        from ml_logger import logger

        if self.log:
            print(f"Logging to {logger.get_dash_url()}.")

        print("hi")
        mesh = self._get_mesh()
        print("mesh received", mesh)
        raw_children = [CameraView(**vars(CameraArgs))]
        if self.render_type == "rgb":
            if self.sky_texture is not None:
                raw_children.append(Sphere(
                    key="background",
                    args=[50, 32, 32],
                    position=[0, 0, 0],
                    rotation=[np.pi / 2, 0, 0],
                    material=dict(side=2, map=f"http://localhost:{self.port}/static/{self.sky_texture}"),
                    materialType="standard",
                ))

        sess.set @ DefaultScene(
            Go1(f"http://localhost:{self.port}/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                joints={name: 0.0 for name in ISAAC_DOF_NAMES}),
            group(
                mesh,
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

        log_data = defaultdict(list)
        step_counter = 0

        for sample_number in range(*self.sample_range):

            print(f"Running sample {sample_number + 1}")

            rgb_buffer = torch.zeros((self.stack_size, CameraArgs.height, CameraArgs.width, 3), device=self.env.device,
                                     dtype=torch.uint8)

            frames = []

            self._set_goals(sample_number)
            t = 0
            while True:
                # set cam position and render
                robot_state = self.env.root_states
                joint_angles = self.env.dof_pos[0, :]
                self._set_robot_pose(robot_state, joint_angles, sess)

                # prep observations ?? 

                # process img, send to the policy
                await sleep(0.001)
                t = time.perf_counter()
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                print(f"grab_render_{step_counter} took {time.perf_counter() - t:.3f} seconds")
                buff = event.value["frame"]
                img: TensorType["height", "width", "channel"] = np.array(PImage.open(BytesIO(buff)))
                raw_img = img.copy()  # for logging purposes

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

                with torch.no_grad():
                    teacher_actions, _, _ = self.teacher_policy(None, obs, None)
                    if self.drop_last:
                        processed_input = processed_input[:-1]

                    # scandots = obs[..., 53:53 + 132].reshape(1, 11, 12)
                    # scandots_inp = torch.nn.functional.interpolate(scandots.unsqueeze(1), size=(45, 80),
                    #                                                mode="bilinear")
                    # student_actions, _, _ = self.policy(scandots_inp, obs)

                    student_actions, _, _ = self.policy(processed_input.reshape(-1, *self.resize_shape)[None, ...], obs)
                    print("action loss", (teacher_actions - student_actions).norm(2, dim=-1).item())
                    if self.use_teacher:
                        pred_actions = teacher_actions
                    else:
                        pred_actions = student_actions

                        # scandots = obs[..., 53:53+132].reshape(1, 11, 12)
                        # scandots_inp = torch.nn.functional.interpolate(scandots.unsqueeze(1), size=(45, 80), mode="bilinear")

                        # pred_actions, _, _ = self.policy(scandots_inp, obs)

                if self.log:
                    logger.save_image(raw_img, f"depth/frame_{step_counter:04d}.png")
                    log_data["obs"].append(obs.cpu().numpy())
                    log_data["teacher_actions"].append(teacher_actions.cpu().numpy())
                    log_data["student_actions"].append(student_actions.cpu().numpy())
                    logger.save_pkl(log_data, "log_data.pkl")

                # step the environment . 
                obs, _, rews, dones, infos = self.env.step(pred_actions.detach())

                if self.record_video:
                    frame = self.env.render(mode="logger")
                    frames.append(frame)

                reset = dones[0].item() == 1
                step_counter += 1
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
                        logger.print(
                            f"Current success rate {self.success_count / (sample_number - self.sample_range[0] + 1)}")

                    if self.record_video:
                        print(f"saving video to {logger.get_dash_url()}")
                        logger.save_video(frames, f"video_{sample_number}.mp4", fps=50)

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
    eval = HurdleEval()
    eval()
