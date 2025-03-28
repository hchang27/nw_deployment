import time
from asyncio import sleep
from io import BytesIO
from pathlib import Path

from params_proto import ParamsProto, PrefixProto
from typing import List

from vuer.events import ClientEvent
from vuer.schemas import DefaultScene, ImageBackground, Plane, AmbientLight, TimelineControls, group, CameraView
from vuer import Vuer, VuerSession
from vuer.schemas import Scene, Urdf, Movable, DirectionalLight, PointLight

from dataclasses import dataclass

from main_street.config import RunArgs
from main_street.envs import *
from main_street.utils import task_registry
from main_street.utils.helpers import euler_from_quaternion, get_vertical_fov, get_horizontal_fov, \
    sample_camera_frustum_batch

import PIL.Image as PImage

from vuer.serdes import jepg, b64jpg

import torch
from isaacgym import gymapi


from ml_logger import logger

import numpy as np

isaacgym_joint_names = [
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
]

# robots = "/Users/alanyu/urop/parkour/main_street/assets/robots/gabe_go1"
robots = "/home/exx/mit/parkour/main_street/assets/robots/"

app = Vuer(static_root=Path(__file__).parent / robots)


def get_mat(position, rotation=None) -> List:
    # Column-major
    mat = np.array([[1, 0, 0, position[0]],
                    [0, 1, 0, position[1]],
                    [0, 0, 1, position[2]],
                    [0, 0, 0, 1]])

    if rotation is not None:
        rotation = torch.tensor(rotation)[None, ...]
        # convert euler to rot matrix

        # gym_to_3js
        extra_rot = torch.tensor([[0., 0., -1],
                                  [-1, 0, 0.],
                                  [0., 1., 0]])

        # extra_rot = torch.eye(3)

        rot_mat = euler_angles_to_matrix(rotation, convention="XYZ")[0] @ extra_rot
        mat[:3, :3] = rot_mat

        mat_column_major = mat.T

    return mat_column_major.reshape(-1).tolist()


class PlayArgs(PrefixProto):
    logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    exptid = None

    direction_distillation = False
    use_estimated_states = False

    num_steps = 500


class CameraArgs(ParamsProto, cli=False):
    width = 640
    height = 360
    fov = get_vertical_fov(105, width, height)  # vertical
    stream = "frame"
    fps = 30
    near = 2
    far = 5
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2


def update_cam(env: Ball) -> List[float]:
    """
    Reads the environment for the camera transform and provides the 3JS camera matrix
    :param env: 
    :return: 
    """
    tf = env.gym.get_camera_transform(env.sim, env.envs[0], env.cam_handles[-1])

    x, y, z = tf.p.x, tf.p.y, tf.p.z

    roll, pitch, yaw = euler_from_quaternion(torch.tensor([[tf.r.x, tf.r.y, tf.r.z, tf.r.w]]))

    roll = roll.squeeze().cpu().item()
    pitch = pitch.squeeze().cpu().item()
    yaw = yaw.squeeze().cpu().item()
    mat = get_mat([x, y, z], [roll, pitch, yaw])

    return mat


@dataclass
class Controller:
    reset = True
    x_command = 0
    y_command = 0
    z_command = 0

    x_scale = 0.075
    y_scale = 0.075
    z_scale = 0.1

    def __str__(self):
        return (f"x: {self.x_command} \n"
                f"y: {self.y_command} \n"
                f"z: {self.z_command} \n"
                f"reset: {self.reset} \n")


controller = Controller()

# RunArgs.task = "go1"
# RunArgs.delay = True
# RunArgs.use_camera = True

# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-17/17.27.09/go1/300" # normal 58x87
# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0_45x80_realsense_8cm/go1/300/2024-01-04/13.16.20"

# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_local_imu_teacher/go1/500/2024-01-04/21.22.11"

RunArgs.task = "go1_ball"
RunArgs.delay = True
# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher/2024-01-05/12.44.09/go1/600/"

# Flat, grav
Go1BallCfg.env = Go1FlatCfg.env
Go1BallCfgPPO = Go1FlatCfgPPO

PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_flat/2024-01-06/15.26.19/go1_flat/200"

PlayArgs.use_estimated_states = True


def set_env(args):
    from ml_logger import logger

    assert (PlayArgs.logger_prefix is None) ^ (
            PlayArgs.exptid is None), "Either logger_prefix or exptid must be not None"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0.,
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
                                    "parkour_step": 1.0,
                                    "parkour_gap": 0.0,
                                    "demo": 0.0}

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

    depth_latent_buffer = []
    # prepare environment
    env: Ball
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(load_checkpoint=PlayArgs.logger_prefix, env=env,
                                                          name=args.task, args=args,
                                                          train_cfg=train_cfg)
    if env.cfg.depth.use_camera:
        policy = ppo_runner.get_depth_actor_inference_policy(device=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    else:
        depth_encoder = None

    return env, policy, estimator, depth_encoder, ppo_runner


def visualize_samples(samples_to_world, session):
    args = [Urdf(key=f"ball_{i}", position=pos, src="http://localhost:8012/static/ball/urdf/ball.urdf") for i, pos in
            enumerate(samples_to_world.cpu().numpy().tolist())]

    session.upsert @ args


env, policy, estimator, depth_encoder, ppo_runner = set_env(RunArgs)


@app.spawn
async def main(session):
    obs = env.get_observations()
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    heightmap = env.height_samples.cpu().numpy() * env.terrain.cfg.vertical_scale

    # cast to uint8?
    heightmap_n = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    heightmap_uint8 = (heightmap_n * 255).astype(np.uint8)

    width_px = heightmap.shape[0]
    length_px = heightmap.shape[1]

    width = width_px * env.terrain.cfg.horizontal_scale
    length = length_px * env.terrain.cfg.horizontal_scale

    scale = heightmap.max() - heightmap.min()
    shift = heightmap.min()

    border_size = env.terrain.border * env.terrain.cfg.horizontal_scale

    origin_true = np.array([border_size, border_size, 0])
    origin_default = np.array([width / 2, length / 2, 0])

    origin_offset = origin_default - origin_true

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = get_horizontal_fov(CameraArgs.fov, CameraArgs.width, CameraArgs.height)
    camera_props.width = CameraArgs.width
    camera_props.height = CameraArgs.height
    env.attach_camera(0, env.envs[0], env.actor_handles[0], force=True, camera_props=camera_props)

    session.set @ Scene(
        AmbientLight(intensity=0.1),
        Movable(
            PointLight(
                intensity=20
            ),
            position=[width / 4, length / 3, 8],
        ),

        Movable(
            PointLight(
                intensity=20
            ),
            position=[width / 2, length / 3, 8],
        ),

        Movable(
            PointLight(
                intensity=20
            ),
            position=[3 * width / 4, length / 3, 8],
        ),

        Movable(
            DirectionalLight(
                intensity=0.5
            ),
            position=[0, -10, 5]
        ),
        Plane(
            args=[length, width, length_px, width_px],
            key='heightmap',
            materialType="standard",
            material=dict(
                displacementMap=b64jpg(heightmap_uint8),
                displacementScale=scale,
                displacementBias=shift,
            ),
            rotation=[0, 0, np.pi / 2],
            position=origin_offset.tolist()
        ),
        TimelineControls(start=0, end=PlayArgs.num_steps, key="timeline"),
        grid=False,
        up=[0, 0, 1],
    )

    await sleep(0.01)

    actions = torch.zeros((1, env.cfg.env.num_actions)).to(env.device)
    horizontal_fov = get_horizontal_fov(CameraArgs.fov, CameraArgs.width, CameraArgs.height)
    while True:
        tf = env.gym.get_camera_transform(env.sim, env.envs[0], env.cam_handles[-1])

        x, y, z = tf.p.x, tf.p.y, tf.p.z
        ball_dx = controller.x_scale * controller.x_command
        ball_dy = controller.y_scale * controller.y_command
        ball_dz = controller.z_scale * controller.z_command

        sample_x, sample_y, sample_z = sample_camera_frustum_batch(horizontal_fov, CameraArgs.width, CameraArgs.height,
                                                                   CameraArgs.near, CameraArgs.far, num_samples=500)

        sample_x = torch.from_numpy(sample_x).to(env.device)
        sample_y = torch.from_numpy(sample_y).to(env.device)
        sample_z = torch.from_numpy(sample_z).to(env.device)

        samples_to_cam = torch.cat([sample_x, sample_y, sample_z, torch.ones_like(sample_x)], dim=-1).float()

        cam_to_world = torch.eye(4).to(env.device)
        cam_to_world[0, -1] = tf.p.x
        cam_to_world[1, -1] = tf.p.y
        cam_to_world[2, -1] = tf.p.z

        euler = euler_from_quaternion(torch.tensor([[tf.r.x, tf.r.y, tf.r.z, tf.r.w]]))

        roll = euler[0].squeeze().cpu().item()
        pitch = euler[1].squeeze().cpu().item()
        yaw = euler[2].squeeze().cpu().item()

        euler = torch.tensor([[roll, pitch, yaw]]).to(env.device)

        rot = euler_angles_to_matrix(euler, convention="XYZ")[0]

        cam_to_world[:3, :3] = rot

        samples_to_world = (cam_to_world @ samples_to_cam.T).T[:, :3]

        # samples_to_world = (samples_to_cam + torch.tensor([x, y, z, 0]).to(env.device))[:, :3]

        print(samples_to_world)

        visualize_samples(samples_to_world, session)

        # drive relative to the robot

        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]

            if PlayArgs.direction_distillation:
                obs[:, 6:8] = 1.5 * yaw

        else:
            depth_latent = None

        if PlayArgs.use_estimated_states:
            priv_states_estimated = estimator(obs[:, :env.cfg.env.n_proprio])
            obs[:,
            ppo_runner.alg.num_prop + ppo_runner.alg.num_scan: ppo_runner.alg.num_prop + ppo_runner.alg.num_scan + ppo_runner.alg.priv_states_dim] = priv_states_estimated

        # actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach(), ball_dx=ball_dx, ball_dy=ball_dy, ball_dz=ball_dz,
                                              relative=True)

        env.gym.step_graphics(env.sim)

        await sleep(0.01)

        time.sleep(2)

    while True:
        await sleep(0.005)


async def on_gamepad(e: ClientEvent, sess: VuerSession):
    axes, buttons = e.value["axes"], e.value["buttons"]

    controller.x_command = -axes[1]
    controller.y_command = -axes[0]
    controller.z_command = -axes[-1]

    if (buttons[0]):  # b
        controller.reset = True

        env.reset()
        print('resetting env!')
        controller.reset = False


async def step_handler(e: ClientEvent, sess: VuerSession):
    step = e.value["step"]

    dof_angles = {isaacgym_joint_names[i]: angle for i, angle in
                  enumerate(env.dof_pos[0, :].cpu().numpy().tolist())}

    roll = env.roll[0].cpu().item()
    pitch = env.pitch[0].cpu().item()

    yaw = env.yaw[0].cpu().item()

    ball_position = env.root_states[1, :3].cpu().numpy().tolist()

    x, y, z = env.root_states[0, :3].cpu().numpy().tolist()

    mat = update_cam(env)

    sess.upsert @ [
        group(
            group(
                group(
                    Urdf(
                        src="http://localhost:8012/static/gabe_go1/urdf/go1.urdf",
                        jointValues=dof_angles,
                        key="go1",
                    ),
                    rotation=[roll, 0, 0],
                    key="roll",
                ),
                key="pitch",
                rotation=[0, pitch, 0],
            ),
            key="yaw",
            position=[x, y, z],
            rotation=[0, 0, yaw],
        ),
        Urdf(key="ball", position=ball_position, src="http://localhost:8012/static/ball/urdf/ball.urdf"),
        CameraView(**vars(CameraArgs), matrix=mat, )
    ]

    await sleep(0.01)


counter = 0


async def collect_render(event: ClientEvent, sess: VuerSession):
    global counter
    import cv2

    # add you render saving logic here.
    counter += 1
    if counter % 1 == 0:
        value = event.value

        buff = value["frame"]
        pil_image = PImage.open(BytesIO(buff))
        img = np.array(pil_image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


app.add_handler("GAMEPADS", on_gamepad)
app.add_handler("TIMELINE_STEP", step_handler)
app.add_handler("CAMERA_VIEW", collect_render)

app.run()
