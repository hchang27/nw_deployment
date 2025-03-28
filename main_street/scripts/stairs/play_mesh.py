from asyncio import sleep
from pathlib import Path

from params_proto import ParamsProto
from vuer.events import ClientEvent
from vuer.schemas import DefaultScene, ImageBackground, Plane, AmbientLight, TimelineControls, group, TriMesh
from vuer import Vuer, VuerSession
from vuer.schemas import Scene, Urdf, Movable, DirectionalLight, PointLight

from dataclasses import dataclass

from main_street.config import RunArgs
from main_street.envs import *
from main_street.envs.base.legged_robot_config import LeggedRobotCfg
from main_street.utils import task_registry
from main_street.utils.helpers import extract_local_roll_pitch

from vuer.serdes import jpg, b64jpg

import torch

from ml_logger import logger

import numpy as np

isaacgym_joint_names = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# robots = "/Users/alanyu/urop/parkour/main_street/assets/robots/gabe_go1"
robots = "/home/exx/mit/parkour/main_street/assets/robots/gabe_go1"

app = Vuer(static_root=Path(__file__).parent / robots, port=8013, uri="ws://localhost:8013")


class PlayArgs(ParamsProto):
    logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    exptid = None

    direction_distillation = False
    use_estimated_states = False

    num_steps = 500


@dataclass
class Controller:
    reset = True
    x_command = 0
    y_command = 0

    def __str__(self):
        return f"x: {self.x_command} \ny: {self.y_command} \nreset: {self.reset} \n"

# LeggedRobotCfg.terrain.mesh_src = "/home/exx/stairs_v1/textured.obj"
# LeggedRobotCfg.terrain.mesh_tf_pos = [0, 0, 5]
# LeggedRobotCfg.terrain.mesh_tf_rot = [90, 0, 0]

# LeggedRobotCfg.terrain.mesh_src = "/home/exx/red_stairs.ply"
# LeggedRobotCfg.terrain.mesh_tf_rot = [242, 21, 5]

LeggedRobotCfg.terrain.mesh_src = "/home/exx/red_stairs_v2/textured.obj"
LeggedRobotCfg.terrain.mesh_tf_pos = [0, 0, 3.5]
LeggedRobotCfg.terrain.mesh_tf_rot = [90, 0, 0]
LeggedRobotCfg.terrain.num_rows = 5
LeggedRobotCfg.terrain.num_cols = 2
LeggedRobotCfg.terrain.terrain_width = 16
LeggedRobotCfg.terrain.terrain_length = 16
LeggedRobotCfg.terrain.border_size = 15
LeggedRobotCfg.terrain.simplify_grid = False  # True

controller = Controller()

# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_flat/2024-01-05/12.59.08/go1_flat/200"
#
RunArgs.task = "go1"
PlayArgs.use_estimated_states = True
RunArgs.delay = True
# PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher/2024-01-05/12.40.55/go1/300"
PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"


def set_env(args):
    from ml_logger import logger

    assert (PlayArgs.logger_prefix is None) ^ (PlayArgs.exptid is None), "Either logger_prefix or exptid must be not None"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.height = [0.02, 0.02]
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
        "parkour_flat": 0.0,
        "parkour_step": 1.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
        "stairs": 0.0,
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

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        load_checkpoint=PlayArgs.logger_prefix, env=env, name=args.task, args=args, train_cfg=train_cfg
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

    return env, policy, estimator, depth_encoder, ppo_runner


env, policy, estimator, depth_encoder, ppo_runner = set_env(RunArgs)


@app.spawn
async def main(session):
    obs = env.get_observations()
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    border_size = env.terrain.border * env.terrain.cfg.horizontal_scale

    length = env.terrain.cfg.horizontal_scale * env.terrain.tot_cols
    width = env.terrain.cfg.horizontal_scale * env.terrain.tot_rows

    print("length", length)
    print("width", width)

    origin_offset = np.array([-border_size, -border_size, 0])

    session.set @ Scene(
        AmbientLight(intensity=0.1),
        Movable(
            PointLight(intensity=20),
            position=[width / 4, length / 3, 8],
        ),
        Movable(
            PointLight(intensity=20),
            position=[width / 2, length / 3, 8],
        ),
        Movable(
            PointLight(intensity=20),
            position=[3 * width / 4, length / 3, 8],
        ),
        Movable(DirectionalLight(intensity=0.5), position=[0, -10, 5]),
        TriMesh(
            key="heightmap",
            vertices=env.terrain.vertices,
            faces=env.terrain.triangles,
            position=origin_offset.tolist(),
            material=dict(wireframe=True),
        ),
        TimelineControls(start=0, end=PlayArgs.num_steps, key="timeline"),
        up=[0, 0, 1],
        grid=False,
    )

    await sleep(0.01)

    frames = []

    # for _ in range(500):
    while True:
        # print(controller)
        target_speed = np.clip(controller.x_command, 0, 1.5)
        target_yaw = controller.y_command

        # target_speed, target_yaw = None, None

        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, : env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]

            if PlayArgs.direction_distillation:
                obs[:, 6:8] = 1.5 * yaw

        else:
            depth_latent = None

        if PlayArgs.use_estimated_states:
            priv_states_estimated = estimator(obs[:, : env.cfg.env.n_proprio])
            obs[
                :,
                ppo_runner.alg.num_prop
                + ppo_runner.alg.num_scan : ppo_runner.alg.num_prop
                + ppo_runner.alg.num_scan
                + ppo_runner.alg.priv_states_dim,
            ] = priv_states_estimated

        actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach(), dx=target_speed, dyaw=target_yaw)

        # if env.cfg.depth.use_camera:
        #     frame, depth = env.render(mode="logger", visualize_depth=True)
        # else:
        #     frame = env.render(mode="logger")

        # print("time: {:.2f}, cmd vx: {:.2f}, actual vx: {:.2f}, cmd yaw: {:.2f}, actual yaw: {:.2f}".format(
        #     env.episode_length_buf[env.lookat_id].item() / 50,
        #     env.commands[env.lookat_id, 0].item(),
        #     env.base_lin_vel[env.lookat_id, 0].item(),
        #     env.delta_yaw[env.lookat_id].item(),
        #     env.base_ang_vel[env.lookat_id, 2].item()))

        await sleep(0.02)

        # frames.append(frame)

    # logger.save_video(frames, "video.mp4", fps=50)
    # print(logger.get_dash_url())

    while True:
        await sleep(0.005)


async def on_gamepad(e: ClientEvent, sess: VuerSession):
    axes, buttons = e.value["axes"], e.value["buttons"]

    controller.x_command = 2 * -axes[1]
    controller.y_command = -axes[2]

    if buttons[0]:  # b
        controller.reset = True

        env.reset()
        print("resetting env!")
        controller.reset = False


async def step_handler(e: ClientEvent, sess: VuerSession):
    step = e.value["step"]

    dof_angles = {isaacgym_joint_names[i]: angle for i, angle in enumerate(env.dof_pos[0, :].cpu().numpy().tolist())}

    roll = env.roll[0].cpu().item()
    pitch = env.pitch[0].cpu().item()

    yaw = env.yaw[0].cpu().item()
    # print("yaw", yaw)

    x, y, z = env.root_states[0, :3].cpu().numpy().tolist()

    sess.upsert @ group(
        group(
            group(
                Urdf(
                    src="http://localhost:8013/static/urdf/go1.urdf",
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
    )


app.add_handler("GAMEPADS", on_gamepad)
app.add_handler("TIMELINE_STEP", step_handler)

app.run()
