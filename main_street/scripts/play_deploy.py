from collections import defaultdict

import isaacgym
from params_proto import ParamsProto

assert isaacgym

import torch
from tqdm import trange

from go1_gym_deploy import ParkourActor
from go1_gym_deploy import task_registry as deployment_task_registry
from go1_gym_deploy.utils import class_to_bear
from go1_gym_deploy.utils.deployment_loader import DeploymentLoader
from main_street.config import RunArgs
from main_street.envs import *
from main_street.utils import task_registry

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


def load_deployment(task_name, device="cuda:0"):
    env_cfg, train_cfg = deployment_task_registry.get_cfgs(task_name)

    env_cfg = class_to_bear(env_cfg)
    train_cfg = class_to_bear(train_cfg)

    env_cfg.depth.use_camera = True
    train_cfg.depth_encoder.if_depth = True

    deployment_loader = DeploymentLoader(env_cfg, train_cfg, device)

    deployment_loader.load(PlayArgs.logger_prefix)

    actor, estimator, depth_encoder, depth_actor = deployment_loader.prepare_inference(device)

    parkour_actor = ParkourActor(actor, estimator, depth_encoder, depth_actor, env_cfg.env, device)

    return parkour_actor


class PlayArgs(ParamsProto):
    # logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    logger_prefix = "/lucid-sim/lucid-sim/scripts/train/2024-03-04/00.25.56/00.25.56/1"
    exptid = None

    direction_distillation = False
    use_estimated_states = False


def play(args):
    from ml_logger import logger

    assert (PlayArgs.logger_prefix is None) ^ (
            PlayArgs.exptid is None), "Either logger_prefix or exptid must be not None"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
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

    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True

    # parkour_actor = load_deployment(RunArgs.task, "cuda:0")

    _, _, _, _, parkour_actor, _ = deployment_task_registry.make_agents(
        RunArgs.task, None, "cuda:0", PlayArgs.logger_prefix
    )

    infos = {}

    frames = []
    depths = []

    depth_latent = None

    log_dict = defaultdict(list)

    for i in trange(500):
        obs_simplified = torch.cat(
            (obs[:, : env.cfg.env.n_proprio], obs[:, -env.cfg.env.history_len * env.cfg.env.n_proprio:]), dim=-1)

        if infos.get("depth", None) is not None:
            infos["depth"] = infos["depth"].squeeze(0)

        actions, depth_latent = parkour_actor(obs_simplified[0:1], infos=infos, depth_default=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach())

        if env.cfg.depth.use_camera:
            frame, depth = env.render(mode="logger", visualize_depth=True)
            frames.append(frame)
            depths.append(depth)
        else:
            frame = env.render(mode="logger")
            frames.append(frame)

        # log_dict['obs'].append(obs_simplified.cpu())
        log_dict['actions_unitree'].append(actions.cpu())

        print(
            "time: {:.2f}, cmd vx: {:.2f}, actual vx: {:.2f}, cmd yaw: {:.2f}, actual yaw: {:.2f}".format(
                env.episode_length_buf[env.lookat_id].item() / 50,
                env.commands[env.lookat_id, 0].item(),
                env.base_lin_vel[env.lookat_id, 0].item(),
                env.delta_yaw[env.lookat_id].item(),
                env.base_ang_vel[env.lookat_id, 2].item(),
            )
        )

        id = env.lookat_id

    # print("bag saved to", logger.get_dash_url())

    with logger.Prefix(PlayArgs.logger_prefix):
        logger.remove("video.mp4")
        logger.save_video(frames, "video.mp4", fps=50)

        if env.cfg.depth.use_camera:
            logger.save_video(depths, "depth_new.mp4", fps=50)

        logger.save_pkl(log_dict, "log_dict.pkl")

        print(f"Saved video {logger.get_dash_url()}")


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # PlayArgs.exptid = None
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch/2023-12-14/16.52.16/go1/200" # teacher

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/2023-12-28/20.44.12/scripts/train/distillation/2023-12-28/20.44.12/go1/200"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_flat/2023-12-26/16.16.14/go1_flat/300"

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/2023-12-28/15.34.56/baselines/launch_distill_0_60x96/2023-12-28/15.34.56/go1/200/"

    RunArgs.task = "go1_flat"
    RunArgs.delay = False
    # RunArgs.use_camera = True

    PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/scripts/train/2024-01-19/23.48.06/go1_flat/200/"

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_grav_realsense/2024-01-06/02.17.41/go1/200/"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/scripts/train/2024-03-04/00.25.56/00.25.56/1"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/2023-12-27/17.35.19/baselines/launch_timing/2023-12-27/17.35.19/go1/300"

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/16.05.36/go1/200"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/16.09.37/a1/100"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/17.03.34/a1/200"

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-17/09.34.57/go1/300"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/scripts/train/distillation/2023-12-17/09.45.25/go1/200"

    ### With direction distillation ###
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-17/17.27.09/go1/300"

    # flat
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_flat/2023-12-26/16.16.14/go1_flat/300"

    # RunArgs.task = "go1_flat"
    # RunArgs.use_camera = False  # True

    PlayArgs.use_estimated_states = True
    # PlayArgs.direction_distillation = True

    play(RunArgs)
