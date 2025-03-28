import isaacgym;
from params_proto import ParamsProto

assert isaacgym
import faulthandler
import os

import torch

from main_street.envs import *
from main_street.utils import task_registry, webviewer
from main_street.config import RunArgs


class PlayArgs(ParamsProto):
    logger_prefix = "/alanyu/scratch/2023/12-24/182025/"


def play(args):
    from ml_logger import logger

    with logger.Prefix(PlayArgs.logger_prefix):
        bag, = logger.load_pkl("dof_log.pkl")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
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
                                    "parkour_flat": 1.0,
                                    "parkour_step": 0.0,
                                    "parkour_gap": 0.0,
                                    "demo": 0.0}

    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

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

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    frames = []
    depths = []
    for i in range(len(bag)):

        actions = torch.tensor(bag[i], device=env.device, requires_grad=False).unsqueeze(0).repeat(env.num_envs, 1)

        obs, _, rews, dones, infos = env.step(actions.detach())

        if env.cfg.depth.use_camera:
            frame, depth = env.render(mode="logger", visualize_depth=True)
            frames.append(frame)
            depths.append(depth)
        else:
            frame = env.render(mode="logger")
            frames.append(frame)

        print("time: {:.2f}, cmd vx: {:.2f}, actual vx: {:.2f}, cmd yaw: {:.2f}, actual yaw: {:.2f}".format(
            env.episode_length_buf[env.lookat_id].item() / 50,
            env.commands[env.lookat_id, 0].item(),
            env.base_lin_vel[env.lookat_id, 0].item(),
            env.delta_yaw[env.lookat_id].item(),
            env.base_ang_vel[env.lookat_id, 2].item()))

        id = env.lookat_id

    if not args.web and PlayArgs.logger_prefix is not None:
        with logger.Prefix(PlayArgs.logger_prefix):
            logger.save_video(frames, "video.mp4", fps=50)

            if env.cfg.depth.use_camera:
                logger.save_video(depths, "depth.mp4", fps=50)

            print(f"Saved video {logger.get_dash_url()}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # PlayArgs.exptid = None
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch/2023-12-14/16.52.16/go1/200"
    RunArgs.task = "go1_flat"
    RunArgs.delay = True
    RunArgs.use_camera = False

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/16.05.36/go1/200"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/16.09.37/a1/100"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-16/17.03.34/a1/200"

    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-17/09.34.57/go1/300"
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/scripts/train/distillation/2023-12-17/09.45.25/go1/200"

    ### With direction distillation ###
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_distill_0/2023-12-17/17.27.09/go1/300"

    # flat
    # PlayArgs.logger_prefix = "/alanyu/scratch/2023/12-24/030709/"
    PlayArgs.logger_prefix = "/alanyu/scratch/2023/12-26/182025/"

    # RunArgs.task = "go1_flat"
    RunArgs.delay = True
    # RunArgs.use_camera = False  # True

    PlayArgs.use_estimated_states = True
    # PlayArgs.direction_distillation = True

    play(RunArgs)
