from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from isaacgym import gymapi
from params_proto import ParamsProto

from main_street.envs.go1.ball.ball import Ball
from main_street.utils import task_registry
from main_street.utils.helpers import get_vertical_fov, get_horizontal_fov

from tqdm import trange
from main_street.config import RunArgs
from main_street.envs.go1.ball.ball_config import Go1BallCfg
import torch

"""
Note: script is only for blind sampling

Directory structure:
/: 
	rollout_001/ // 
		sample_001/
			ball_location.npy

			obs_0001.npy // observations, pre-policy [1, num_obs]
			actions_0001.npy // actions, pre-step [1, 12]
			state_0001.npy // root state, [1, 13]

			obs_0002.npy
			actions_0002.npy
		sample_002/
			...
"""

LOG_INFO = []
SAMPLE_INFO = defaultdict(list)


class SampleArgs(ParamsProto, cli=False):
    logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    exptid = None

    direction_distillation = False
    use_estimated_states = False

    num_rollouts = 1

    render_video = False

    dataset_prefix = "/lucid-sim/lucid-sim/datasets/ball/debug/00003/rollouts/"


def set_env(args):
    from ml_logger import logger

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = 1
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
    env: Ball
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        load_checkpoint=SampleArgs.logger_prefix, env=env, name=args.task, args=args, train_cfg=train_cfg
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

    if env_cfg.env.center_robot:
        x_min, x_max = env.terrain.vertices[:, 0].min(), env.terrain.vertices[:, 0].max()
        y_min, y_max = env.terrain.vertices[:, 1].min(), env.terrain.vertices[:, 1].max()

        width = x_max - x_min
        length = y_max - y_min

        env.env_origins[:] = torch.tensor([x_min + width / 2, y_min + length / 2, 0], device=env.device).repeat(
            env_cfg.env.num_envs, 1)

    # env.reset()
    return env, policy, estimator, depth_encoder, ppo_runner


def play(env, policy, estimator, depth_encoder, ppo_runner):
    global LOG_INFO, SAMPLE_INFO

    from ml_logger import logger

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = Go1BallCfg.ball.view.horizontal_fov
    camera_props.width = Go1BallCfg.ball.view.width
    camera_props.height = Go1BallCfg.ball.view.height
    env.attach_camera(0, env.envs[0], env.actor_handles[0], force=True, camera_props=camera_props)

    for rollout in trange(SampleArgs.num_rollouts, desc="rollouts"):
        frames = []
        egos = []
        obs, _ = env.reset()
        for timestep in trange(Go1BallCfg.env.episode_length_s * 50, desc=f"rollout_{rollout:04d}"):
            sample_num = timestep // Go1BallCfg.ball.resampling_time

            if timestep % Go1BallCfg.ball.resampling_time == 0 and timestep > 0:
                LOG_INFO.append(deepcopy(SAMPLE_INFO))
                SAMPLE_INFO = defaultdict(list)

            depth_latent = None

            if SampleArgs.use_estimated_states:
                priv_states_estimated = estimator(obs[:, : env.cfg.env.n_proprio])
                obs[
                :,
                ppo_runner.alg.num_prop
                + ppo_runner.alg.num_scan: ppo_runner.alg.num_prop
                                           + ppo_runner.alg.num_scan
                                           + ppo_runner.alg.priv_states_dim,
                ] = priv_states_estimated

            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

            robot_state = env.root_states[0].cpu().numpy()

            if timestep % Go1BallCfg.ball.resampling_time == 5:
                ball_location = env.root_states[1, :3].cpu().numpy()
                SAMPLE_INFO["ball_location"] = ball_location.astype(np.half)
            SAMPLE_INFO["obs"].append(obs.cpu().numpy().astype(np.half))
            SAMPLE_INFO["actions"].append(actions.detach().cpu().numpy().astype(np.half))
            SAMPLE_INFO["states"].append(robot_state.astype(np.half))
            SAMPLE_INFO["dofs"].append(env.dof_pos[0, :].cpu().numpy().tolist())

            obs, _, rews, dones, infos = env.step(actions.detach())

            if SampleArgs.render_video:
                frame, ego = env.render(mode="logger")
                frames.append(frame)
                egos.append(ego)

            env.gym.step_graphics(env.sim)

        logger.save_pkl(LOG_INFO, f"rollout_{rollout:04d}.pkl")
        if SampleArgs.render_video:
            logger.save_video(frames, f"frames_{rollout:04d}.mp4", fps=50)
            logger.save_video(egos, f"ego_{rollout:04d}.mp4", fps=50)
        LOG_INFO = []

    print(f"Saved to {logger.get_dash_url()}! 🛌")


def main(args):
    env, policy, estimator, depth_encoder, ppo_runner = set_env(args)
    play(env, policy, estimator, depth_encoder, ppo_runner)


if __name__ == "__main__":
    from ml_logger.job import instr, RUN
    from main_street.envs.base.legged_robot_config import LeggedRobotCfg

    LeggedRobotCfg.terrain.terrain_width = 16
    LeggedRobotCfg.terrain.terrain_length = 16

    RunArgs.task = "go1_ball_sampling"
    RunArgs.delay = True

    Go1BallCfg.env.episode_length_s = 20
    Go1BallCfg.ball.resampling_time = 50 * 0.25

    RunArgs.seed = 50

    SampleArgs.use_estimated_states = True
    SampleArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher/2024-01-05/12.44.09/go1/600/"
    SampleArgs.num_rollouts = 1

    SampleArgs.dataset_prefix = "/lucid-sim/lucid-sim/datasets/ball/debug/00005/rollouts/"

    SampleArgs.render_video = True

    RUN.prefix = SampleArgs.dataset_prefix

    thunk = instr(main, RunArgs)
    thunk()
