"""
Autocollect data from rollouts according to the provided labels (from label_goals.py).
"""
from collections import defaultdict
from pathlib import Path

from params_proto import ParamsProto
from tqdm import trange

from main_street.config import RunArgs
from main_street.envs import *
from main_street.envs.base.legged_robot_config import LeggedRobotCfg
from main_street.utils import task_registry

import numpy as np

ROLLOUT_COUNT = 0
LOG_INFO = defaultdict(list)


class PlayArgs(ParamsProto):
    logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    dataset_prefix = "/lucid-sim/lucid-sim/datasets/stairs/debug/00001/"

    direction_distillation = False
    use_estimated_states = False


def set_env(args):
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


def entrypoint():
    global LOG_INFO

    from ml_logger import logger

    with logger.Prefix(PlayArgs.dataset_prefix):
        params = logger.read_params("labels")
        labels = logger.load_yaml("labels.yaml")

    LeggedRobotCfg.terrain.mesh_src = params["mesh_src"]
    LeggedRobotCfg.terrain.mesh_tf_pos = params["mesh_tf_pos"]
    LeggedRobotCfg.terrain.mesh_tf_rot = params["mesh_tf_rot"]
    LeggedRobotCfg.terrain.num_goals = 2

    env, policy, estimator, depth_encoder, ppo_runner = set_env(RunArgs)

    # log the gym corner align translation
    with logger.Prefix(PlayArgs.dataset_prefix):
        PlayArgs.gym_tf_pos = (-env.terrain.corner_shift).tolist() + [0]
        logger.log_params(play=vars(PlayArgs))

    num_labels = len(labels)
    labels_list = [(marker["start"], marker["goal"]) for marker in labels]

    # Set in terrain
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    sample_done = False
    for sample_num in trange(num_labels, desc="sample_num"):
        env.set_terrain_goal(labels_list[sample_num], 0, 0, shift_corners=True)
        obs, _ = env.reset()
        while not sample_done:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, : env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]

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

            robot_state = env.root_states[0].cpu().numpy()

            LOG_INFO["obs"].append(obs.cpu().numpy().astype(np.half))
            LOG_INFO["actions"].append(actions.detach().cpu().numpy().astype(np.half))
            LOG_INFO["states"].append(robot_state.astype(np.half))
            LOG_INFO["dofs"].append(env.dof_pos[0, :].cpu().numpy().tolist())

            obs, _, rews, dones, infos = env.step(actions.detach())
            sample_done = dones[0]

        sample_done = False

        # save and clear log info
        with logger.Prefix(str(Path(PlayArgs.dataset_prefix) / "rollouts")):
            logger.save_pkl(LOG_INFO, f"rollout_{sample_num:04d}.pkl")
        LOG_INFO = defaultdict(list)


if __name__ == "__main__":
    LeggedRobotCfg.terrain.n_sample_pcd = 100_000
    LeggedRobotCfg.terrain.num_rows = 1
    LeggedRobotCfg.terrain.num_cols = 1
    LeggedRobotCfg.terrain.terrain_width = 16
    LeggedRobotCfg.terrain.terrain_length = 16
    LeggedRobotCfg.terrain.border_size = 15
    LeggedRobotCfg.terrain.simplify_grid = False  # True

    RunArgs.task = "go1"
    PlayArgs.use_estimated_states = True
    RunArgs.delay = True
    # PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher/2024-01-05/12.40.55/go1/300"
    PlayArgs.logger_prefix = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"

    entrypoint()
