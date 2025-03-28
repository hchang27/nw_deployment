from collections import defaultdict

import isaacgym
from isaacgym import gymapi
from params_proto import ParamsProto

assert isaacgym
import os

import torch
from tqdm import trange

from main_street.config import RunArgs
from main_street.envs import *

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


class PlayArgs(ParamsProto):
    logger_prefix = "/lucid-sim/lucid-sim/debug/launch/2023-12-14/15.37.25/200"
    exptid = None

    viz_ego = False

    direction_distillation = False
    use_estimated_states = False


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: f"{m:0>15}")
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def play(args):
    from ml_logger import logger

    assert (PlayArgs.logger_prefix is None) ^ (PlayArgs.exptid is None), "Either logger_prefix or exptid must be not None"

    if PlayArgs.exptid is not None:
        exptid = PlayArgs.exptid
        log_pth = f"../logs/{args.proj_name}/" + exptid

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
        "parkour": 1.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.0,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
        "stairs": 0.0,
    }

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
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    # don't reset
    train_cfg.policy.init_noise_std = 0

    if PlayArgs.exptid is None:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            load_checkpoint=PlayArgs.logger_prefix, env=env, name=args.task, args=args, train_cfg=train_cfg
        )
    else:
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
            load_checkpoint=log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True
        )

    if env.cfg.depth.use_camera:
        policy = ppo_runner.get_depth_actor_inference_policy(device=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    frames = []
    depths = []

    log_dict = defaultdict(list)

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 105.44168090820312
    camera_props.width = 960
    camera_props.height = 600

    # policy = ParkourActor(device=env.device)
    # PolicyArgs.use_camera = RunArgs.use_camera
    # policy.load(f"{PlayArgs.logger_prefix}/checkpoints/model_last.pt")

    env.attach_camera(0, env.envs[0], env.actor_handles[0], force=True, camera_props=camera_props)

    for i in trange(500):
        tf = env.gym.get_camera_transform(env.sim, env.envs[0], env.cam_handles[-1])

        # x, y, z = tf.p.x, tf.p.y, tf.p.z
        # print(f"Camera position: {x:.2f}, {y:.2f}, {z:.2f}")
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

        # obs[:, 6:8] = 0.0
        # print("Yaw:", obs[0, 6:8])
        obs_simplified = torch.cat((obs[:, : env.cfg.env.n_proprio], obs[:, -env.cfg.env.history_len * env.cfg.env.n_proprio :]), dim=-1)

        if PlayArgs.use_estimated_states:
            priv_states_estimated = estimator(obs[:, : env.cfg.env.n_proprio])
            obs[
                :,
                ppo_runner.alg.num_prop + ppo_runner.alg.num_scan : ppo_runner.alg.num_prop
                + ppo_runner.alg.num_scan
                + ppo_runner.alg.priv_states_dim,
            ] = priv_states_estimated

        actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        # actions, *_ = policy(ego=None, obs=obs.detach(), vision_latent=None)

        log_dict["obs"].append(obs_simplified.cpu())
        log_dict["actions_isaac"].append(actions.cpu())
        log_dict["depth"].append(infos["depth"].cpu() if infos["depth"] is not None else None)

        obs, _, rews, dones, infos = env.step(actions.detach())
        # print(rews)

        # FIXME: clean this up
        if PlayArgs.viz_ego:
            frame = env.render(mode="logger")
            frames.append(frame)
        else:
            if env.cfg.depth.use_camera:
                frame, depth = env.render(mode="logger", visualize_depth=True)
                # print(depth.min())
                frames.append(frame)
                depths.append(depth)
            else:
                frame = env.render(mode="logger")
                frames.append(frame)

    if not args.web and PlayArgs.logger_prefix is not None:
        with logger.Prefix(PlayArgs.logger_prefix):
            logger.save_video(frames, "video.mp4", fps=50)
            if len(depths) > 0:
                logger.save_video(depths, "depth.mp4", fps=50)

            logger.save_pkl(log_dict, "log_dict.pkl")
            print(f"Saved video {logger.get_dash_url()}")


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    from main_street.envs.base.legged_robot_config import LeggedRobotCfg
    from main_street.envs.go1.go1_config import Go1RoughCfg

    LeggedRobotCfg.commands.max_ranges.lin_vel_x = [0.5, 0.8]
    RunArgs.seed = 1920
    # PlayArgs.logger_prefix = "/instant-feature/scratch/2024/04-02/041520/"
    # PlayArgs.direction_distillation = True

    RunArgs.task = "go1"
    # PlayArgs.logger_prefix = "/lucidsim/lucidsim/parkour/baselines/launch/go1_stairs/300"
    # PlayArgs.logger_prefix = "/lucidsim/lucidsim/parkour/stairs_v2/launch/go1_stairs/200/checkpoints/model_last.pt"
    PlayArgs.logger_prefix = "/lucidsim/lucidsim/parkour/baselines/launch_gabe_go1_higher_speeds/go1/300"
    # PlayArgs.logger_prefix = "/instant-feature/scratch/2024/05-18/220933"
    # PlayArgs.logger_prefix = "/instant-feature/scratch/2024/05-23/003246/"
    RunArgs.delay = True
    # RunArgs.use_camera = True

    Go1RoughCfg.depth.dis_noise = 0

    RunArgs.seed = 10000

    PlayArgs.use_estimated_states = True

    LeggedRobotCfg.terrain.no_roughness = True

    delay = 6
    LeggedRobotCfg.domain_rand.action_curr_step = [delay, delay]
    play(RunArgs)
