from typing import Literal

import torch
from params_proto import Proto
from vuer import VuerSession
from vuer.schemas import Plane, Sphere, group

from lucidsim_old.terraria.eval.hurdle import HurdleEval


class TrackingEval(HurdleEval):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "hurdle/scene_99999"

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
    ball_color = "red"

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

        return env

    def _get_material_kwargs(self, color, texture=None, repeatTexture=False, ball_mask=False):
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
            raise NotImplementedError
            # return dict()

    def _get_mesh(self):
        mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color, self.plane_texture))

        ball_material_kwargs = self._get_material_kwargs(self.ball_color, self.ball_texture, ball_mask=True)
        ball = Sphere(
            key="ball",
            args=[self.ball_radius, 20, 20],
            **ball_material_kwargs,
        )

        return group(
            mesh,
            ball,
        )

    def _set_goals(self, sample_number):
        return

    def _set_robot_pose(self, robot_state, joint_angles, sess: VuerSession):
        super()._set_robot_pose(robot_state[0:1], joint_angles, sess)

        # update ball position
        ball_location = robot_state[1:2, :3].cpu().numpy().tolist()
        sess.update @ Sphere(key="ball",
                             position=ball_location[0],
                             args=[self.ball_radius, 20, 20],
                             **self._get_material_kwargs(self.ball_color, self.ball_texture, ball_mask=True))
