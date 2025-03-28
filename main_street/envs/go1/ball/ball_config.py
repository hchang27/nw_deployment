import numpy as np

from main_street.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO


class Go1BallCfg(Go1RoughCfg, cli=False):
    class env(Go1RoughCfg.env, cli=False):
        center_robot = True
        yaw_resampling_rate = 1

    class ball:
        curriculum = False
        max_level = 8
        init_level = 0

        speed_increment = 0.5  # 0.25

        revolving = False

        resampling_time = 5 * 50

        stopping_distance = 0.25

        class view:
            position = [0.29, 0.0, 0.02]  # from default: 2 cm forward, 3cm left

            width = 640
            height = 360
            horizontal_fov = 105
            vertical_fov = 30  # get_vertical_fov(horizontal_fov, width, height)  # vertical
            stream = "frame"
            fps = 30
            near = 0.5
            far = 1.0
            key = "ego"
            showFrustum = True
            downsample = 1
            distanceToCamera = 2

            spawn_padding = 0.25

        class asset:
            file = "{MAIN_ST_ROOT_DIR}/assets/robots/ball/urdf/ball.urdf"
            disable_gravity = True
            fix_base_link = True

        class ranges:
            r = [2.0, 3.0]
            phi = [-np.pi / 4, np.pi / 4]
            theta = [np.pi / 3, np.pi / 2]

    class commands(Go1RoughCfg.commands, cli=False):
        class max_ranges(Go1RoughCfg.commands.max_ranges, cli=False):
            lin_vel_x = [1.0, 1.5]


class Go1BallCfgPPO(Go1RoughCfgPPO, cli=False):
    pass
