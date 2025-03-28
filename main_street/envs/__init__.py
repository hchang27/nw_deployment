import platform

# Only run these when not on Macs.
if platform.system() != "Darwin":
    from .a1.a1_parkour_config import A1ParkourCfg, A1ParkourCfgPPO
    from .base.legged_robot import LeggedRobot
    from .go1.ball.ball import Ball
    from .go1.ball.ball_config import Go1BallCfg, Go1BallCfgPPO
    from .go1.ball.ball_sampling import BallSampling
    from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
    from .go1.go1_stairs_config import Go1StairsCfg
    from .go1.go1_hurdle_config import Go1HurdleCfg
    from .go1.go1_flat_config import Go1FlatCfg, Go1FlatCfgPPO

    from main_street.task_registry import task_registry

    task_registry.register("a1", LeggedRobot, A1ParkourCfg, A1ParkourCfgPPO)

    task_registry.register("go1", LeggedRobot, Go1RoughCfg, Go1RoughCfgPPO)
    task_registry.register("go1_stairs", LeggedRobot, Go1StairsCfg, Go1RoughCfgPPO)
    task_registry.register("go1_hurdle", LeggedRobot, Go1HurdleCfg, Go1RoughCfgPPO)

    task_registry.register("go1_flat", LeggedRobot, Go1FlatCfg, Go1FlatCfgPPO)
    task_registry.register("go1_ball", Ball, Go1BallCfg, Go1BallCfgPPO)
    task_registry.register("go1_ball_sampling", BallSampling, Go1BallCfg, Go1BallCfgPPO)
