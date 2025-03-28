from main_street.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from main_street.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO


class Go1FlatCfg(Go1RoughCfg, cli=False):
    class env(Go1RoughCfg.env, cli=False):
        n_scan = 0
        num_observations = LeggedRobotCfg.env.n_proprio + n_scan + LeggedRobotCfg.env.history_len * LeggedRobotCfg.env.n_proprio + LeggedRobotCfg.env.n_priv_latent + LeggedRobotCfg.env.n_priv

    class terrain(Go1RoughCfg.terrain, cli=False):
        mesh_type = "trimesh"
        measure_heights = False

        curriculum = True
        flat_mask = True

        terrain_dict = {"smooth slope": 0.,
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
                        "demo": 0.0,
                        "stairs": 0}

        terrain_proportions = list(terrain_dict.values())

    # class control(Go1RoughCfg.control):
    #     control_type = "actuator_net"
    #     actuator_net_path = "../../assets/actuator_nets/unitree_go1.pt"


class Go1FlatCfgPPO(Go1RoughCfgPPO, cli=False):
    class estimator(Go1RoughCfgPPO.estimator, cli=False):
        num_scan = Go1FlatCfg.env.n_scan
