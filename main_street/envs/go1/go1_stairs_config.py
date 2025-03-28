from main_street.envs.go1.go1_config import Go1RoughCfg


class Go1StairsCfg(Go1RoughCfg, cli=False):
    class terrain(Go1RoughCfg.terrain):
        terrain_dict = {
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
            "parkour_flat": 1 / 3,
            "parkour_step": 0.0,
            "parkour_gap": 0.0,
            "demo": 0.0,  # ,
            "stairs": 2 / 3,
        }

        terrain_proportions = list(terrain_dict.values())
