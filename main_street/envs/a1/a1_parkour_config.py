from main_street.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1ParkourCfg(LeggedRobotCfg, cli=False):
    class init_state(LeggedRobotCfg.init_state, cli=False):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control, cli=False):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset, cli=False):
        # file = '{MAIN_ST_ROOT_DIR}/assets/robots/go1/urdf/go1_new.urdf'
        file = '{MAIN_ST_ROOT_DIR}/assets/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]  # , "thigh", "calf"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards, cli=False):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25


class A1ParkourCfgPPO(LeggedRobotCfgPPO, cli=False):
    class algorithm(LeggedRobotCfgPPO.algorithm, cli=False):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner, cli=False):
        run_name = ''
        experiment_name = 'rough_a1'
