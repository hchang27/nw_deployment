from main_street.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
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

    class init_state_slope(LeggedRobotCfg.init_state):
        pos = [0.5, 0.0, 0.24]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,  # [rad]
            'RL_hip_joint': 0.03,  # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,  # [rad]

            'FL_thigh_joint': 1.0,  # [rad]
            'RL_thigh_joint': 1.8,  # [rad]1.8
            'FR_thigh_joint': 1.0,  # [rad]
            'RR_thigh_joint': 1.8,  # [rad]

            'FL_calf_joint': -2.2,  # [rad]
            'RL_calf_joint': -1.2,  # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -1.2,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{MAIN_ST_ROOT_DIR}/assets/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]  # , "thigh", "calf"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        # class scales( LeggedRobotCfg.rewards.scales ):
        # torques = -0.0002
        # dof_pos_limits = -10.0


class A1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_a1'
