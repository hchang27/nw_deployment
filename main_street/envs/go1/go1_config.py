from main_street.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1RoughCfg(LeggedRobotCfg, cli=False):
    # uncomment for expanded history and clock inputs 
    # class env(LeggedRobotCfg.env):
    #     history_len = 50
    #     n_proprio = LeggedRobotCfg.env.n_proprio + 6 # 6 for clock inputs
    #     observe_clock_inputs = True
    #     num_observations = n_proprio + LeggedRobotCfg.env.n_scan + history_len * n_proprio + LeggedRobotCfg.env.n_priv_latent + LeggedRobotCfg.env.n_priv

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

    class depth(LeggedRobotCfg.depth, cli=False):
        # position = [0.29, 0.0425, 0.0167]  # from default: 2 cm forward, 3cm left 

        # position = [0.29, 0.0425, 0.08]  # from default: 2 cm forward, 3cm left
        # position = [0.29, 0.0, 0.11]
        position = [0.27, 0.0, 0.015]
        original = (80, 45)
        resized = (80, 45)
        near_clip = 0.28
        dis_noise = 0.03
        
        buffer_len = 2

        # horizontal_fov = 105.4500503540039
        # original = (96, 60)  # 1.6 AR
        # resized = (96, 60)  # 1.6 AR

    class init_state_slope(LeggedRobotCfg.init_state, cli=False):
        pos = [0.56, 0.0, 0.24]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,  # [rad]
            'RL_hip_joint': 0.03,  # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,  # [rad]

            'FL_thigh_joint': 1.0,  # [rad]
            'RL_thigh_joint': 1.9,  # [rad]1.8
            'FR_thigh_joint': 1.0,  # [rad]
            'RR_thigh_joint': 1.9,  # [rad]

            'FL_calf_joint': -2.2,  # [rad]
            'RL_calf_joint': -0.9,  # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,  # [rad]
        }

    class control(LeggedRobotCfg.control, cli=False):
        # PD Drive parameters:

        control_type = 'P'

        # control_type = "actuator_net"
        # actuator_net_path = "../../assets/actuator_nets/unitree_go1.pt"

        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 0.6}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset, cli=False):
        # file = '{MAIN_ST_ROOT_DIR}/assets/robots/go1/urdf/go1_new.urdf'
        file = '{MAIN_ST_ROOT_DIR}/assets/robots/gabe_go1/urdf/go1.urdf'
        # file = '{MAIN_ST_ROOT_DIR}/assets/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        flip_visual_attachments = False
        terminate_after_contacts_on = ["base"]  # , "thigh", "calf"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards, cli=False):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25


class Go1RoughCfgPPO(LeggedRobotCfgPPO, cli=False):
    class algorithm(LeggedRobotCfgPPO.algorithm, cli=False):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner, cli=False):
        run_name = ''
        experiment_name = 'rough_a1'

    class estimator(LeggedRobotCfgPPO.estimator, cli=False):
        num_prop = Go1RoughCfg.env.n_proprio

    class depth_encoder(LeggedRobotCfgPPO.depth_encoder, cli=False):
        if_depth = Go1RoughCfg.depth.use_camera
        depth_shape = Go1RoughCfg.depth.resized
        buffer_len = Go1RoughCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = Go1RoughCfg.depth.update_interval * 24
