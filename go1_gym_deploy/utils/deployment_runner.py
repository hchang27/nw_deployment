import copy
import time
import warnings
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pickle


from go1_gym_deploy.modules.base.lcm_agent import BasicLCMAgent
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING
from go1_gym_deploy.scripts.build_trt_engine import build_engine

try:
    from jtop import jtop
except ModuleNotFoundError:
    warnings.warn("jtop not found, cannot log jetson stats. OK if not running on jetson.")


class DeploymentRunner:
    def __init__(self, policy, lcm_agent: BasicLCMAgent, mode):
        self.lcm_agent = lcm_agent
        self.policy = policy

        self.button_states = np.zeros(4)

        self.trigger_pressed = False
        self.mode = mode

        self.log_dict = {}
        self.img_buffer = None
        self.img_memory_length = 10



    def load_dict(self, scene):
        dict_dir = f"/home/unitree/nw_deploy/parkour/go1_gym_deploy/scripts/ckpts/go2_deploy/{scene}/dataset_stats.pkl"
        with open(dict_dir, "rb") as f:
            stats = pickle.load(f)
        return stats

    def process_data(self, newest_frame):
        newest_frame = newest_frame.transpose(2, 0, 1) / 255.0
        newest_frame = torch.from_numpy(newest_frame).float().to("cuda")
        newest_frame = F.interpolate(newest_frame.unsqueeze(0), size=(180, 320), mode='bilinear', align_corners=False)
        newest_frame = newest_frame.unsqueeze(0)

        if self.img_buffer is None:
            self.img_buffer = deque([newest_frame] * self.img_memory_length, maxlen=self.img_memory_length)
        else:
            self.img_buffer.append(newest_frame)

        img_input = torch.cat(list(self.img_buffer), dim=1)
        return img_input    
    
    # def process_data(self, newest_frame):
    #     newest_frame = cv2.resize(newest_frame, (320, 180), interpolation=cv2.INTER_LINEAR)[None, None, ...]

    #     newest_frame = newest_frame.transpose(0, 1, 4, 2, 3)

    #     if self.img_buffer is None:
    #         self.img_buffer = deque([newest_frame] * self.img_memory_length, maxlen=self.img_memory_length)
    #     else:
    #         self.img_buffer.append(newest_frame)

    #     img_input = np.concatenate(list(self.img_buffer), axis=1)
    #     return img_input

    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        agent = self.lcm_agent
        agent.compute_observations()
        joint_pos = agent.dof_pos  # isaacgym indexing, uncentered

        print(f"Current joint positions: {joint_pos}")

        # Final goal is relative to the nominal pose
        if low:
            final_goal = np.array(
                [
                    0.0,
                    0.3,
                    -0.7,
                    0.0,
                    0.3,
                    -0.7,
                    0.0,
                    0.3,
                    -0.7,
                    0.0,
                    0.3,
                    -0.7,
                ]
            )
        else:
            final_goal = np.zeros(12)

        nominal_joint_pos = agent.default_dof_pos  # in isaacgym indexing

        print("About to calibrate; the robot will stand [Press R2 to calibrate]")
        while wait:
            if self.lcm_agent.se.data["trigger"][1]:
                self.trigger_pressed = False
                break
            # time.sleep(0.001)

        cal_action = np.zeros((1, 12))
        target_sequence = []

        # target is the next position, relative to the nominal pose. centered, isaacgym indexing
        
        target = (joint_pos - nominal_joint_pos).cpu()
        for t in range(200):
            if torch.max(torch.abs(target - torch.from_numpy(final_goal))) > 0.01:
                blend_ratio = np.minimum(t / 150., 1)
                action = blend_ratio * (target - final_goal)
                target -= action
                target_sequence += [copy.deepcopy(target)]

        print(f"Nominal pose: {nominal_joint_pos}")
        print(f"Target sequence: {target_sequence}")
        print(f"Number of steps: {len(target_sequence)}")
        dif_list = []
        dif_from_nominal = []
        for target in target_sequence:
            next_target = copy.deepcopy(target)

            action_scale = agent.action_scale

            next_target = next_target / action_scale
            cal_action[:, 0:12] = next_target

            agent.step(torch.from_numpy(cal_action[:, JOINT_IDX_MAPPING]).to(device=agent.default_dof_pos.device), debug=False)
            agent.compute_observations()
            # dif_list.append(agent.dof_pos - target.to("cuda") - nominal_joint_pos)
            # dif_from_nominal.append(agent.dof_pos - nominal_joint_pos)
            time.sleep(0.2)

        print("Starting pose calibrated [Press R2 to start controller]")
        while wait:
            try:
                agent.compute_observations()
                self.button_states = self.lcm_agent.se.get_buttons()
                if self.lcm_agent.se.data["trigger"][1]:
                    self.trigger_pressed = False
                    print("out!")
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("Quitting!")
                exit()

        time.sleep(1)

        control_obs = self.lcm_agent.reset()
        return control_obs

    def run(
        self,
        max_steps=100000000,
        logging=True,
        action_replay_log=None,
        obs_replay_log=None,
        vision_key="rgb",
        wait=True,
        debug=False,
    ):
        assert all((self.lcm_agent is not None, self.policy is not None)), "Missing modules!"

        if action_replay_log is not None:
            max_steps = min(max_steps, len(action_replay_log))
        elif obs_replay_log is not None:
            max_steps = min(max_steps, len(obs_replay_log["obs"]))
            for key in obs_replay_log:
                for i in range(len(obs_replay_log[key])):
                    obs_replay_log[key][i] = obs_replay_log[key][i].to("cuda")

        print("Getting initial observations for calibration...")
        obs = self.lcm_agent.reset()
        obs = obs.to(torch.float32)
        # self.policy = torch.jit.freeze(self.policy.eval())
        # build TensorRT engine
        scene = "hurdle_fah_indoor_two_hurdle_wood_v1_NoCones"
        policy = "policy_last"
        stats = self.load_dict(scene)
        trt_engine = build_engine(scene, policy, mode=self.mode)

        # warm up
        if self.lcm_agent.cam_node is not None:
            newest_frame = self.lcm_agent.retrieve_vision(force=True)

            # self.policy(newest_frame, obs, vision_latent=torch.zeros(1, 32, device=obs.device))
            # import pdb; pdb.set_trace()
            # self.policy(newest_frame, obs)
            if self.mode == "depth":
                newest_frame = torch.stack([newest_frame] * 3, dim=1)[None, ...]
            
            processed_data = self.process_data(newest_frame)
            input_data = (processed_data, obs)
            trt_engine.run(input_data, 'torch_cuda') #haoran turn off for test


        print("Calibrating Robot")
        obs = self.calibrate(wait=wait)
        obs = obs.to(torch.float32)

        time.sleep(0.001)

        # Compute the most recent depth
        if self.lcm_agent.cam_node is not None:
            newest_frame = self.lcm_agent.retrieve_vision(force=True)
            self.lcm_agent.extras[vision_key] = newest_frame
        infos = self.lcm_agent.extras

        vision_latent = None

        self.log_dict = defaultdict(list)

        print("Starting the control loop now!")
        # # time.sleep(2) # wait for trigger to release
        # self.policy.eval()

        # if hasattr(self.policy, "reset_hiddens"):
        #     self.policy.reset_hiddens()

        vision_buffer = deque([infos.get(vision_key, None)] * 10, maxlen=10)
        delay = 0
        # NOTE reply the action!
        hcaction = np.load("/home/unitree/nw_deploy/parkour/hc_test/proc_action.npy")
        hcaction = torch.from_numpy(hcaction).to("cuda")
        try:
            print("infos", infos)

            # reset image buffer for deployment
            self.img_buffer = None 
            obs_list = []
            image_list = []
            action_list = []
            max_steps = 1000
            for i in range(max_steps):
                if action_replay_log is not None:
                    action = action_replay_log[i]
                elif obs_replay_log is not None:
                    assert print("this should not go here")
                    obs, vision = obs_replay_log["obs"][i], obs_replay_log["vision"][i]
                    # obs = obs.to("cuda")
                    # vision = vision.to("cuda")
                    with torch.no_grad():
                        action, *extra = self.policy(vision, obs)
                else:
                    # print("in")
                    with torch.no_grad():
                        # print("hi")
                        # action, *extra = self.policy(vision_buffer[-1 - delay], obs, vision_latent=vision_latent)
                        newest_frame = vision_buffer[-1 - delay]
                        if self.mode == "depth":
                            newest_frame = torch.stack([newest_frame] * 3, dim=1)[None, ...]
                        processed_data = self.process_data(newest_frame)
                        input_data = (processed_data, obs)
                        action = trt_engine.run(input_data, 'torch_cuda')[0]
                        action = action[0][0].unsqueeze(0)
                        action.detach().cpu() * stats["action_std"] + stats["action_mean"]

                        if i <= 100:
                            image_list.append(processed_data)
                            obs_list.append(obs)
                            action_list.append(action)
                        # if len(extra) > 0:
                        #     print("Yaw", extra[-1])
                        #     vision_latent = extra[0]
                # action = hcaction[i].unsqueeze(0)
                obs, infos = self.lcm_agent.step(action, debug=debug)
                obs = obs.to(torch.float32)

                vision_buffer.append(infos[vision_key])

                # bad orientation emergency stop
                rpy = self.lcm_agent.imu[0]
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                if self.lcm_agent.se.data["trigger"][1]:
                    obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.trigger_pressed = False
                    while not self.lcm_agent.se.data["trigger"][1]:
                        time.sleep(0.001)
                    self.trigger_pressed = False

                    # reset
                    obs = self.lcm_agent.reset()
                    obs = obs.to(torch.float32)
                    if hasattr(self.policy, "reset_hiddens"):
                        self.policy.reset_hiddens()

                    print("Starting again now!")
                    time.sleep(1.0)

            torch.save(torch.tensor(torch.stack(image_list)), '/home/unitree/nw_deploy/parkour/hc_test/real_image.pth')
            torch.save(torch.tensor(torch.stack(obs_list)), '/home/unitree/nw_deploy/parkour/hc_test/real_obs.pth')
            torch.save(torch.tensor(torch.stack(action_list)), '/home/unitree/nw_deploy/parkour/hc_test/real_action.pth')
            # finally, return to the nominal pose
            self.calibrate(wait=False)
            print("Finished running, returning to nominal pose")

        except KeyboardInterrupt:
            # from ml_logger import logger

            # print("Dashboard", logger.get_dash_url())
            # with logger.Sync():
            #     logger.save_pkl(self.log_dict, "log_dict.pkl")
            # print("Quitting!")
            exit()
