import time
from abc import abstractmethod
from typing import List

import lcm
import numpy as np
import torch
from torchtyping import TensorType

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.modules.base.state_estimator import (
    JOINT_IDX_MAPPING,
    BasicStateEstimator,
)

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

"""
Middleman between state estimation and policy.
Prep observations from SE and publish actions to the robot. 
"""


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class BasicLCMAgent:
    dof_names = [
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

    def __init__(
        self,
        state_estimator: BasicStateEstimator,
        dt,
        default_joint_angles: dict,
        stiffness_dict: dict,
        damping_dict: dict,
        action_scale: float,
        obs_scales,
        device: str,
        cam_node,
    ):
        self.default_joint_angles = default_joint_angles
        self.stiffness_dict = stiffness_dict
        self.damping_dict = damping_dict
        self.action_scale = action_scale
        self.obs_scales = obs_scales

        self.se: BasicStateEstimator = state_estimator

        self.device: str = device

        self.num_dofs = 12
        self.p_gains = np.zeros(self.num_dofs)
        self.d_gains = np.zeros(self.num_dofs)

        self._prepare_cfg()

        self.timestep = 0
        self.time = time.time()
        self.dt = dt

        self.prev_body_ang_vel = torch.zeros(3, device=self.device)

        self.cam_node = cam_node

    def _prepare_cfg(self):
        """
        Load:
        - default joint angles
        - PD gains
        - commands scale (TODO)
        - obs scale
        """
        assert self.num_dofs == len(self.dof_names), "Number of DOFs mismatch"

        self.default_dof_pos = torch.zeros(self.num_dofs, device=self.device)[None, ...]

        # populate the default angles (in rad) from the config file
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos[0, i] = self.default_joint_angles[name]

            found = False
            for dof_name in self.stiffness_dict.keys():
                if dof_name in name:
                    self.p_gains[i] = self.stiffness_dict[dof_name]
                    self.d_gains[i] = self.damping_dict[dof_name]
                    found = True

            assert found, f"PD gains not found for joint {name}"

    @abstractmethod
    def compute_observations(self):
        """
        Read command from command profile / state estimator and return observation for policy
        """

    def publish_action(self, action: TensorType["batch", 12], hard_reset=False, debug=False) -> List:
        """
        WARNING: Make sure the drive mode is set to POSITION, other modes not supported right now.

        IMPORTANT: actions is expected to be in the UNITREE indexing format.

        Note that we do not scale the hip positions
        """
        command_for_robot = pd_tau_targets_lcmt()
        joint_pos_target = action[0, :12].detach()

        # scale the actions
        joint_pos_target *= self.action_scale
        joint_pos_target = joint_pos_target.flatten()

        # add offset
        joint_pos_target += self.default_dof_pos[0][JOINT_IDX_MAPPING]

        joint_pos_target = joint_pos_target.cpu().numpy().tolist()

        joint_vel_target = np.zeros(12).tolist()

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10**6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        # self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (
        #         self.joint_vel_target - self.dof_vel) * self.d_gains

        if not debug:
            # print("Publishing action", joint_pos_target)
            self.se.lc.publish("pd_plustau_targets", command_for_robot.encode())
        else:
            pass
            # print(joint_pos_target)

        return joint_pos_target

    def reset(self) -> TensorType["batch", "num_observations"]:
        self.actions = torch.zeros(12, device=self.device)
        self.time = time.time()
        self.timestep = 0

        self.compute_observations()
        return self.obs_buf

    @abstractmethod
    def step(self, actions, hard_reset=False):
        """
        Reads the actions by the agent and returns next [observation, reward, done, info]
        """
        pass
