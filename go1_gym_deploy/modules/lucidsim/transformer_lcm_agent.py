import time
import warnings
from collections import deque

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch import TensorType

from behavior_cloning.go1_model.transformers.transformer_policy import TransformerPolicyArgs
from cxx.modules.parkour_actor import DeploymentParams
from go1_gym_deploy.modules.base.lcm_agent import BasicLCMAgent
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING
from go1_gym_deploy.utils import get_rotation_matrix_from_rpy


class TransformerLCMAgent(BasicLCMAgent):
    def __init__(
        self,
        *,
        state_estimator,
        cam_node,
        device,
        stack_size=7,
        imagenet_pipe=False,
        render_type="rgb",
        # for depth:
        near_clip=0.28,
        far_clip=2.0,
        resize_shape=(45, 80),
        # fillers
        **rest,
    ):
        super().__init__(
            state_estimator,
            DeploymentParams.dt,
            DeploymentParams.default_joint_angles,
            DeploymentParams.stiffness_dict,
            DeploymentParams.damping_dict,
            DeploymentParams.action_scale,
            DeploymentParams.obs_scales,
            device,
            cam_node,
        )

        if rest:
            warnings.warn(f"Redundant arguments are passed in: {rest}")

        self.n_proprio = TransformerPolicyArgs.obs_dim
        self.flat_mask = DeploymentParams.flat_mask
        self.clip_actions = DeploymentParams.clip_actions
        self.action_scale = DeploymentParams.action_scale
        self.clip_observations = DeploymentParams.clip_observations

        self.render_type = render_type

        self.stack_size = stack_size

        self.height = DeploymentParams.height
        self.width = DeploymentParams.width

        self.previous_actions = torch.zeros((1, 12), device=device, dtype=torch.float32)  # using unitree indexing
        self.previous_contacts = torch.zeros((1, 4), device=device, dtype=torch.float32)  # using unitree indexing

        self.extras = {}

        self.resize_transform = torchvision.transforms.Resize(
            (self.height, self.width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )

        self.near_clip = near_clip
        self.far_clip = far_clip
        self.resize_shape = resize_shape

        pipeline = [
            # self.resize_transform,
            T.ToImage(),
        ]

        self.channels = 1 if self.render_type == "depth" else 3
        if imagenet_pipe:
            pipeline += [
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            pipeline += [
                T.ToDtype(torch.float32, scale=False),
                T.Normalize(mean=[127.5] * self.channels, std=[255] * self.channels),
            ]

        self.transform = T.Compose(pipeline)
        self.frame_buffer = None
        self.obs_queue = None

    def prepare_buffers(self):
        """
        Prepare buffers before computing observations.
        """

        self.base_ang_vel = self._from_se("deuler")
        self.prev_body_ang_vel = self.base_ang_vel
        self.imu = self._from_se("euler")
        self.dof_pos_unitree_index = self._from_se("joint_pos")
        self.dof_vel_unitree_index = self._from_se("joint_vel")
        self.dof_pos = self.dof_pos_unitree_index[:, JOINT_IDX_MAPPING]
        self.contact_state_unitree_index = self._from_se("contact_state")
        self.contact_filt_unitree_index = torch.logical_or(self.contact_state_unitree_index, self.previous_contacts)
        self.previous_contacts = self.contact_state_unitree_index[:]

        self.R = get_rotation_matrix_from_rpy(self.imu.squeeze(0))
        self.projected_gravity = self.R.T @ torch.tensor([0, 0, -1.0], device=self.device)[..., None]

    def _from_se(self, key):
        return torch.from_numpy(self.se.data[key]).to(self.device)

    def compute_observations(self, skip_depth=False):
        """
        Compute observations from state estimator.
        """
        self.prepare_buffers()

        # FIXME: deal with this yaw stuff
        command = self.se.get_command()
        target_speed = command[0]

        target_yaw = command[2]

        self.delta_yaw = torch.zeros_like(self.imu[:, 2], device=self.device) + target_yaw
        self.delta_next_yaw = self.delta_yaw

        if self.flat_mask:
            mask_obs = torch.tensor([[0.0, 1.0]], device=self.device)
        else:
            mask_obs = torch.tensor([[1.0, 0.0]], device=self.device)

        obs_buf = torch.cat(
            (
                (self.base_ang_vel * self.obs_scales["ang_vel"]),
                self.imu[:, :2],  # roll, pitch
                0 * self.delta_yaw[None, ...],
                self.delta_yaw[None, ...],
                self.delta_next_yaw[None, ...],
                torch.zeros((1, 2), device=self.device),
                torch.tensor([[target_speed]], device=self.device),
                mask_obs,
                (self.dof_pos_unitree_index - self.default_dof_pos[0:1, JOINT_IDX_MAPPING]) * self.obs_scales["dof_pos"],
                ((self.dof_vel_unitree_index) * self.obs_scales["dof_vel"]),
                self.previous_actions,
                self.contact_filt_unitree_index.float() - 0.5,
            ),
            dim=-1,
        )

        if self.obs_queue is None:
            self.obs_queue = deque([obs_buf] * self.stack_size, maxlen=self.stack_size)
        else:
            self.obs_queue.append(obs_buf)

        # most recent observation is at the end
        obs = torch.cat(tuple(self.obs_queue))[None, ...]
        self.obs_buf = obs

        if self.cam_node is not None:
            self.extras["vision"] = self.retrieve_vision()

    def step(self, actions, hard_reset=False, debug=False):
        """
        actions: from policy output, in unitree indexing. Converted to isaacgym here to be compatiable
        with default dof indexing
        """
        if len(actions.shape) == 1:
            actions = actions[None, ...]

        assert actions.shape == (1, 12), f"Expected shape (1, 12), got {actions.shape}"

        self.previous_actions = actions.clone()  # before scaling, using unitree indexing

        clip_actions = self.clip_actions / self.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        joint_pos_target = self.publish_action(self.actions, hard_reset=hard_reset, debug=debug)

        time_elapsed = time.time() - self.time
        if time_elapsed < self.dt:
            time.sleep(self.dt - time_elapsed)
        # elif time_elapsed < 1:
        #     raise ValueError("The inference time is too long :(")
        if debug:
            print(f"Step time: {time.time() - self.time}, dt: {self.dt}")
        self.time = time.time()

        self.compute_observations()
        clip_obs = self.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.timestep += 1

        return self.obs_buf, self.extras

    def retrieve_vision(self, force=False):
        """
        Return the next depth image at the specified update interval.

        Also preprocesses the depth image for direct input into policy.

        """
        frame = self.cam_node.frame[self.render_type]
        # frame = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        frame = self.process_frame(frame)

        return frame

    def process_frame(self, frame):
        if self.render_type == "depth":
            # raise NotImplementedError
            if self.near_clip is not None and self.far_clip is not None:
                frame = cv2.resize(frame, self.resize_shape[::-1], interpolation=cv2.INTER_CUBIC)
                frame = np.clip(frame, self.near_clip, self.far_clip)

                frame = (frame - self.near_clip) / (self.far_clip - self.near_clip)

                frame = (frame[..., None] * 255).astype(np.uint8)

        frame = torch.from_numpy(frame).to(self.device)

        if self.frame_buffer is None:
            # first step, fill with the frame
            # self.frame_buffer = frame[None, :].repeat(self.stack_size, 1, 1, 1)
            self.frame_buffer = deque([self.transform(frame[None, :].permute(0, 3, 1, 2))] * self.stack_size, maxlen=self.stack_size)
        else:
            # self.frame_buffer = torch.cat((self.frame_buffer[1:], frame[None, :]), dim=0)
            self.frame_buffer.append(self.transform(frame[None, :].permute(0, 3, 1, 2)))
            # self.frame_buffer = self.frame_buffer[:-1] + [frame[None, :].permute(0, 3, 1, 2)]

        # processed_input: TensorType["stack", "channels", "height", "width"] = self.frame_buffer.permute(0, 3, 1, 2)
        processed_input: TensorType["stack", "channels", "height", "width"] = torch.cat(tuple(self.frame_buffer), dim=0)
        # processed_input = processed_input.to(self.device, non_blocking=True).contiguous()
        # processed_input = self.transform(processed_input)

        return processed_input[None, ...]
