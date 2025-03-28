from typing import Literal

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T

from go1_gym_deploy.modules.parkour.parkour_lcm_agent import ParkourLCMAgent


class LucidLCMAgent(ParkourLCMAgent):
    def __init__(
        self,
        *,
        # control
        dt,
        default_joint_angles: dict,
        stiffness_dict: dict,
        damping_dict: dict,
        action_scale: float,
        obs_scales,
        clip_actions,
        clip_observations,
        # policy
        n_proprio,
        history_len,
        # depth
        height,
        width,
        update_interval,
        near_clip,
        far_clip,
        # terrain
        flat_mask: bool,
        # other nodes
        state_estimator,
        cam_node,
        device,
        # lucidsim
        stack_size: int,
        imagenet_pipe: bool,
        render_type: Literal["rgb", "depth"],
        normalize_depth: bool,
        compute_deltas: bool,
        drop_last: bool,
        **rest,
    ):
        super().__init__(
            dt=dt,
            default_joint_angles=default_joint_angles,
            stiffness_dict=stiffness_dict,
            damping_dict=damping_dict,
            action_scale=action_scale,
            obs_scales=obs_scales,
            clip_actions=clip_actions,
            clip_observations=clip_observations,
            n_proprio=n_proprio,
            history_len=history_len,
            height=height,
            width=width,
            update_interval=update_interval,
            near_clip=near_clip,
            far_clip=far_clip,
            flat_mask=flat_mask,
            state_estimator=state_estimator,
            cam_node=cam_node,
            device=device,
            **rest,
        )

        self.height = height
        self.width = width

        frame_size = (self.height, self.width)

        if imagenet_pipe:
            pipeline = [
                T.Resize(frame_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            pipeline = [
                T.Resize(frame_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=False),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
            ]

        self.transform = T.Compose(pipeline)

        self.frame_buffer = torch.zeros((stack_size, self.height, self.width, 3), device=self.device, dtype=torch.uint8)

        self.normalize_depth = normalize_depth
        self.compute_deltas = compute_deltas
        self.drop_last = drop_last

        self.render_type = render_type

    def retrieve_depth(self, force=False):
        if self.cam_node is not None:
            if force or (self.timestep % self.update_interval == 0):
                frame = self.cam_node.frame[self.render_type]

        frame = self.process_frame(frame)

        return frame

    def process_frame(self, frame):
        if self.render_type == "depth":
            # clip
            frame = np.clip(frame, self.near_clip, self.far_clip)

            if self.normalize_depth:
                frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
                frame = frame.astype(np.uint8)
                frame = 255 - frame

            # make 3d
            frame = np.stack((frame, frame, frame), axis=-1)

        frame = cv2.resize(frame, (self.width, self.height))

        frame = torch.from_numpy(frame).to(device=self.device)

        self.frame_buffer = torch.cat((self.frame_buffer[1:], frame[None, :]), dim=0)

        processed_input = self.frame_buffer.permute(0, 3, 1, 2).contiguous()
        processed_input = processed_input.to(self.device, non_blocking=True).contiguous()
        processed_input = self.transform(processed_input)

        if self.compute_deltas:
            processed_input[:-1] = processed_input[-1:] - processed_input[:-1]

        if self.drop_last:
            processed_input = processed_input[:-1]

        # flatten
        return processed_input.reshape(-1, self.height, self.width)[None, ...]
