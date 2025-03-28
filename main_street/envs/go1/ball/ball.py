import os

import cv2
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_rotate, quat_rotate_inverse
from torchtyping import TensorType
from tqdm import tqdm

from main_street import MAIN_ST_ROOT_DIR
from main_street.envs.base.legged_robot import LeggedRobot

from main_street.utils.helpers import spherical_to_cartesian, project_point
from main_street.envs.go1.ball.ball_config import Go1BallCfg
import torch


class Ball(LeggedRobot):
    cfg: Go1BallCfg

    def render(self, mode="rgb_array", env_id=0):
        if mode == "logger":
            bx, by, bz = self.root_states[env_id, :3]

            # Note: the first parameter is the camera handle, which is a separate counter for each environment.

            cam_location = [bx, by - 4, bz + 4]

            camera_handle = 1 if self.cfg.depth.use_camera else 0
            ego_handle = camera_handle + 1

            self.gym.set_camera_location(
                camera_handle,
                self.envs[env_id],
                gymapi.Vec3(*cam_location),
                gymapi.Vec3(bx, by, bz),
            )

            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR)
            ego = self.gym.get_camera_image(self.sim, self.envs[env_id], ego_handle, gymapi.IMAGE_COLOR)

            w, h = img.shape
            image = img.reshape([w, h // 4, 4])

            w, h = ego.shape
            ego_image = ego.reshape([w, h // 4, 4])

            base_to_world = self.root_states[self.robot_actor_idxs[env_id], :3]
            ball_to_world = self.root_states[self.ball_actor_idxs[env_id], :3]

            dist_to_ball = torch.norm(base_to_world - ball_to_world)

            front_to_base = dist_to_ball * torch.tensor(
                [[1.0, 0.0, 0.0]], device=self.device
            )
            rear_to_base = 0.0 * torch.tensor([[-1.0, 0.0, 0.0]], device=self.device)

            front_to_world = quat_rotate(self.base_quat[env_id:env_id + 1], front_to_base) + base_to_world
            rear_to_world = quat_rotate(self.base_quat[env_id:env_id + 1], rear_to_base) + base_to_world

            front_pt = project_point(
                front_to_world.squeeze(0),
                torch.tensor(cam_location, device=self.device),
                torch.tensor((bx, by, bz), device=self.device),
                self.render_camera_props,
            )
            rear_pt = project_point(
                rear_to_world.squeeze(0),
                torch.tensor(cam_location, device=self.device),
                torch.tensor((bx, by, bz), device=self.device),
                self.render_camera_props,
            )

            cv2.line(image, front_pt, rear_pt, (125, 255, 125), 2)

            return image, ego_image

        elif mode == "rgb_array":
            super().render(mode)

    def get_ball_resets(self, env_ids, first_spawn=False):
        """
        Returns a tensor (num_envs, 6) giving the ball position and velocity for each environment.
        These are expected to be in cartesian.

        Difference is that we spawn using the spherical coordinates.
        """
        r = torch_rand_float(
            *self.cfg.ball.ranges.r, (len(env_ids), 1), device=self.device
        ).squeeze(-1)
        theta = torch_rand_float(
            *self.cfg.ball.ranges.theta, (len(env_ids), 1), device=self.device
        ).squeeze(-1)
        phi = torch_rand_float(
            *self.cfg.ball.ranges.phi, (len(env_ids), 1), device=self.device
        ).squeeze(-1)

        self.ball_speeds[env_ids] = torch.zeros(
            (len(env_ids),), device=self.device
        )  # dphi

        # get ball to world translation
        ball_to_base = spherical_to_cartesian(torch.stack((r, theta, phi), dim=-1))

        if first_spawn:
            # no root states are available 
            ball_to_world = ball_to_base + torch.tensor(
                [0, 0, self.base_init_state[2]], device=self.device
            )
            return torch.cat(
                (ball_to_world, torch.zeros_like(ball_to_world, device=self.device)),
                dim=1,
            )

        base_to_world_rotation = self.root_states[self.robot_actor_idxs[env_ids], 3:7]
        base_to_world_translation = self.root_states[self.robot_actor_idxs[env_ids], :3]

        ball_to_world = (
                quat_rotate(base_to_world_rotation, ball_to_base)
                + base_to_world_translation
        )

        if self.cfg.ball.curriculum or self.cfg.ball.init_level > 0:
            self.ball_speeds[env_ids] = (self.ball_env_levels[env_ids] * self.cfg.ball.speed_increment)

            sign_phi = torch.sign(
                torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device)
            ).squeeze(-1)
            self.ball_speeds[env_ids] *= sign_phi

        # pick random direction to throw the ball, only horizontal
        rand_theta = torch_rand_float(
            0, 2 * np.pi, (len(env_ids), 1), device=self.device
        ).squeeze(-1)

        dir = torch.sin(rand_theta) * torch.tensor([[1, 0, 0]], device=self.device) + \
              torch.cos(rand_theta) * torch.tensor([[0, 1, 0]], device=self.device)

        ball_state = torch.cat(
            (ball_to_world, self.ball_speeds * dir), dim=1
        )

        return ball_state

    def step_external_assets(self, ball_dx=None, ball_dy=None, ball_dz=None, **kwargs):
        """
        Update position according to the speed, if revolving. 
        :return: 
        """
        if self.cfg.ball.revolving:
            delta_phi = self.ball_speeds * self.dt
            cos_delta_phi = torch.cos(delta_phi)
            sin_delta_phi = torch.sin(delta_phi)

            rotation = torch.zeros(len(delta_phi), 2, 2).to(self.device)
            rotation[:, 0, 0] = cos_delta_phi.squeeze()
            rotation[:, 0, 1] = -sin_delta_phi.squeeze()
            rotation[:, 1, 0] = sin_delta_phi.squeeze()
            rotation[:, 1, 1] = cos_delta_phi.squeeze()

            origin_xy = self.env_origins[:, :2]

            relative_xy = self.root_states[self.ball_actor_idxs, :2] - origin_xy
            rotated_xy = torch.bmm(rotation, relative_xy.unsqueeze(-1)).squeeze(-1)

            self.root_states[self.ball_actor_idxs, :2] = rotated_xy + origin_xy

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(self.ball_actor_idxs),
                len(self.ball_actor_idxs),
            )
        elif (ball_dx is not None) and (ball_dy is not None) and (ball_dz is not None):
            self.root_states[self.ball_actor_idxs, :3] += torch.tensor([[ball_dx, ball_dy, ball_dz]],
                                                                       device=self.device)

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(self.ball_actor_idxs),
                len(self.ball_actor_idxs),
            )

    def update_ball_levels(self, env_ids):
        if not self.cfg.ball.curriculum:
            return
        average_alignment = (
                self.cumulative_alignment[env_ids] / self.episode_length_buf[env_ids]
        )

        # Compute the new levels for each env
        # if average alignment is less than 0.25, decrease the level
        self.ball_env_levels[env_ids] -= 1 * (average_alignment < 0.5)  # 0.25
        # if average alignment is greater than 0.75, increase the level
        self.ball_env_levels[env_ids] += 1 * (average_alignment > 0.85)  # 0.75

        # Clip levels
        self.ball_env_levels[env_ids] = torch.clamp(
            self.ball_env_levels[env_ids], 0, self.cfg.ball.max_level - 1
        )

        # Reset the cumulative alignment buffer
        self.cumulative_alignment[env_ids] = 0

    ########################## Callbacks ##########################

    def _gather_cur_goals(self, future=0) -> TensorType["num_envs", 3]:
        # TODO: Note future is not considered as we just care about the current goal 
        if not self.init_done:
            return torch.zeros((self.num_envs, 3), device=self.device)
        return self.root_states[self.ball_actor_idxs, :3]

    def _generate_external_assets(self):
        self.has_external_assets = True

        asset_path = self.cfg.ball.asset.file.format(MAIN_ST_ROOT_DIR=MAIN_ST_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = self.cfg.ball.asset.disable_gravity
        asset_options.fix_base_link = self.cfg.ball.asset.fix_base_link

        self.ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_ball_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)

        self.ball_handles = []
        self.ball_actor_idxs = []
        self.ball_env_levels = torch.zeros(self.num_envs, device=self.device,
                                           dtype=torch.int32) + self.cfg.ball.init_level
        self.ball_speeds = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_alignment = torch.zeros(self.num_envs, device=self.device)

        self.ball_spawns = self.get_ball_resets(
            torch.arange(self.num_envs, device=self.device), first_spawn=True
        )[:, :3]

    def _add_external_assets(self, env_handle, i):
        ball_spawn_pose = gymapi.Transform()
        ball_spawn_pose.p = gymapi.Vec3(*self.ball_spawns[i])

        # last two args: collision group and collision filter ( common set bit == collision enabled )
        ball_handle = self.gym.create_actor(
            env_handle, self.ball_asset, ball_spawn_pose, "ball", i, 0
        )

        self.envs.append(env_handle)
        self.ball_handles.append(ball_handle)

        self.ball_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

    def _set_asset_idxs(self):
        self.ball_actor_idxs = torch.Tensor(self.ball_actor_idxs).to(device=self.device, dtype=torch.int32)

    def _reset_external_assets(self, env_ids):
        self.update_ball_levels(env_ids)

        next_ball_states = self.get_ball_resets(env_ids)
        self.root_states[self.ball_actor_idxs[env_ids], :3] = next_ball_states[:, :3]
        self.root_states[self.ball_actor_idxs[env_ids], 7:10] = next_ball_states[:, 3:]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(self.ball_actor_idxs[env_ids]),
                                                     len(env_ids))

        # self.extras["episode"]["mean_ball_env_levels"] = self.ball_env_levels.to(torch.float32).mean()

    def _compute_ball_alignment(self):
        base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        ball_to_base = (
                quat_rotate_inverse(base_quat, self.root_states[self.ball_actor_idxs, :3]) - self.root_states[
                                                                                             self.robot_actor_idxs, :3]
        )
        ball_to_base_unit = ball_to_base / torch.norm(ball_to_base, dim=1, keepdim=True)
        alignment = ball_to_base_unit[:, 0]
        return alignment
