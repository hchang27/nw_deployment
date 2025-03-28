import time
import warnings

import torch
import torchvision

from cxx.modules.parkour_actor import PolicyArgs, DeploymentParams
from go1_gym_deploy.modules.base.lcm_agent import BasicLCMAgent
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING
from go1_gym_deploy.utils import get_rotation_matrix_from_rpy


class ParkourLCMAgent(BasicLCMAgent):
    def __init__(
        self,
        *,
        state_estimator,
        cam_node,
        device,
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

        self.n_proprio = PolicyArgs.n_proprio
        self.flat_mask = DeploymentParams.flat_mask
        self.history_len = PolicyArgs.history_len
        self.clip_actions = DeploymentParams.clip_actions
        self.action_scale = DeploymentParams.action_scale
        self.clip_observations = DeploymentParams.clip_observations
        self.update_interval = DeploymentParams.update_interval
        self.near_clip = DeploymentParams.near_clip
        self.far_clip = DeploymentParams.far_clip

        self.privileged_obs = torch.zeros(
            (1, PolicyArgs.n_scan + PolicyArgs.n_priv + PolicyArgs.n_priv_latent), device=device, dtype=torch.float32
        )

        self.height = DeploymentParams.height
        self.width = DeploymentParams.width

        self.previous_actions = torch.zeros((1, 12), device=device, dtype=torch.float32)  # using unitree indexing
        self.previous_contacts = torch.zeros((1, 4), device=device, dtype=torch.float32)  # using unitree indexing

        self.obs_history_buf = torch.zeros(1, self.history_len, self.n_proprio, device=device, dtype=torch.float32)
        self.extras = {}

        self.resize_transform = torchvision.transforms.Resize(
            (self.height, self.width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )

        self.device = device

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
                # 1,  # env class != flat
                # 0,  # env class == flat
                (self.dof_pos_unitree_index - self.default_dof_pos[0:1, JOINT_IDX_MAPPING]) * self.obs_scales["dof_pos"],
                # default dof pos is in isaacgym
                ((self.dof_vel_unitree_index) * self.obs_scales["dof_vel"]),
                self.previous_actions,
                self.contact_filt_unitree_index.float() - 0.5,
            ),
            dim=-1,
        )

        # note does not include privileged info
        self.obs_buf = torch.cat([obs_buf, self.privileged_obs, self.obs_history_buf.view(1, -1)], dim=-1)
        obs_buf[:, 6:8] = 0  # mask yaw for history buf

        if not skip_depth:
            self.extras["depth"] = self.retrieve_vision()

        # prepare for the next timestep
        if self.timestep <= 1:
            self.obs_history_buf = torch.stack([obs_buf] * self.history_len, dim=1)
        else:
            self.obs_history_buf = torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1)

    def step(self, actions, hard_reset=False, debug=False):
        """
        actions: from policy output, in unitree indexing. Converted to isaacgym here to be compatiable
        with default dof indexing
        """
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

    def retrieve_vision(self, force=True):
        """
        Return the next depth image at the specified update interval.

        Also preprocesses the depth image for direct input into policy.

        """
        depth = None
        if self.cam_node is not None:
            if self.timestep % self.update_interval == 0 or force:
                # depth = self.cam_node.frame["depth"]
                # ts = self.cam_node.frame["grab_ts"]
                rgb = self.cam_node.frame["rgb"]
                # print(f"Depth ts: {ts}")
                # print('used', time.perf_counter())

        depth = self.process_depth(depth)

        return rgb

    def process_depth(self, depth):
        """
        Process the ZED depth image by:
        1. Converting into meters [ already done in zed_node init_params]
        2. Invert
        2. Clip distances
        3. Resize
        4. Normalize
        """
        if depth is None:
            return

        depth = torch.from_numpy(depth).to(self.device)

        # depth = -depth

        # crop
        depth = self._crop_depth_image(depth)

        # clip
        depth = torch.clip(depth, self.near_clip, self.far_clip)
        depth = self.resize_transform(depth[None, :]).squeeze()
        depth = self._normalize_depth_image(depth)

        return depth[None, ...]

    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def _normalize_depth_image(self, depth_image):
        # depth_image = depth_image * -1
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        return depth_image
