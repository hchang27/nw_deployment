
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate

from main_street.envs.go1.ball.ball import Ball
from main_street.envs.go1.ball.ball_config import Go1BallCfg
from main_street.utils.helpers import sample_camera_frustum_batch


class BallSampling(Ball):
    cfg: Go1BallCfg

    def __init__(self, *args, **kwargs):
        self.reach_goal_count = 0
        self.sample_count = 0
        self.triggered_flag = False
        super().__init__(*args, **kwargs)

    def reset(self):
        """ Reset all robots, and force a resampling of the ball location in the frustum """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False),
            force_ball_resample=True)

        self.reach_goal_count = self.sample_count = 0
        return obs, privileged_obs

    def get_ball_resets(self, env_ids, first_spawn=False):
        """
        """
        view_cfg = self.cfg.ball.view
        new_x, new_y, new_z = sample_camera_frustum_batch(
            view_cfg.horizontal_fov,
            view_cfg.width,
            view_cfg.height,
            view_cfg.near,
            view_cfg.far,
            len(env_ids)
        )

        samples_x = torch.from_numpy(new_x).to(device=self.device)
        samples_y = torch.from_numpy(new_y).to(device=self.device)
        samples_z = torch.from_numpy(new_z).to(device=self.device)

        samples_to_cam = torch.cat((samples_x, samples_y, samples_z, torch.ones_like(samples_x).to(self.device)),
                                   dim=1).float()

        cam_to_robot = torch.eye(4).to(self.device)
        cam_to_robot[:3, -1] = torch.tensor(view_cfg.position).to(self.device)

        samples_to_robot = (cam_to_robot @ samples_to_cam.T).T
        if first_spawn:
            # no root states are available 
            ball_to_world = samples_to_robot[:, :3] + torch.tensor(
                [0, 0, self.base_init_state[2]], device=self.device
            )
            return torch.cat(
                (ball_to_world, torch.zeros_like(ball_to_world, device=self.device)),
                dim=1,
            )

        ball_to_world = quat_rotate(self.root_states[self.robot_actor_idxs[env_ids], 3:7], samples_to_robot[:, :3]) + \
                        self.root_states[self.robot_actor_idxs[env_ids], :3]

        # if ball is under the ground, we reflect over the ground plane
        ball_to_world[:, 2] = torch.max(ball_to_world[:, 2], -ball_to_world[:, 2])
        ball_to_world[:, 2] = torch.max(ball_to_world[:, 2] - self.cfg.ball.view.spawn_padding,
                                        self.cfg.ball.view.spawn_padding + torch.zeros_like(ball_to_world[:, 2],
                                                                                            device=self.device))

        self.ball_speeds[env_ids] = torch.zeros(
            (len(env_ids),), device=self.device
        )  # dphi

        ball_state = torch.cat(
            [ball_to_world, torch.zeros((len(env_ids), 1), device=self.device)], dim=1)

        return ball_state

    def get_ball_resets_batched(self, env_ids, first_spawn=False, batch_size=10):
        """
        Sample ball resets in batches 
        """
        view_cfg = self.cfg.ball.view
        new_x, new_y, new_z = sample_camera_frustum_batch(
            view_cfg.horizontal_fov,
            view_cfg.width,
            view_cfg.height,
            view_cfg.near,
            view_cfg.far,
            len(env_ids) * batch_size
        )

        new_x = new_x.reshape(len(env_ids), batch_size, 1)
        new_y = new_y.reshape(len(env_ids), batch_size, 1)
        new_z = new_z.reshape(len(env_ids), batch_size, 1)

        samples_x = torch.from_numpy(new_x).to(device=self.device)
        samples_y = torch.from_numpy(new_y).to(device=self.device)
        samples_z = torch.from_numpy(new_z).to(device=self.device)

        samples_to_cam = torch.cat((samples_x, samples_y, samples_z, torch.ones_like(samples_x).to(self.device)),
                                   dim=-1).float()

        cam_to_robot = torch.eye(4).to(self.device)
        cam_to_robot[:3, -1] = torch.tensor(view_cfg.position).to(self.device)

        # samples_to_robot = (cam_to_robot @ samples_to_cam.T).T

        samples_to_robot = (samples_to_cam @ cam_to_robot.T)
        if first_spawn:
            # no root states are available 
            ball_to_world = samples_to_robot[:, :, :3] + torch.tensor(
                [0, 0, self.base_init_state[2]], device=self.device
            )

            return torch.cat(
                (ball_to_world, torch.zeros_like(ball_to_world, device=self.device)),
                dim=-1,
            )

        # FIXME: won't work for more than one environment. Just picked the first one 
        ball_to_world = quat_rotate(self.root_states[self.robot_actor_idxs[env_ids], 3:7].repeat(batch_size, 1),
                                    samples_to_robot[0, :, :3]) + \
                        self.root_states[self.robot_actor_idxs[env_ids], :3]

        # pick one that is positive Z
        ball_to_world = ball_to_world[torch.where(ball_to_world[:, 2] > 0)[0][0], :].unsqueeze(0)

        self.ball_speeds[env_ids] = torch.zeros(
            (len(env_ids),), device=self.device
        )

        ball_state = torch.cat(
            [ball_to_world, torch.zeros((len(env_ids), 1), device=self.device)], dim=1)

        return ball_state

    def step_external_assets(self, force_ball_resample=False, **kwargs):
        # stop the robot when it is close enough

        sample_idxs = torch.where(self.episode_length_buf % self.cfg.ball.resampling_time == 0)[0]

        if force_ball_resample:
            sample_idxs = torch.arange(self.num_envs, device=self.device)

        if len(sample_idxs) > 0:
            # ball_states = self.get_ball_resets(sample_idxs)

            # FIXME: For more than one ball, this will need to be changed

            ball_states = self.get_ball_resets_batched(sample_idxs, batch_size=20)
            self.sample_count += 1
            self.triggered_flag = False

            count = 0
            # print('resetting ball')
            while not torch.any(ball_states[:, 2] > 0):
                print("ball is under the ground, resampling")
                ball_states = self.get_ball_resets_batched(sample_idxs)
                count += 1
                if count > 10:
                    raise RuntimeError("Soemthing bad happened, ball is under the ground for 100+ samples in a row")

            self._resample_commands(sample_idxs)

            self.root_states[self.ball_actor_idxs[sample_idxs], :3] = ball_states[:, :3]
            self.root_states[self.ball_actor_idxs[sample_idxs], 7:10] = ball_states[:, 3:]

            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(self.ball_actor_idxs[sample_idxs]),
                len(sample_idxs),
            )

        elif self.cfg.ball.stopping_distance is not None:
            stop_idxs = torch.where(torch.norm(self.target_pos_rel, dim=1) < self.cfg.ball.stopping_distance)[0]
            stop_idxs = stop_idxs & torch.where(self.episode_length_buf % self.cfg.ball.resampling_time > 5)[0]
            if len(stop_idxs) > 0:
                self.commands[stop_idxs, 0] = 0.0
                if not self.triggered_flag:
                    self.triggered_flag = True
                    self.reach_goal_count += 1
                    print(f"Reached goal {self.reach_goal_count}")
                    print('stopping!')

    def _reset_external_assets(self, env_ids):
        self.update_ball_levels(env_ids)

        # FIXME: For more than one ball, this will need to be changed

        next_ball_states = self.get_ball_resets_batched(env_ids, batch_size=20)

        count = 0
        while not torch.any(next_ball_states[:, 2] > 0):
            print("ball is under the ground, resampling")
            next_ball_states = self.get_ball_resets_batched(env_ids)
            count += 1
            if count > 10:
                raise RuntimeError("Soemthing bad happened, ball is under the ground for 100+ samples in a row")

        self.root_states[self.ball_actor_idxs[env_ids], :3] = next_ball_states[:, :3]
        self.root_states[self.ball_actor_idxs[env_ids], 7:10] = next_ball_states[:, 3:]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(self.ball_actor_idxs[env_ids]),
                                                     len(env_ids))

        # self.extras["episode"]["mean_ball_env_levels"] = self.ball_env_levels.to(torch.float32).mean()
