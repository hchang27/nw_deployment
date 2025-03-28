from random import randint
from typing import Dict, List

import numpy as np
import torch


class Buffer:

    def __init__(self, trajectories: List[Dict[str, torch.Tensor]],
                 batch_size: int, K: int, obs_dim: int, act_dim: int,
                 max_ep_len: int, rtg_scale: float, states_mean: torch.Tensor,
                 states_std: torch.Tensor, device: str):
        """
        we pin the memory of these parameters, therefore the batch_size
        should be fixed.

        self.batch_size == 64

        states Size([64, 20, 11]) float32
        actions Size([64, 20, 3]) float32
        rewards Size([64, 20, 1]) float32
        dones Size([64, 20]) int64
        rtg Size([64, 21, 1]) float32
        timesteps Size([64, 20]) int64
        attention_mask Size([64, 20]) float64

        Args:
            batch_size ():
            K ():
            obs_dim ():
            act_dim ():
            max_ep_len (): used to compute the arange indices
            device ():
        """
        self.batch_size = batch_size
        self.K = K
        self.device = device

        self.num_trajectories = len(trajectories)
        self.range_trajs = np.arange(self.num_trajectories)
        self.traj_lengths = np.array([len(traj['observations']) for traj in trajectories])

        max_ep_len = max_ep_len or max(self.traj_lengths)

        # add padding to K because the rolling context window
        self.range_inds = torch.arange(max_ep_len).to(device)
        # self.range_inds = torch.arange(max_ep_len + K).to(device)
        # self.range_inds[max_ep_len:] = max_ep_len - 1
        self.trajectories = []
        for traj in trajectories:
            traj_copy = traj.copy()
            observations = traj_copy.pop('observations')
            new_traj = {"observations": (observations - states_mean) / states_std, **traj_copy}
            self.trajectories.append(new_traj)

        # declare buffers
        self.s = torch.zeros(batch_size, K, obs_dim).to(device)
        self.a = torch.full([batch_size, K, act_dim], -10.).to(device)
        self.r = torch.zeros(batch_size, K, 1).to(device)
        self.d = torch.full([batch_size, K], 2, dtype=torch.long).to(device)
        # the return to go is off by one step.
        self.rtg = torch.full([batch_size, K + 1, 1], 1 / rtg_scale).to(device)
        self.timesteps = torch.zeros(batch_size, K, dtype=torch.long).to(device)
        self.mask = torch.zeros(batch_size, K).to(device)

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = self.traj_lengths / self.traj_lengths.sum()

    def get_batch(self):
        batch_inds = np.random.choice(
            self.range_trajs,
            size=self.batch_size,
            replace=True,
            p=self.p_sample,  # re-weights so we sample according to timesteps
        )
        for i, ind in enumerate(batch_inds):
            # for i, ind in tqdm(enumerate(batch_inds), desc="collate batch", leave=True):
            traj = self.trajectories[ind]

            S = traj['observations']
            A = traj['actions']
            R = traj['rewards']
            DONES = traj['terminals']
            RTG = traj['return-to-go']

            l = self.traj_lengths[ind]
            start = randint(0, l - 1)
            end = start + self.K
            chunk_l = min(end, l) - start

            self.s[i, -chunk_l:] = S[start:end]
            self.a[i, -chunk_l:] = A[start:end]
            self.r[i, -chunk_l:] = R[start:end, None]
            self.d[i, -chunk_l:] = DONES[start:end]
            self.rtg[i, -chunk_l - 1:] = RTG[start:end + 1, None]
            # this needs to be handled spatially because range_inds include placeholders.
            self.timesteps[i, -chunk_l:] = self.range_inds[start:start + chunk_l]
            self.mask[i, -chunk_l:] = 1

            self.s[i, :-chunk_l] = 0
            self.a[i, :-chunk_l] = -10
            self.r[i, :-chunk_l] = 0
            self.d[i, :-chunk_l] = 2
            self.rtg[i, :-chunk_l - 1] = 0
            self.timesteps[i, :-chunk_l] = 0
            self.mask[i, :-chunk_l] = 0

        return self.s, self.a, self.r, self.d, self.rtg, self.timesteps, self.mask
