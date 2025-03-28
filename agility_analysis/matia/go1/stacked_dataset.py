import os
from typing import TYPE_CHECKING, List, Literal

import math
import numpy as np
import torch
from torchtyping import TensorType

from agility_analysis.matia.go1.dataset import LucidDreams

if TYPE_CHECKING:
    from agility_analysis.matia.go1.main import TrainCfg


class StackedLucidDreams(LucidDreams):
    def __init__(
        self,
        tensor_size,
        stack_size: int,
        compute_deltas: bool,
        local_dataset_root: str,
        scenes: List[str],
        # sampling
        trajectories_per_scene,
        image_type: Literal["rgb", "depth", "augmented"],
        # system
        device,
        transform=None,
        crop_image_size=None,
        normalize_depth: bool = True,
        gpu_load=False,
        # catchall
        **_
    ):
        from ml_logger import ML_Logger

        self.tensor_size = tensor_size

        self.loader = ML_Logger(root=local_dataset_root)
        self.normalize_depth = normalize_depth

        self.transform = transform

        self.device = device

        trajectories = trajectories_per_scene

        self.crop_image_size = crop_image_size

        self.image_type = image_type

        self.gpu_load = gpu_load

        self.compute_deltas = compute_deltas

        self.obs_data = []
        data = []
        self.targets = []

        for i, scene in enumerate(scenes):
            trajectory_subfolder = os.path.join(scene, "trajectories")
            traj_filelist = sorted(self.loader.glob(os.path.join(trajectory_subfolder, "trajectory*.pkl")))
            traj_nums = [int(traj.split(".")[0].split("_")[-1]) for traj in traj_filelist]
            for traj_idx in trajectories[i]:
                idx = traj_nums.index(traj_idx)
                traj = traj_filelist[idx]

                images = self.load_images(scene, traj_idx, False)  # use_optical_flows=(self.image_type == "augmented"))
                (rollout,) = self.loader.load_pkl(traj)

                if len(images) == 0:
                    raise RuntimeError(f"Could not find images for {scene} {traj_idx}")

                stack_cutoff = len(images) - len(images) % stack_size

                # self.obs_data.extend(rollout["obs"][:stack_cutoff:stack_size])
                # self.targets.extend(rollout["actions"][:stack_cutoff:stack_size])

                self.obs_data.extend(rollout["obs"][:stack_cutoff])
                self.targets.extend(rollout["actions"][:stack_cutoff])

                data.extend(images[:stack_cutoff])

        num_stack = int(len(data) / stack_size)

        self.obs_data = np.concatenate(self.obs_data, axis=0)
        self.obs_data: TensorType["num_stack", "num_obs"] = torch.from_numpy(self.obs_data)
        self.obs_data: TensorType["num_stack", "stack_size", "num_obs"] = self.obs_data.reshape(num_stack, stack_size,
                                                                                                -1)

        data = torch.stack(data, dim=0)
        data = data[:num_stack * stack_size]

        image_size = data.shape[-2:]

        DataT = TensorType["num_stack", "stack_size", "channel", "height", "width"]

        # Note: within the stack, the most recent image is at the end
        assert data.dtype == torch.uint8
        # self.data: DataT = torch.permute(data, [0, 3, 1, 2]).reshape(num_stack, stack_size, 3,
        #                                                              *image_size).contiguous()
        self.data: DataT = data.reshape(num_stack, stack_size, 3, *image_size).contiguous()

        self.targets = np.concatenate(self.targets, axis=0)
        self.targets: TensorType["num_stack", "num_actions"] = torch.from_numpy(self.targets)
        self.targets: TensorType["num_stack", "stack_size", "num_actions"] = self.targets.reshape(num_stack, stack_size,
                                                                                                  -1)
        # self.indices = torch.arange(len(self.data))

    def sample_batch(self, batch_size, shuffle=True):
        """Batch Iterator for the dataset.

        Returns: batches of image, observation, and action label data.
        """
        # if shuffle:
        #     self.indices = torch.randperm(len(self.data))
        self.indices = torch.arange((len(self.targets) // batch_size) * batch_size)

        if shuffle:
            self.indices = torch.randperm(len(self.indices))

        n_chunks = math.ceil(len(self.indices) / batch_size)
        for batch_inds in torch.chunk(self.indices, n_chunks):
            images = self.data[batch_inds]
            obs = self.obs_data[batch_inds]
            targets = self.targets[batch_inds]

            images = images.to(self.device, non_blocking=True).contiguous()
            obs = obs.to(self.device, non_blocking=True).contiguous().float()
            targets = targets.to(self.device, non_blocking=True).contiguous().float()

            if self.image_type == "depth" and self.normalize_depth:
                min = images.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0][..., None, None, None]
                max = images.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0][..., None, None, None]
                images = (images - min) / (max - min + 1e-3) * 255
                images = images.to(torch.uint8)

            if self.transform is not None:
                batch, stack, channels, height, width = images.shape
                images = self.transform(images.reshape(batch * stack, channels, height, width))

                height, width = images.shape[-2:]
                images = images.reshape(batch, stack, channels, height, width)

                # frame subtraction: compute the difference between the most recent frame and the previous frames
                if self.compute_deltas:
                    images[:, :-1] = images[:, -1:] - images[:, :-1]

                # plt.imshow(images[1, 0].cpu().permute(1, 2, 0));
                # plt.show()
            yield images, obs, targets


if __name__ == "__main__":
    from agility_analysis.matia.go1.main import TrainCfg

    dataset = StackedLucidDreams(stack_size=3, **vars(TrainCfg))

    # plt.imshow(self.data[11, 0].permute(1, 2, 0));
    # plt.show()
    # plt.imshow(self.data[11, 1].permute(1, 2, 0));
    # plt.show()
    # plt.imshow(self.data[11, 2].permute(1, 2, 0));
    # plt.show()

    x = next(dataset.sample_batch(TrainCfg.batch_size, True))
