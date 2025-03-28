import math
import os
from typing import TYPE_CHECKING, List, Literal

import numpy as np
import torch
from torchtyping import TensorType

if TYPE_CHECKING:
    from agility_analysis.matia.go1.main import TrainCfg


class Scandots:
    def __init__(
            self,
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

        self.loader = ML_Logger(root=local_dataset_root)
        self.normalize_depth = normalize_depth

        self.transform = transform

        self.device = device

        trajectories = trajectories_per_scene

        self.crop_image_size = crop_image_size

        self.image_type = image_type

        self.gpu_load = gpu_load

        self.obs_data = []
        data = []
        self.targets = []

        for i, scene in enumerate(scenes):
            trajectory_subfolder = os.path.join(scene, "trajectories")
            traj_filelist = sorted(self.loader.glob(os.path.join(trajectory_subfolder, "trajectory*.pkl")))
            for traj_idx in trajectories[i]:
                traj = traj_filelist[traj_idx]
                traj_num = int(traj.split(".")[0].split("_")[-1])

                # images = self.load_images(scene, traj_num)

                (rollout,) = self.loader.load_pkl(traj)
                self.obs_data.extend(rollout["obs"])
                self.targets.extend(rollout["actions"])

                # data.extend(images)

        self.obs_data = np.concatenate(self.obs_data, axis=0)
        self.obs_data: TensorType["timesteps", "num_obs"] = torch.from_numpy(self.obs_data)

        # data = np.stack(data, axis=0)
        # data = torch.from_numpy(data)

        # DataT = TensorType["timesteps", "channel", "height", "width"]
        # should be uint8. perform depth normlization during batch sampling as a preprocessing step
        # self.data: DataT = torch.permute(data, [0, 3, 1, 2]).contiguous()

        self.targets = np.concatenate(self.targets, axis=0)
        self.targets: TensorType["timesteps", "num_actions"] = torch.from_numpy(self.targets)

        self.indices = torch.arange(len(self.targets))

    def sample_batch(self, batch_size, shuffle=True):
        """Batch Iterator for the dataset.
        
        Returns: batches of image, observation, and action label data.
        """
        if shuffle:
            self.indices = torch.randperm(len(self.targets))

        n_chunks = math.ceil(len(self.targets) / batch_size)
        for batch_inds in torch.chunk(self.indices, n_chunks):
            # images = self.data[batch_inds]
            obs = self.obs_data[batch_inds]
            targets = self.targets[batch_inds]

            # images = images.to(self.device, non_blocking=True).contiguous().float()
            obs = obs.to(self.device, non_blocking=True).contiguous().float()

            batch_size = len(obs)
            scandots = obs[..., 53:53 + 132].reshape(batch_size, 11, 12)
            # upsample to (45, 80)
            scandots = torch.nn.functional.interpolate(scandots.unsqueeze(1), size=(45, 80), mode="bilinear")

            targets = targets.to(self.device, non_blocking=True).contiguous().float()

            # if self.transform is not None:
            #     images = self.transform(images)

            yield scandots, obs, targets


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from agility_analysis.matia.go1.main_ball_b2 import TrainCfg

    # local_dataset_root: str = os.path.join(os.environ["DATASETS"], "lucidsim", "scenes", "experiments", "real2real")
    # scenes = ["ball/scene_00002"]
    local_dataset_root = os.path.join(os.environ["DATASETS"], "lucidsim", "scenes")
    scenes = ["cone-debug/scene_00001"]
    trajectories_per_scene = [[0]]

    image_type: Literal["rgb", "depth", "augmented"] = "augmented"

    dataset = LucidDreams(local_dataset_root=local_dataset_root,
                          scenes=scenes,
                          trajectories_per_scene=trajectories_per_scene,
                          image_type=image_type,
                          device=TrainCfg.device,
                          transform=None,
                          crop_image_size=TrainCfg.crop_image_size,
                          normalize_depth=True,
                          gpu_load=False,
                          )

    x = next(dataset.sample_batch(TrainCfg.batch_size, True))

    images, obs, targets = x
    yaws = obs[:, 6:7]

    for i in range(10):
        plt.imshow(images.cpu()[i].permute(1, 2, 0))
        # title with the target label
        plt.title(yaws[i].cpu().item())
        plt.show()
        print("----")
