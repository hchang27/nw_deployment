import math
import os
from typing import TYPE_CHECKING, List, Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchtyping import TensorType
from tqdm import tqdm

from lucidsim_old.utils import center_crop

if TYPE_CHECKING:
    from agility_analysis.matia.go1.main import TrainCfg


class LucidDreams:
    def __init__(
            self,
            tensor_size,
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

                images = self.load_images(scene, traj_idx)

                (rollout,) = self.loader.load_pkl(traj)
                self.obs_data.extend(rollout["obs"])
                self.targets.extend(rollout["actions"])

                data.extend(images)

        self.obs_data = np.concatenate(self.obs_data, axis=0)
        self.obs_data: TensorType["timesteps", "num_obs"] = torch.from_numpy(self.obs_data)

        data = torch.stack(data, dim=0)

        DataT = TensorType["timesteps", "channel", "height", "width"]
        assert data.dtype == torch.uint8
        self.data: DataT = data.contiguous()
        self.targets = np.concatenate(self.targets, axis=0)
        self.targets: TensorType["timesteps", "num_actions"] = torch.from_numpy(self.targets)

        self.indices = torch.arange(len(self.data))

    def load_images(self, scene, traj_num, use_optical_flows=False):
        """
        Load the images for a given trajectory
        """
        loader = self.loader

        if use_optical_flows:
            img_mask = f"stacked_lucid_dreams/imagen_{traj_num:04d}*/*.png"
        else:
            if "augmented" not in self.image_type:
                img_mask = f"ego_views/{self.image_type}_{traj_num:04d}/*.png"
            else:
                parsed = self.image_type.split("_")
                if len(parsed) == 1:
                    img_mask = f"lucid_dreams/imagen_{traj_num:04d}*/*.png"
                elif len(parsed) == 2:
                    trial_id = int(parsed[1])
                    img_mask = f"lucid_dreams/imagen_{traj_num:04d}_{trial_id:04d}/*.png"
                else:
                    raise ValueError(f"Invalid image type {self.image_type}")

        image_files = self.loader.glob(img_mask, wd=scene)
        image_files = sorted(image_files)
        images = []

        resize_tf = T.Resize(self.tensor_size, interpolation=T.InterpolationMode.BILINEAR)

        with loader.Prefix(scene):
            for f in tqdm(image_files, desc=f"Loading {scene} {traj_num}"):
                buff = loader.load_file(f)
                img = np.array(Image.open(buff))

                if self.crop_image_size is not None:
                    img = center_crop(img, *self.crop_image_size[::-1])
                    
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)

                img = torch.from_numpy(img).permute(2, 0, 1)

                if self.gpu_load:
                    # resize and let it sit on the GPU as a uint8
                    img = resize_tf(img).to(self.device, non_blocking=True).contiguous()

                images.append(img)

        return images

    def sample_batch(self, batch_size, shuffle=True):
        """Batch Iterator for the dataset.
        
        Returns: batches of image, observation, and action label data.
        """
        if shuffle:
            self.indices = torch.randperm(len(self.data))

        n_chunks = math.ceil(len(self.data) / batch_size)
        for batch_inds in torch.chunk(self.indices, n_chunks):
            images = self.data[batch_inds]
            obs = self.obs_data[batch_inds]
            targets = self.targets[batch_inds]

            images = images.to(self.device, non_blocking=True).contiguous()
            obs = obs.to(self.device, non_blocking=True).contiguous().float()
            targets = targets.to(self.device, non_blocking=True).contiguous().float()

            if self.image_type == "depth" and self.normalize_depth:
                min = images.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0][..., None, None, None]
                max = images.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0][..., None, None, None]
                images = (images - min) / (max - min + 1e-3) * 255
                images = images.to(torch.uint8)

            if self.transform is not None:
                images = self.transform(images)

            yield images, obs, targets


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
