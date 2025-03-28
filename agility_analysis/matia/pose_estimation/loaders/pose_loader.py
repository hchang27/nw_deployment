import warnings

import numpy as np

import math
from torchtyping import TensorType
from typing import List, Literal
import os

from tqdm import tqdm
from PIL import Image
from agility_analysis.matia.go1.dataset import LucidDreams
from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig
import torch

import itertools

from lucidsim_old.utils import quat_rotate, euler_from_quaternion, quat_mul, quat_conjugate


class PoseLoader:

    def __init__(
            self,
            root,
            prefix,
            scenes: List[str],
            # sampling
            trajectories_per_scene,
            image_type: Literal["rgb", "depth", "augmented"],
            # system
            device,
            transform=None,
            camera_to_base_pos=None,
            lookback=(2, 3),  # how many frames to look back; inclusive
            # catchall
            **_
    ):
        from ml_logger import ML_Logger

        self.loader = ML_Logger(root=root, prefix=prefix)

        self.transform = transform

        self.device = device

        if isinstance(trajectories_per_scene, int):
            trajectories_per_scene = (0, trajectories_per_scene)
        trajectories = range(*trajectories_per_scene)
        self.image_type = image_type

        data = []

        state_info = []

        for scene in scenes:
            trajectory_subfolder = os.path.join(scene, "trajectories")
            traj_filelist = sorted(self.loader.glob(os.path.join(trajectory_subfolder, "trajectory*.pkl")))
            for traj_idx in trajectories:
                traj = traj_filelist[traj_idx]
                (rollout,) = self.loader.load_pkl(traj)
                state_info.extend(rollout["states"])
                traj_num = int(traj.split(".")[0].split("_")[-1])
                data.extend(self.load_images(scene, traj_num))

        state_info = torch.from_numpy(np.stack(state_info, axis=0)[:, :7])
        pos, quat = state_info[:, :3], state_info[:, 3:]

        pos = pos.float()
        quat = quat.float()

        # get camera tfs, if being used for pose estimation 
        if camera_to_base_pos is not None:
            cam_positions = pos + quat_rotate(quat, torch.tensor(camera_to_base_pos).repeat(len(pos), 1))
            roll, pitch, yaw = euler_from_quaternion(quat)

            self.poses = torch.cat([cam_positions, quat], dim=1)

        resize_count = 0
        shape = data[0].shape
        for i, img in enumerate(data):
            if img.shape != shape:
                img = np.array(Image.fromarray(img).resize(shape[:2][::-1]))
                resize_count += 1
                data[i] = img
        warnings.warn(f"Resized {resize_count} images")

        data = np.stack(data, axis=0)
        data = torch.from_numpy(data)

        DataT = TensorType["timesteps", "channel", "height", "width"]
        self.data: DataT = torch.permute(data, [0, 3, 1, 2]).contiguous()

        self.idx_pairs = torch.tensor(
            self.create_index_pairs(len(self.data), lookback))  # note: the order is accounted for

    def create_index_pairs(self, buffer_size, lookback):
        """
        Create pairs of indexes for training, where the difference between indexes does not exceed 'lookback'.

        Parameters:
        buffer_size (int): The size of the data buffer.
        lookback (int): The maximum allowed difference between indexes in a pair.

        Returns:
        list of tuples: A list containing pairs of indexes.
        """
        indices = range(buffer_size)
        all_pairs = itertools.product(indices, repeat=2)
        filtered_pairs = [pair for pair in all_pairs if lookback[0] <= abs(pair[0] - pair[1]) <= lookback[1]]

        # randomize and pick the first 50 pairs
        # np.random.shuffle(filtered_pairs)
        # filtered_pairs = filtered_pairs[:500]

        return filtered_pairs

    def load_images(self, scene, traj_num):
        """
        Load the images for a given trajectory
        """
        loader = self.loader

        img_mask = f"ego_views/{self.image_type}_{traj_num:04d}/*.png" if self.image_type != "augmented" else f"lucid_dreams/imagen_{traj_num:04d}*/*.png"
        image_files = self.loader.glob(img_mask, wd=scene)
        image_files = sorted(image_files)
        images = []

        with loader.Prefix(scene):
            for f in tqdm(image_files, desc=f"Loading {scene} {traj_num}"):
                buff = loader.load_file(f)
                img = np.array(Image.open(buff))
                images.append(img)

        return images

    def get_gt_transforms(self, idx_pairs):
        """
        Returns TensorType["batch", "6"] representing the relative transformations. 
        """
        # Extract rotations and positions for each pair
        rot_1 = self.poses[idx_pairs[:, 0], 3:]
        pos_1 = self.poses[idx_pairs[:, 0], :3]
        rot_2 = self.poses[idx_pairs[:, 1], 3:]
        pos_2 = self.poses[idx_pairs[:, 1], :3]

        rel_rot = quat_mul(quat_conjugate(rot_2), rot_1)

        # Compute the relative position
        pos_diff = pos_1 - pos_2
        rel_pos = quat_rotate(quat_conjugate(rot_2), pos_diff)

        # Convert to euler angles
        roll, pitch, yaw = euler_from_quaternion(rel_rot)

        # rel_pos = pos_1_to_2 = quat_rotate(quat_conjugate(rot_2), pos_1) - pos_2

        # rot_1_to_2 = quat_mul(quat_conjugate(rot_2), rot_1)
        # roll, pitch, yaw = euler_from_quaternion(rot_1_to_2)

        return torch.cat([rel_pos, roll[..., None], pitch[..., None], yaw[..., None]], dim=1)
        # return torch.cat([rel_pos, rel_rot], dim=1)

    def sample_batch(self, batch_size, shuffle=True):
        """Batch Iterator for the dataset.

        Returns: batches of image, observation, and action label data.
        """
        if shuffle:
            shuffle_idxs = torch.randperm(len(self.idx_pairs))
            self.idx_pairs = self.idx_pairs[shuffle_idxs]

        n_chunks = math.ceil(len(self.idx_pairs) / batch_size)

        self.poses = self.poses.to(self.device, non_blocking=True).contiguous().float()

        for batch_inds in torch.chunk(self.idx_pairs, n_chunks):

            # compute gt transforms 
            images = self.data[batch_inds].to(self.device, non_blocking=True).contiguous()
            targets = self.get_gt_transforms(batch_inds).to(self.device, non_blocking=True).contiguous().float()

            images = images.to(self.device, non_blocking=True).contiguous()
            targets = targets.to(self.device, non_blocking=True).contiguous().float()

            if self.transform is not None:
                images = self.transform(images)

            yield images, targets


if __name__ == '__main__':
    from agility_analysis.matia.pose_estimation.main import TrainCfg

    TrainCfg.trajectories_per_scene = 1
    loader = PoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__)
    x = next(loader.sample_batch(TrainCfg.batch_size, True))

    x[1][:, 3:] * 100
    x[0].shape, x[1].shape

    x[1]
