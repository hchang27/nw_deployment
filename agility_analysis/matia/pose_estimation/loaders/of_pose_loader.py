from torchtyping import TensorType

from agility_analysis.matia.pose_estimation.loaders.pose_loader import PoseLoader
from torchvision.models.optical_flow import raft_large

import math

import torchvision.transforms as T

import torch


class OFPoseLoader(PoseLoader):
    """
    data: image + OF pairs, 
    target: gt transformation 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flow_model = raft_large(pretrained=True, progress=False).to(self.device)
        self.flow_model.eval()

        self.raft_preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=RunArgs.resize_to), # no resize for now FIXME 
        ])

        self.raft_postprocess = T.Compose([
            T.Resize(size=(45, 80)),
            T.Normalize(mean=0.5, std=0.5),
        ])

    def _get_flow_maps(self, batch_idxs: TensorType["batch", "2", "torch.uint8"], skip_normalize=False) -> TensorType[
        "batch", "channel", "resize_height", "resize_width", torch.float32]:
        source_imgs = self.data[batch_idxs[:, 0]]
        target_imgs = self.data[batch_idxs[:, 1]]

        # preprocess for RAFT 
        source_imgs = self.raft_preprocess(source_imgs).contiguous().to(self.device)
        target_imgs = self.raft_preprocess(target_imgs).contiguous().to(self.device)

        assert source_imgs.shape == target_imgs.shape, "source and target images should have the same shape."

        # to avoid out of memory
        # flows = []
        # num_chunks = math.ceil(len(source_imgs) / 8)

        with torch.inference_mode():
            flows = self.flow_model(source_imgs, target_imgs)[-1]

            # normalize here. For now just going to divide by image size
            if not skip_normalize:
                width = source_imgs.shape[-1]
                height = source_imgs.shape[-2]

                flows[:, 0] /= width
                flows[:, 1] /= height

        return flows

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

            flow_maps = self.raft_postprocess(
                self._get_flow_maps(batch_inds).to(self.device, non_blocking=True)).contiguous()

            targets = self.get_gt_transforms(batch_inds).to(self.device, non_blocking=True).contiguous().float()

            images = images.to(self.device, non_blocking=True).contiguous()
            targets = targets.to(self.device, non_blocking=True).contiguous().float()

            if self.transform is not None:
                images = self.transform(images)

            yield images, flow_maps, targets


if __name__ == '__main__':
    from agility_analysis.matia.pose_estimation.main import TrainCfg
    from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig
    from matplotlib import pyplot as plt

    TrainCfg.trajectories_per_scene = 1
    # TrainCfg.batch_size = 4

    loader = OFPoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__)
    x = next(loader.sample_batch(TrainCfg.batch_size, True))
