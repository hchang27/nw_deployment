import math
import torch
from torchvision.utils import flow_to_image
from tqdm import tqdm

from agility_analysis.matia.pose_estimation.loaders.of_pose_loader import OFPoseLoader
from agility_analysis.matia.pose_estimation.main import TrainCfg
from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig
from matplotlib import pyplot as plt

from more_itertools import chunked
from ml_logger import logger

if __name__ == '__main__':
    TrainCfg.trajectories_per_scene = (0, 1)
    TrainCfg.scenes = ["mit_stairs/stairs_0001_v1"]
    # TrainCfg.batch_size = 4
    TrainCfg.image_type = "rgb"
    loader = OFPoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__)

    traj_length = 647

    step = 2

    idx_pairs = torch.arange(0, traj_length, step=step)
    idx_pairs = torch.stack([idx_pairs[:-1], idx_pairs[1:]], dim=1)

    chunk_size = 4
    num_chunks = math.ceil(len(idx_pairs) / chunk_size)
    count = 0
    # for i, idx_pair in tqdm(enumerate(idx_pairs), desc="Generating flow images..."):
    for i, idx_pairs_chunk in tqdm(enumerate(chunked(idx_pairs, chunk_size)), desc="Generating flow images..."):
        pairs = torch.stack(idx_pairs_chunk)
        flow_map = loader._get_flow_maps(pairs, skip_normalize=True).cpu()
        flow_img = flow_to_image(flow_map)
        for img in flow_img:
            logger.save_image(img.permute(1, 2, 0).numpy(), f"flows/flow_{count:04d}.png")
            count += 1

    files = sorted(logger.glob("flows/flow_*.png"))
    logger.make_video(files, "flows.mp4", fps=50 / step)

    print(f"Saved {len(files)} flow images to {logger.get_dash_url()}")
