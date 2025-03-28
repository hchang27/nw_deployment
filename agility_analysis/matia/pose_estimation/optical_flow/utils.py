import math
import torch
from more_itertools import chunked
from torchtyping import TensorType
from torchvision.utils import flow_to_image
import torch.nn.functional as F
from tqdm import tqdm


def compute_flow(source_batch: TensorType["batch", "channel", "height", "width"],
                 target_batch: TensorType["batch", "channel", "height", "width"],
                 raft_model):
    flow = raft_model(target_batch, source_batch)[-1]
    return flow


def warp_forward(x, flo):
    """
    warp an image/tensor (im1) to im2, according to the optical flow
    x: [B, C, H, W] (im1)
    flo: [B, 2, H, W] flow (1 -> 2)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # Invert the flow by multiplying by -1
    vgrid = grid - flo  # Change here
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size(), device=x.device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def warp_batch(
        source: TensorType["batch", "channel", "height", "width"],
        flow: TensorType["batch", "2", "height", "width"], ):
    """
    Important: flow should not be normalized. It should remain in the pixel space. 
    """
    warped = warp_forward(source, flow)
    return warped


def compute_flow_chunked(
        loader,
        idx_pairs: TensorType["batch", "2"],
        chunk_size: int = 8,
):
    flows = []
    for i, idx_pairs_chunk in tqdm(enumerate(chunked(idx_pairs, chunk_size)), desc="Generating flow images..."):
        pairs = torch.stack(idx_pairs_chunk)
        flow_map = loader._get_flow_maps(pairs, skip_normalize=True).cpu()
        flows.append(flow_map)

    return torch.cat(flows, dim=0)


def consistent_dream(
        rgb_loader,
        dream_loader,
        logger,
        step,
        chunk_size=4,
        save_video=False,
):
    rgb_length = len(rgb_loader.data)
    dream_length = len(dream_loader.data)

    assert rgb_length == dream_length, "rgb and dream should have the same length"

    starting_idxs = torch.arange(0, rgb_length - step + 1, step=step)

    # compute the flow between starting idx and starting idx + 1
    idx_pairs = torch.stack([starting_idxs, starting_idxs + 1], dim=1)

    for i in range(step):
        if i == 0:
            warped = rgb_loader.raft_preprocess(dream_loader.data)[idx_pairs[:, 0]]
        else:
            flow = compute_flow_chunked(rgb_loader, idx_pairs, chunk_size=chunk_size)
            warped = warp_batch(warped, flow.cpu())
            idx_pairs += 1

        for j, img in enumerate(warped):
            logger.save_image(0.5 * (1 + img.permute(1, 2, 0)), f"frame_{i + step * j:04d}.png")

    if save_video:
        filenames = sorted(logger.glob("frame_*.png"))

        print("Saving video...")

        logger.make_video(filenames, "flow.mp4", fps=50)
        print("Saved images to", logger.get_dash_url())


if __name__ == '__main__':
    from agility_analysis.matia.pose_estimation.loaders.pose_loader import PoseLoader
    from agility_analysis.matia.pose_estimation.loaders.of_pose_loader import OFPoseLoader
    from agility_analysis.matia.pose_estimation.main import TrainCfg
    from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig
    import matplotlib.pyplot as plt
    from ml_logger import logger

    TrainCfg.trajectories_per_scene = (0, 1)
    TrainCfg.scenes = ["mit_stairs/stairs_0001_v1"]

    TrainCfg.image_type = "rgb"
    rgb_loader = OFPoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__)

    TrainCfg.image_type = "augmented"
    dream_loader = PoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__)

    consistent_dream(rgb_loader, dream_loader, step=25, chunk_size=4, log=True)
    exit()
    traj_length = 647

    step = 50
    starting_idxs = torch.arange(0, traj_length - step + 1, step=step)

    # compute the flow between starting idx and starting idx + 1
    idx_pairs = torch.stack([starting_idxs, starting_idxs + 1], dim=1)

    for i in range(step):
        if i == 0:
            warped = processed_dream = rgb_loader.raft_preprocess(dream_loader.data)[idx_pairs[:, 0]]
        else:
            flow = compute_flow_chunked(rgb_loader, idx_pairs, chunk_size=4)
            warped = warp_batch(warped, flow.cpu())
            idx_pairs += 1

        for j, img in enumerate(warped):
            logger.save_image(0.5 * (1 + img.permute(1, 2, 0)), f"flows/warped_{i + step * j:04d}.png")

    filenames = sorted(logger.glob("flows/warped_*.png"))
    len(filenames)

    print("Saving video...")

    logger.make_video(filenames, "flows.mp4", fps=50)
    print("Saved images to", logger.get_dash_url())
