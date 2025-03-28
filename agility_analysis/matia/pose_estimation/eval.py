import asyncio

from vuer import Vuer
from vuer.schemas import DefaultScene, Sphere
from agility_analysis.matia.pose_estimation.main import TrainCfg, Go1RGBConfig
from agility_analysis.matia.pose_estimation.loaders.pose_loader import PoseLoader
import torchvision.transforms.v2 as T
import torch

from lucidsim_old.utils import quat_from_euler_xyz, quat_conjugate, quat_mul, quat_rotate, euler_from_quaternion

app = Vuer()

pipeline = [
    T.ToImage(),
    T.ToDtype(torch.float32),
    T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
]
transform = T.Compose([
    T.Resize((60, 96), interpolation=T.InterpolationMode.BILINEAR),
    # T.RandomCrop((60, 96), padding=4),
    *pipeline
])

TrainCfg.trajectories_per_scene = 1
loader = PoseLoader(**TrainCfg.__dict__, **Go1RGBConfig.__dict__, transform=transform)

traj_length = 647

idx_pairs = torch.arange(traj_length)
idx_pairs = torch.stack([idx_pairs[:-1], idx_pairs[1:]], dim=1)

gt_transforms = loader.get_gt_transforms(idx_pairs)

images = loader.data[idx_pairs].to(loader.device, non_blocking=True).contiguous()
images_tf = transform(images)


def accumulate_deltas(deltas):
    """
    Accumulate deltas to get the absolute pose. Deltas are the pose of the previous frame relative to the next frame.
    """

    dpos, drotation = deltas[:, :3], deltas[:, 3:]
    dquat = quat_from_euler_xyz(drotation[:, 0], drotation[:, 1], drotation[:, 2])

    current_pos = torch.zeros(3)[None, ...]  # Initialize current position
    current_rot = torch.tensor([0., 0., 0., 1.])[None, ...]  # Initialize current rotation (identity quaternion)

    positions = [current_pos.clone()]
    rotations = [current_rot.clone()]
    for dp, dq in zip(dpos, dquat):
        # Update rotation: Quaternion multiplication of current_rot and the conjugate of dq
        current_rot = quat_mul(current_rot, quat_conjugate(dq[None, ...]))

        current_pos = quat_rotate(current_rot, -quat_rotate(quat_conjugate(dq[None, ...]), dp[None, ...])) + current_pos

        positions.append(current_pos.clone())
        rotations.append(current_rot.clone())

    return positions, rotations


gt_positions, _ = accumulate_deltas(gt_transforms)
print(gt_positions[::10])

from ml_logger import logger

model = logger.torch_load("/alanyu/scratch/2024/01-23/191928/checkpoints/net_last.pts")

with torch.no_grad():
    deltas = []
    image_stacked = torch.cat([images_tf[:, 0, 0:1], images_tf[:, 1, 0:1]], dim=1)
    results = model(image_stacked) / 100
    deltas = results.cpu()

estimated_positions, _ = accumulate_deltas(deltas)


import numpy as np
print(estimated_positions[::10])
print(np.array(estimated_positions[::2]) - np.array(gt_positions[::2]))

@app.spawn(start=True)
async def main(sess):
    sess.set @ DefaultScene(
        # Sphere(args=[0.05, 32, 32], material=dict(color="red")),
    )
    # 
    # for position in gt_positions[::10]:
    #     sess.upsert @ Sphere(args=[0.05, 32, 32], position=position.tolist()[0], material=dict(color="red"))

    for position in estimated_positions[::10]:
        sess.upsert @ Sphere(args=[0.05, 32, 32], position=position.tolist()[0], material=dict(color="blue"))

    while True:
        await asyncio.sleep(0.1)
