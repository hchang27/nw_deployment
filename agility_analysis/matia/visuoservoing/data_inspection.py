from asyncio import sleep

import pandas as pd
from params_proto import ParamsProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Arrow


def batched_quat_rotate_inverse_np(q, v):
    q_w = q[:, :, -1:]
    q_vec = q[:, :, :3]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0

    prod = q_vec[:, :, None, :] @ v[:, :, :, None]

    c = 2 * q_vec * prod[:, :, :, 0]

    return a - b + c


def vector_to_euler(vectors):
    rolls = np.arctan2(vectors[:, 1], vectors[:, 0])
    pitches = np.arcsin(vectors[:, 2])
    yaws = np.arctan2(np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2), vectors[:, 2])
    return rolls, pitches, yaws


def vector_to_intrinsic_euler(vec):
    norm_vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    rolls = np.arctan2(norm_vec[:, 1], norm_vec[:, 0])

    pitches = np.arcsin(norm_vec[:, 2])
    yaws = np.arctan2(
        np.sqrt(norm_vec[:, 0] ** 2 + norm_vec[:, 1] ** 2), norm_vec[:, 2]
    )
    return np.stack([rolls, pitches, yaws]).T


from pathlib import Path


class Args(ParamsProto, cli=False):
    rollout_path = Proto(
        env="$DATASETS/lucidsim/ball/rollout_0000.pkl",
        help="path to the rollout file. Casted to a pathlib.Path object.",
        dtype=Path,
    )



print("Loading from file://" + Args.rollout_path.parent.__str__())

import numpy as np


from ml_logger import ML_Logger
import os

loader = ML_Logger(root=os.getcwd())

(data,) = loader.load_pkl(Args.rollout_path)
df = pd.DataFrame(data)

# the shape is [batch, 3]
ball_location = np.stack(df["ball_location"].tolist())  # world frame location

# the shape is [batch, time, obs_dim]
robot_states = np.stack(df["states"].tolist())  # world frame location
robot_obs = np.stack(df["obs"].tolist())[:, :, 0, :]  # observations

# the shape is [batch, time, 3]
robot_com = robot_states[:, :, :3]

robot_quat = robot_states[:, :, 3:7]

ball_to_robot = batched_quat_rotate_inverse_np(robot_quat, ball_location[:, None, :])
target_heading = ball_to_robot - robot_com


app = Vuer()

points = robot_com.reshape(-1, 3)
points -= points[0]
print(f"points.shape: {points.shape}")

heading_angles = vector_to_intrinsic_euler(target_heading.reshape(-1, 3))

scale = np.linalg.norm(target_heading.reshape(-1, 3), axis=-1)
delta_yaw = robot_obs[:, :, 6].reshape(-1)


@app.spawn(start=True)
async def visualize(sess: VuerSession):
    sess.set @ DefaultScene()
    await sleep(0.1)

    for i in range(len(points)):
        await sleep(0.001)
        if i % 10 != 0:
            continue

        sess.upsert @ Arrow(
            position=[0.01 * i, 0, 0],
            direction=[0, float(delta_yaw[i]), 0],
            key=f"heading-{i}",
        )
    while True:
        await sleep(1)
