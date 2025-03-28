import numpy as np

from vuer.schemas import Urdf, group, Sphere, Box
from torchtyping import TensorType
import torch
from typing import Tuple

ISAAC_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def get_three_mat(position, rotation):
    """
    Args:
        position: List[float] in world coordinates
        rotation: List[float] roll, pitch, yaw

    Returns: 16 item List[float] Three.js matrix in column major order
    """
    r, p, y = rotation
    mat = np.eye(4)
    mat[:3, 3] = position
    mat[:3, :3] = np.array(
        [
            [
                np.cos(y) * np.cos(p),
                np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r),
                np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r),
            ],
            [
                np.sin(y) * np.cos(p),
                np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r),
                np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r),
            ],
            [-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)],
        ]
    )

    extra_rot = np.array([[0.0, 0.0, -1], [-1, 0, 0.0], [0.0, 1.0, 0]])
    mat[:3, :3] = mat[:3, :3] @ extra_rot

    return mat.T.reshape(-1).tolist()


def sphere(key, position):
    return Sphere(
        key=key,
        position=position,
        args=[0.25, 20, 20],
        scale=1.0,
        material=dict(color="green", wireframe=True),
    )


def cube(key, position):
    return Box(
        key=key,
        args=[0.25, 0.25, 0.25, 40, 40, 40],
        position=position,
        material=dict(color="blue", wireframe=True),
    )


def Go1(src, joints, position=(0, 0, 0), global_rotation=(0, 0, 0), **kwargs):
    """

    Args:
        src: path to URDF (e.g.  "http://localhost:8013/static/urdf/go1.urdf")
        joints: dictionary from joint names to angles (rad)
        position: in world coordinates
        global_rotation: in rad
        **kwargs:

    Returns: Urdf schema

    """
    r, p, y = global_rotation
    return group(
        group(
            group(Urdf(src=src, jointValues=joints, key="go1", **kwargs), rotation=[r, 0, 0], key="roll"), rotation=[0, p, 0], key="pitch"
        ),
        rotation=[0, 0, y],
        position=position,
        key="yaw",
    )


def euler_from_quaternion(quat_angle: TensorType["batch", 4]) -> Tuple[TensorType["batch", 1]]:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians
