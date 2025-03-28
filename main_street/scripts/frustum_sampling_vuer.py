import isaacgym
from asyncio import sleep
from pathlib import Path

from params_proto import ParamsProto
from torchtyping import TensorType
from typing import List

from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import AmbientLight, Scene, Movable, DirectionalLight, Plane, TimelineControls, CameraView, Urdf, \
    Sphere

from main_street.envs import *
from main_street.utils.helpers import get_vertical_fov, get_horizontal_fov, sample_camera_frustum_batch

import numpy as np
import torch

from vuer.serdes import b64jpg

app = Vuer(static_root=Path(__file__).parent)

cam_to_world = torch.eye(4)
cam_to_world[:3, -1] = torch.tensor([0, 0, 10])


def three_to_torch(mat: List[float]) -> TensorType[4, 4]:
    """
    List is column major ordering
    """
    return torch.tensor(mat).reshape(4, 4).float().T


class TerrainArgs(ParamsProto, cli=False):
    scale = 1.375
    shift = -0.02
    length = 30
    width = 100
    heightmap_uint8 = np.load('heightmap.npy')


class SampleArgs(ParamsProto, cli=False):
    num_samples = 50
    radius = 0.1


class CameraArgs(ParamsProto, cli=False):
    width = 640
    height = 360
    fov = get_vertical_fov(105, width, height)  # vertical
    stream = "frame"
    fps = 30
    near = 2
    far = 5
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2


@app.spawn
async def main(session):
    width_px, length_px = TerrainArgs.heightmap_uint8.shape

    session.set @ Scene(
        AmbientLight(intensity=0.1),
        Movable(
            DirectionalLight(
                intensity=0.5
            ),
            position=[0, -10, 5]
        ),
        Plane(
            args=[TerrainArgs.length, TerrainArgs.width, length_px, width_px],
            key='heightmap',
            materialType="standard",
            material=dict(
                displacementMap=b64jpg(TerrainArgs.heightmap_uint8),
                displacementScale=TerrainArgs.scale,
                displacementBias=TerrainArgs.shift,
            ),
            rotation=[0, 0, np.pi / 2],
        ),
        TimelineControls(start=0, end=10, key="timeline"),
        CameraView(**vars(CameraArgs), position=[0, 0, 10]),
        grid=False,
        up=[0, 0, 1],
    )

    await sleep(0.01)


async def step_handler(e: ClientEvent, sess: VuerSession):
    horizontal_fov = get_horizontal_fov(CameraArgs.fov, CameraArgs.width, CameraArgs.height)
    sample_x, sample_y, sample_z = sample_camera_frustum_batch(horizontal_fov, CameraArgs.width, CameraArgs.height,
                                                               CameraArgs.near, CameraArgs.far,
                                                               num_samples=SampleArgs.num_samples)

    sample_x = torch.from_numpy(sample_x)
    sample_y = torch.from_numpy(sample_y)
    sample_z = torch.from_numpy(sample_z)
    samples_to_cam = torch.cat([sample_x, sample_y, sample_z, torch.ones_like(sample_x)], dim=-1).float()

    extra_rot = torch.eye(4).float()

    extra_rot[:3, :3] = torch.tensor([[0., 0., -1],
                                      [-1, 0, 0.],
                                      [0., 1., 0]]).T

    samples_to_world = (cam_to_world @ extra_rot @ samples_to_cam.T).T[:, :3]

    args = [
        Sphere(key=f"ball_{i}", args=[SampleArgs.radius, 200, 200], position=pos, materialType="standard",
               material=dict(color="red"))
        for i, pos in
        enumerate(samples_to_world.cpu().numpy().tolist())]

    sess.upsert @ args


async def update_camera(e: ClientEvent, sess: VuerSession):
    if e.key != "ego":
        return
    global cam_to_world
    cam_to_world = three_to_torch(e.value["matrix"])


app.add_handler("TIMELINE_STEP", step_handler)
app.add_handler("CAMERA_MOVE", update_camera)
app.run()
