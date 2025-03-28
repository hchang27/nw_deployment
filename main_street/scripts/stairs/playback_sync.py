from asyncio import sleep
from asyncio.exceptions import TimeoutError
from io import BytesIO
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np
import open3d as o3d
import PIL.Image as PImage
import torch
from ml_logger import logger
from params_proto import ParamsProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, group, Obj, TriMesh, DefaultScene

from main_street.scripts.stairs.dataset import Dataset
from main_street.scripts.utils import (
    Go1,
    get_three_mat,
    euler_from_quaternion,
    ISAAC_DOF_NAMES,
)


class Args(ParamsProto):
    serve_root = Proto(env="$HOME")
    urdf_file = "mit/parkour/main_street/assets/robots/gabe_go1/urdf/go1.urdf"

    asset_file = "datasets/lucidsim/scenes/mit_stairs/stairs_0001_v1/assets/textured.obj"
    material_file = "datasets/lucidsim/scenes/mit_stairs/stairs_0001_v1/assets/textured.mtl"

    mesh_tf_pos: List[float]  # = [0, 0, 1.4]
    gym_tf_pos: List[float]  # used in gym to align corner of mesh with (0, 0) TODO: save it
    mesh_tf_rot: List[float]  # = [90, 0, 0]

    dataset_prefix = Path("/lucid-sim/lucid-sim/datasets/stairs/debug/00001")

    rollout_id = 0

    render_type: Literal["rgb", "depth"] = "rgb"
    save_video: bool = True


class CameraArgs(ParamsProto, cli=False):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.4
    far = 8.0
    key = "ego"
    showFrustum = True
    downsample = 2
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False


frames = []

queries = dict(grid=False)
if Args.render_type == "depth":
    queries["background"] = "000000"

app = Vuer(static_root=Args.serve_root, queries=queries)

with logger.Prefix(Args.dataset_prefix):
    gym_tf_pos = logger.read_params("play")["gym_tf_pos"]
    params = logger.read_params("labels")
    labels = logger.load_yaml("labels.yaml")

labels_list = [(label["start"], label["goal"]) for label in labels]

Args.mesh_tf_pos = params["mesh_tf_pos"]
Args.gym_tf_pos = gym_tf_pos
Args.mesh_tf_rot = params["mesh_tf_rot"]

dataset = Dataset(Args.dataset_prefix / "rollouts")
rollout = dataset[Args.rollout_id]
start, end = labels_list[Args.rollout_id]
num_steps = len(rollout["obs"])


def get_mesh(render_type):
    mesh = None
    if render_type == "rgb":
        mesh = Obj(
            key="terrain",
            src="http://localhost:8012/static/" + Args.asset_file,
            mtl="http://localhost:8012/static/" + Args.material_file,
            position=Args.mesh_tf_pos,
            rotation=np.deg2rad(Args.mesh_tf_rot),
        )
    elif render_type == "depth":
        trimesh = o3d.io.read_triangle_mesh(str(Args.serve_root / Args.asset_file))
        mesh = TriMesh(
            key="terrain",
            vertices=np.array(trimesh.vertices),
            faces=np.array(trimesh.triangles),
            position=Args.mesh_tf_pos,
            rotation=np.deg2rad(Args.mesh_tf_rot),
            materialType="depth",
        )

    else:
        raise NotImplementedError

    return mesh


def save(frames):
    if Args.save_video:
        print(f"saving {len(frames)} total frames to video")
        with logger.Prefix(str(Args.dataset_prefix / "rollouts")), logger.Sync():
            logger.save_video(
                frames,
                f"{Args.render_type}_{Args.rollout_id:04d}.mp4",
                fps=CameraArgs.fps,
            )
            # for id, frame in tqdm(enumerate(frames), desc="saving frames"):
            #     logger.save_image(frame, f"{Args.render_type}_{Args.rollout_id:04d}/frame_{id:04d}.png")
            print(f"Done saving video to the dataset folder: {logger.get_dash_url()}")


def handle_step(step, sess):
    global num_steps, done, frames

    joint_values = {name: angle for name, angle in zip(ISAAC_DOF_NAMES, rollout["dofs"][step])}

    quat_t = torch.from_numpy(rollout["states"][step][None, 3:7])
    global_rot = euler_from_quaternion(quat_t.float())
    r, p, y = [angle.item() for angle in global_rot]

    position = rollout["states"][step][:3]
    cam_position = position + np.array(CameraArgs.cam_to_base)

    position = position.tolist()
    cam_position = cam_position.tolist()

    mat = get_three_mat(cam_position, [r, p, y])
    sess.upsert @ (
        Go1(
            f"http://localhost:8012/static/{Args.urdf_file}",
            joint_values,
            global_rotation=(r, p, y),
            position=position,
        ),
    )

    sess.update @ CameraView(**vars(CameraArgs), matrix=mat)


# @app.add_handler("CAMERA_VIEW")
# async def camera_handler(event, session):
#     print('hey')
#     buff = event.value["frame"]
#     pil_image = PImage.open(BytesIO(buff))
#     img = np.array(pil_image)
#     frames.append(img)
#
#     cv2.imshow("frame", img)
#     cv2.waitKey(1)


@app.spawn
async def main(sess: VuerSession):
    global num_steps
    mesh = get_mesh(Args.render_type)
    sess.set @ DefaultScene(
        group(
            group(
                mesh,
                rotation=[0, 0, -np.pi / 2],
            ),
            position=Args.gym_tf_pos,
        ),
        rawChildren=[
            CameraView(
                **vars(CameraArgs),
            ),
        ],
        up=[0, 0, 1],
    )

    await sleep(2.0)

    while True:
        try:
            await sess.grab_render(key=CameraArgs.key, quality=0.9)
            print("Ready")
            await sleep(0.01)
            break
        except TimeoutError:
            print("setup timeout")

    step = 0
    # while step < num_steps:
    while True:
        try:
            event = await sess.grab_render(key=CameraArgs.key, quality=1)
            value = event.value
            buff = value["frame"]
            pil_image = PImage.open(BytesIO(buff))
            img = np.array(pil_image)
            # frames.append(img)
            cv2.imshow("frame", img)
            cv2.waitKey(1)

            handle_step(step % num_steps, sess)
            step += 1
        except TimeoutError:
            print("timeout")
        await sleep(0.1)

    save(frames)


app.run()
