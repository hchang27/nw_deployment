from io import BytesIO

import numpy as np
from pathlib import Path

import open3d as o3d
from tqdm import tqdm
from vuer.events import ClientEvent

from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, group, Obj, TimelineControls, TriMesh, Scene, Plane, DefaultScene, Sphere
from asyncio import sleep

from main_street.scripts.utils import get_three_mat, euler_from_quaternion, ISAAC_DOF_NAMES, Go1, quat_rotate
from main_street.scripts.ball.ball_dataset import BallDataset
import torch
from params_proto import ParamsProto, PrefixProto
from typing import Literal
from ml_logger import logger
import PIL.Image as PImage
import asyncio


class Args(ParamsProto):
    serve_root = Path("/home/exx/mit/parkour/main_street/assets/")
    urdf_file = "robots/gabe_go1/urdf/go1.urdf"
    # urdf_file = "robots/gabe_go1/xml/go1.xml"

    # dataset_prefix = Path("/lucid-sim/lucid-sim/datasets/ball/debug/00001")
    dataset_prefix = Path("/lucid-sim/lucid-sim/datasets/ball/debug/00005/")

    rollout_id = 0

    ball_radius = 0.22

    render_type: Literal["rgb", "depth"] = "depth"
    save_video: bool = True


class CameraArgs(PrefixProto):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.4
    far = 8.0
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False
    monitor = True


frames = []

queries = dict(grid=False)
if Args.render_type == "depth":
    queries["background"] = "000000"

app = Vuer(static_root=str(Args.serve_root), queries=queries)

dataset = BallDataset(Args.dataset_prefix / "rollouts")
rollout = dataset[Args.rollout_id]

num_samples = len(rollout)
steps_per_sample = len(rollout[0]["obs"])
num_steps = num_samples * steps_per_sample


def get_material_kwargs(render_type, color):
    if render_type == "depth":
        return dict(materialType="depth")
    elif render_type == "rgb":
        return dict(materialType="standard", material=dict(color=color))
    else:
        raise ValueError(f"render_type {render_type} not supported")


@app.spawn
async def main(sess: VuerSession):
    global num_steps, frames, steps_per_sample
    mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                 **get_material_kwargs(Args.render_type, "gray"))
    sess.set @ DefaultScene(
        group(
            group(
                mesh,
                rotation=[0, 0, -np.pi / 2],
            ),
            # rawChildren=[
            #     CameraView(**vars(CameraArgs)),
            # ],

        ),
        rawChildren=[CameraView(**vars(CameraArgs))],
        up=[0, 0, 1],

    )
    print("num_steps", num_steps)

    await sleep(1.0)
    while True:
        try:
            await sess.grab_render(key=CameraArgs.key, quality=0.9)
            print("Ready")
            await sleep(0.01)
            break
        except TimeoutError:
            print("setup timeout")
    step = 0
    handle_step(0, sess)
    while step < num_steps:
        try:
            handle_step(step, sess)
            event = await sess.grab_render(key=CameraArgs.key, quality=0.5)
            value = event.value
            buff = value["frame"]
            pil_image = PImage.open(BytesIO(buff))
            img = np.array(pil_image)

            frames.append(img)

            step += 1
        except asyncio.exceptions.TimeoutError:
            print("timeout")
        # await sleep(0.01)
        await sleep(0.00)

    save()


def handle_step(step, sess):
    sample_id = step // steps_per_sample
    idx = step % steps_per_sample

    sample = rollout[sample_id]

    joint_values = {name: angle for name, angle in zip(ISAAC_DOF_NAMES, sample["dofs"][idx])}
    quat_t = torch.from_numpy(sample["states"][idx][None, 3:7])
    global_rot = euler_from_quaternion(quat_t.float())
    r, p, y = [angle.item() for angle in global_rot]

    position = sample["states"][idx][:3]
    cam_position = position.astype(np.float32) + quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
        0].numpy()

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
        Sphere(
            key="ball",
            position=sample["ball_location"].tolist(),
            args=[Args.ball_radius, 20, 20],
            **get_material_kwargs(Args.render_type, "red"),
        )
    )

    sess.update @ (
        CameraView(**vars(CameraArgs), matrix=mat),
    )


def save():
    global frames
    if Args.save_video:
        print(f"saving {len(frames)} total frames to video")
        with logger.Prefix(str(Args.dataset_prefix / "rollouts")):
            logger.save_video(frames, f"{Args.render_type}_{Args.rollout_id:04d}.mp4", fps=CameraArgs.fps)
            for id, frame in tqdm(enumerate(frames), desc="saving frames"):
                logger.save_image(frame, f"{Args.render_type}_{Args.rollout_id:04d}/frame_{id:04d}.png")
            print(f"Done saving video to the dataset folder: {logger.get_dash_url()}")


app.run()
