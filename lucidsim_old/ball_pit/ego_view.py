import os
from asyncio import sleep
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import Obj, TriMesh, CameraView, DefaultScene, group, Sphere, Plane

from lucidsim_old.dataset import Dataset
import open3d as o3d

import torch
from lucidsim_old.utils import ISAAC_DOF_NAMES, euler_from_quaternion, get_three_mat, Go1, quat_rotate
import PIL.Image as PImage
from lucidsim_old import ROBOT_DIR
import asyncio


class CameraArgs(PrefixProto):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.4
    far = 5.0
    key = "ego"
    showFrustum = True
    downsample = 2
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False
    monitor = True


class EgoViewRender(PrefixProto):
    prefix = Proto(env="$DATASETS/lucidsim/scenes/")
    dataset_prefix = "ball/scene_00002"

    rollout_id = 0
    ball_radius = 0.22

    render_type: Literal["rgb", "depth"] = "depth"
    save_video: bool = False
    serve_root = Proto(env="$HOME")

    plane_color = "gray"  # only used in RGB rendering mode
    ball_color = "red"

    def __post_init__(self):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.prefix)

        # CameraArgs._update(deps)
        self.frames = []

        queries = dict(grid=False)
        if self.render_type == "depth":
            queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))
        self.rollout = self.dataset[self.rollout_id]
        self.num_steps = len(self.rollout["obs"])

        print(self.app)

    def _get_material_kwargs(self, color):
        if self.render_type == "depth":
            return dict(materialType="depth")
        elif self.render_type == "rgb":
            return dict(materialType="standard", material=dict(color=color))
        else:
            raise ValueError(f"render_type {self.render_type} not supported")

    def _save(self):
        if self.save_video:
            print(f"saving {len(self.frames)} total frames to video")
            with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                self.logger.save_video(self.frames, f"{self.render_type}_{self.rollout_id:04d}.mp4", fps=CameraArgs.fps)
                print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

    def _handle_step(self, step, sess):
        print(f"step {step}/{self.num_steps}")
        joint_values = {name: angle for name, angle in zip(ISAAC_DOF_NAMES, self.rollout["dofs"][step])}
        quat_t = torch.from_numpy(self.rollout["states"][step][None, 3:7])
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = [angle.item() for angle in global_rot]

        position = self.rollout["states"][step][:3]
        cam_position = position.astype(np.float32) + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
                           0].numpy()

        position = position.tolist()
        cam_position = cam_position.tolist()

        material_kwargs = self._get_material_kwargs(self.ball_color)

        mat = get_three_mat(cam_position, [r, p, y])

        ball_location = self.rollout["ball_location"][step].tolist()

        sess.upsert @ (
            Go1(
                f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                joint_values,
                global_rotation=(r, p, y),
                position=position,
            ),
            Sphere(
                key="ball",
                position=ball_location,
                args=[self.ball_radius, 20, 20],
                **material_kwargs,
            )
        )

        sess.update @ CameraView(**vars(CameraArgs), matrix=mat)

    async def main(self, sess: VuerSession):
        print("hi")
        print(f"num_steps", self.num_steps)
        mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color))
        sess.set @ DefaultScene(
            group(
                group(
                    mesh,
                ),
            ),
            rawChildren=[
                CameraView(
                    **vars(CameraArgs),
                ),
            ],
            up=[0, 0, 1],
        )

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
        self._handle_step(0, sess)
        while step < self.num_steps:
            try:
                self._handle_step(step, sess)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                buff = event.value["frame"]
                pil_image = PImage.open(BytesIO(buff))
                img = np.array(pil_image)
                self.frames.append(img)

                if self.save_video:
                    with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                        self.logger.save_image(img, f"{self.render_type}_{self.rollout_id:04d}/frame_{step:04d}.png")

                step += 1
            except asyncio.exceptions.TimeoutError:
                print("timeout")
            await sleep(0.00)

        self._save()

    def __call__(self, *args, **kwargs):
        print("starting")
        self.app.spawn(self.main)
        self.app.run()


if __name__ == "__main__":
    ego_view = EgoViewRender()
    ego_view()
