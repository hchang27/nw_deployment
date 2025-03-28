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

from lucidsim_old.job_queue import JobQueue
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
    monitor = False


class EgoViewRender(PrefixProto):
    prefix = Proto(env="$DATASETS/lucidsim/scenes/")
    dataset_prefix = "ball/scene_00001"

    rollout_id = 20
    ball_radius = 0.22

    render_type: Literal["rgb", "depth"] = "depth"
    save_video: bool = True
    serve_root = Proto(env="$HOME")

    plane_color = "gray"  # only used in RGB rendering mode
    ball_color = "red"

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.prefix)
        
        print("rollout id", self.rollout_id)

        CameraArgs._update(deps)
        self.frames = {}

        queries = dict(grid=False)
        # if self.render_type == "depth":
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))
        self.rollout = self.dataset[self.rollout_id]
        self.num_steps = len(self.rollout["obs"])

        self.job_queue = JobQueue()
        self._setup_job_queue()

        self.done = False

        print(self.app)

    def _setup_job_queue(self):
        for step in range(self.num_steps):
            self.job_queue.append(step)

    def _get_material_kwargs(self, color):
        if self.render_type == "depth":
            return dict(materialType="depth")
        elif self.render_type == "rgb":
            return dict(materialType="standard", material=dict(color=color))
        else:
            raise ValueError(f"render_type {self.render_type} not supported")

    def _save(self):
        if self.save_video:
            self.frames = np.array([self.frames[i] for i in range(len(self.frames))])  # sorted

            print(f"saving {len(self.frames)} total frames to video")
            with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                self.logger.save_video(self.frames, f"{self.render_type}_{self.rollout_id:04d}.mp4", fps=CameraArgs.fps)

                for step, image in enumerate(self.frames):
                    self.logger.save_image(image, f"{self.render_type}_{self.rollout_id:04d}/frame_{step:04d}.png")

                print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

                exit()

    def _handle_step(self, step, sess):
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

        sess.update @ [CameraView(**vars(CameraArgs), matrix=mat), Sphere(
            key="ball",
            position=ball_location,
            args=[self.ball_radius, 20, 20],
            **material_kwargs,
        )]

    async def main(self, sess: VuerSession):
        print("hi")
        print(f"num_steps", self.num_steps)
        mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color))
        material_kwargs = self._get_material_kwargs(self.ball_color)
        sess.set @ DefaultScene(
            group(
                group(
                    mesh,
                ),
                Sphere(
                    key="ball",
                    args=[self.ball_radius, 20, 20],
                    **material_kwargs,
                )
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
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        self._handle_step(0, sess)
        while len(self.job_queue) > 0:
            print("jobs left:", len(self.job_queue))
            print("total results", len(self.frames))
            job, mark_done, put_back = self.job_queue.take()

            id = job['job_params']
            try:
                print("Took frame", job['job_params'])
                self._handle_step(job['job_params'], sess)
                await sleep(0.2)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                value = event.value
                buff = value["frame"]
                pil_image = PImage.open(BytesIO(buff))
                img = np.array(pil_image)
                self.frames[id] = img

                mark_done()
                await sleep(0.0)
                self.job_queue.house_keeping()

                # if self.save_video:
                #     with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                #         self.logger.save_image(img, f"{self.render_type}_{self.rollout_id:04d}/frame_{step:04d}.png")

            except asyncio.exceptions.TimeoutError:
                print("Oops, something went wrong. Putting the job back to the queue.")
                put_back()

        if not self.done:
            self._save()
            self.done = True

    def __call__(self, *args, **kwargs):
        print("starting")
        self.app.spawn(self.main)
        self.app.run()


if __name__ == "__main__":
    ego_view = EgoViewRender()
    ego_view()
