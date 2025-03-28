import asyncio
import os
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import PIL.Image as PImage
import torch
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, DefaultScene, Plane, Sphere, group

from lucidsim_old.dataset import Dataset
from lucidsim_old.job_queue import JobQueue
from lucidsim_old.utils import euler_from_quaternion, get_three_mat, quat_rotate


class CameraArgs(PrefixProto):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.2
    far = 1.2
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = False


class EgoViewRender(PrefixProto):
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "ball/scene_00005"

    rollout_range = [0, 2]
    ball_radius = 0.15

    render_type: Literal["rgb", "depth"] = "rgb"
    save_video: bool = True
    serve_root = Proto(env="$HOME")

    plane_color = "black"  # only used in RGB rendering mode
    ball_color = "red"

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        CameraArgs._update(deps)

        queries = dict(grid=False)
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.frames = {}
        self.lengths = {}
        self.rollouts = {}

        for rollout_id in range(*self.rollout_range):
            self.rollouts[rollout_id] = self.dataset[rollout_id]
            self.lengths[rollout_id] = len(self.rollouts[rollout_id]["obs"])
            self.frames[rollout_id] = {}

        self.job_queue = JobQueue()
        self._setup_job_queue()

        self.done = False

        print(self.app)

    def _setup_job_queue(self):
        for rollout_id in range(*self.rollout_range):
            for step in range(self.lengths[rollout_id]):
                self.job_queue.append((rollout_id, step))

    def _get_material_kwargs(self, color):
        if self.render_type == "depth":
            return dict(materialType="depth")
        elif self.render_type == "rgb":
            return dict(materialType="standard", material=dict(color=color))
        else:
            raise ValueError(f"render_type {self.render_type} not supported")

    def _save(self):
        if self.save_video:
            for rollout_id in range(*self.rollout_range):
                with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                    file_list = self.logger.glob(f"{self.render_type}_{rollout_id:04d}/*.png")
                    file_list = sorted(file_list)
                    self.logger.make_video(file_list, f"{self.render_type}_{rollout_id:04d}.mp4", fps=CameraArgs.fps)

                    print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

            exit()

    def _handle_step(self, rollout_id, step, sess):
        rollout = self.rollouts[rollout_id]
        quat_t = torch.from_numpy(rollout["states"][step][None, 3:7])
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = (angle.item() for angle in global_rot)

        position = rollout["states"][step][:3]
        cam_position = position.astype(np.float32) + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
                           0].numpy()

        cam_position = cam_position.tolist()

        material_kwargs = self._get_material_kwargs(self.ball_color)

        mat = get_three_mat(cam_position, [r, p, y])

        ball_location = rollout["ball_location"][step].tolist()

        sess.update @ [CameraView(**vars(CameraArgs), matrix=mat), Sphere(
            key="ball",
            position=ball_location,
            args=[self.ball_radius, 20, 20],
            **material_kwargs,
        )]

    async def _housekeeping(self):
        while True:
            self.job_queue.house_keeping()
            await sleep(5.0)

    async def main(self, sess: VuerSession):
        print("hi")
        # print(f"num_steps", self.num_steps)
        mesh = Plane(args=[500, 500, 10, 10], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color))
        material_kwargs = self._get_material_kwargs(self.ball_color)
        sess.set @ DefaultScene(
            group(
                # group(
                #     mesh,
                # ),
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

        loop = asyncio.get_running_loop()
        loop.create_task(self._housekeeping())

        self._handle_step(0, 0, sess)
        while len(self.job_queue) > 0:
            print("jobs left:", len(self.job_queue))
            job, mark_done, put_back = self.job_queue.take()

            rollout_id, step = job['job_params']
            try:
                print("Took frame", job['job_params'])
                self._handle_step(rollout_id, step, sess)
                await sleep(0.2)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                buff = event.value["frame"]
                pil_image = PImage.open(BytesIO(buff))
                img = np.array(pil_image)

                mark_done()
                await sleep(0.0)

                if self.save_video:
                    subdir = os.path.join(self.dataset_prefix, "ego_views")
                    with self.logger.Prefix(subdir):
                        path = self.logger.save_image(img, f"{self.render_type}_{rollout_id:04d}/frame_{step:04d}.png")
                        print("saved to", path)

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
