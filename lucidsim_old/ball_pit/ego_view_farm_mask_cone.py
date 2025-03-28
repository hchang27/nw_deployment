import asyncio
import os
import random
import sys
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import PIL.Image as PImage
import torch
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import Box, CameraView, Cylinder, DefaultScene, Plane, Sphere, group

from lucidsim_old.dataset import Dataset
from lucidsim_old.job_queue import JobQueue
from lucidsim_old.utils import euler_from_quaternion, get_three_mat, quat_rotate


def get_expanded_fov(expansion_factor, fov):
    """
    Compute the expanded field of view given the expansion factor and the original field of view.

    Expansion factor: the ratio of the new pixel width to the old pixel width.
    """

    fov_rad = np.deg2rad(fov)

    new_fov_rad = 2 * np.arctan(np.tan(fov_rad / 2) * expansion_factor)
    return np.rad2deg(new_fov_rad)


class CameraArgs(PrefixProto):
    width = 1280  # 640
    height = 720  # 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.1
    far = 50.0
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = True


class RenderArgs(CameraArgs):
    width = 1280
    height = 768

    expansion_factor = height / CameraArgs.height

    fov = get_expanded_fov(expansion_factor, CameraArgs.fov)
    print(f"Rendering fov: {fov}")


class EgoViewRender(PrefixProto):
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "cone-debug/scene_00001"

    background_img = "snowy.jpeg"
    plane_texture = "bricks.jpeg"
    ball_texture = None  # "soccer_sph_s.png"

    rollout_range = [0, 1]
    cylinder_args = [0.025, 0.1, 0.15, 20]

    render_type: Literal["rgb", "depth", "object_mask", "background_mask"] = "background_mask"
    save_video: bool = True
    serve_root = Proto(env="$HOME")

    # only used in RGB rendering mode
    plane_color = "black"
    ball_color = "orange"
    room_color = "blue"

    box_probability = 1.0
    box_size_range = [5, 8]

    check_missing = False

    port = 8013

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)
        if self.render_type in ["rgb", "depth"]:
            queries["background"] = "000000"
        elif self.render_type == "object_mask":
            queries["background"] = "000000"
        elif self.render_type == "background_mask":
            queries["background"] = "ffffff"

        if self.render_type == "rgb":
            RenderArgs.far = 1_000

        self.app = Vuer(static_root=self.serve_root, queries=queries, port=self.port,
                        uri=f"ws://localhost:{self.port}")

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
        # glob the missing images
        finished_files = self.logger.glob(f"{self.render_type}**/*.png",
                                          wd=os.path.join(self.dataset_prefix, "ego_views"))
        done = set()

        if self.check_missing and finished_files is not None:
            for file in finished_files:
                rollout_id = int(file.split("/")[0].split("_")[-1])
                step = int(file.split("/")[-1].split("_")[-1].split(".")[0])
                done.add((rollout_id, step))

        for rollout_id in range(*self.rollout_range):
            for step in range(self.lengths[rollout_id]):
                if (rollout_id, step) not in done:
                    self.job_queue.append((rollout_id, step))

        if len(self.job_queue.queue) == 0:
            print("All jobs are done, exiting")
            sys.exit()

    def _get_material_kwargs(self, color, texture, is_ball=False):
        if self.render_type == "depth":
            return dict(materialType="depth", material=dict())
        elif self.render_type == "rgb":
            if texture is not None:
                return dict(materialType="standard",
                            material=dict(map=f"http://localhost:{self.port}/static/{texture}"),
                            mapRepeat=[10000, 1000])
            return dict(materialType="standard", material=dict(color=color))
        elif self.render_type == "object_mask":
            return dict(materialType="basic", material=dict(color="white" if is_ball else "black"))
        elif self.render_type == "background_mask":
            return dict(materialType="basic", material=dict(color="white" if not is_ball else "black"))
        else:
            raise ValueError(f"render_type {self.render_type} not supported")

    def _save(self):
        if self.save_video:
            for rollout_id in range(*self.rollout_range):
                with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                    file_list = self.logger.glob(f"{self.render_type}_{rollout_id:04d}/*.png")
                    file_list = sorted(file_list)
                    self.logger.make_video(file_list, f"{self.render_type}_{rollout_id:04d}.mp4", fps=RenderArgs.fps)

                    print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

            exit()

    def _handle_step(self, rollout_id, step, sess):
        rollout = self.rollouts[rollout_id]
        quat_t = torch.from_numpy(rollout["states"][step][None, 3:7])
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = (angle.item() for angle in global_rot)

        position = rollout["states"][step][:3]
        cam_position = position.astype(np.float32) + \
                       quat_rotate(quat_t.float(), torch.tensor([RenderArgs.cam_to_base]))[
                           0].numpy()

        cam_position = cam_position.tolist()

        material_kwargs = self._get_material_kwargs(self.ball_color, self.ball_texture, is_ball=True)

        mat = get_three_mat(cam_position, [r, p, y])

        ball_location = rollout["ball_location"][step].tolist()

        components = [CameraView(**vars(RenderArgs), matrix=mat), Cylinder(
            key="ball",
            position=ball_location,
            rotation=[np.pi / 2, 0, 0],
            args=self.cylinder_args,
            **material_kwargs,
        )]
        if self.render_type != "rgb":
            if random.random() < self.box_probability:
                box_size = random.uniform(*self.box_size_range)
                box_location = rollout["ball_location"][step].tolist()
                box_materials = self._get_material_kwargs(self.room_color, None, is_ball=False)
                box_materials['material']['side'] = 2
                components.append(Box(
                    key="box",
                    position=box_location,
                    args=[box_size, box_size, box_size],
                    **box_materials,
                ))
            else:
                # remove the box
                box_materials = self._get_material_kwargs(self.room_color, None, is_ball=False)
                box_materials['material']['side'] = 2
                components.append(Box(key="box", position=[0, 0, -1000], args=[0, 0, 0], **box_materials))

        sess.update @ components

    async def _housekeeping(self):
        while True:
            self.job_queue.house_keeping()
            await sleep(5.0)

    async def main(self, sess: VuerSession):
        # print(f"num_steps", self.num_steps)
        mesh = Plane(args=[500, 500, 1000, 1000], position=[0, 0, 0], key="ground",
                     **self._get_material_kwargs(self.plane_color, self.plane_texture))
        material_kwargs = self._get_material_kwargs(self.ball_color, self.ball_texture)
        if self.render_type != "rgb":
            sess.set @ DefaultScene(
                group(
                    group(
                        mesh,
                        Box(args=[15, 15, 15], key="box")
                    ),
                    Cylinder(
                        key="ball",
                        args=self.cylinder_args,
                        rotation=[np.pi / 2, 0, 0],
                        **material_kwargs,
                    )
                ),
                rawChildren=[
                    CameraView(
                        **vars(RenderArgs),
                    ),
                ],
                up=[0, 0, 1],
            )
        elif self.render_type == "rgb" and self.background_img is not None:
            # we just care about computing the flow map
            sess.set @ DefaultScene(
                mesh,
                Sphere(
                    key="background",
                    args=[200, 32, 32],
                    position=[0, 0, 0],
                    rotation=[np.pi / 2, 0, 0],
                    material=dict(side=2, map=f"http://localhost:8012/static/{self.background_img}"),
                ),
                Cylinder(
                    key="ball",
                    args=self.cylinder_args,
                    rotation=[np.pi / 2, 0, 0],
                    **material_kwargs,
                ),
                rawChildren=[CameraView(**vars(RenderArgs))],
                up=[0, 0, 1],
            )

        await sleep(1.0)

        print("hi")
        while True:
            try:
                await sess.grab_render(key=RenderArgs.key, quality=0.9)
                print("Ready")
                await sleep(0.01)
                break
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        loop = asyncio.get_running_loop()
        loop.create_task(self._housekeeping())

        self._handle_step(self.rollout_range[0], 0, sess)
        while len(self.job_queue) > 0:
            print("jobs left:", len(self.job_queue))
            job, mark_done, put_back = self.job_queue.take()

            rollout_id, step = job['job_params']
            try:
                print("Took frame", job['job_params'])
                self._handle_step(rollout_id, step, sess)
                await sleep(0.2)
                event = await sess.grab_render(key=RenderArgs.key, quality=1.0)
                buff = event.value["frame"]
                pil_image = PImage.open(BytesIO(buff))

                # resize image to the desired size (accounting for different display sizes)
                pil_image = pil_image.resize((RenderArgs.width, RenderArgs.height), PImage.ANTIALIAS)

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
