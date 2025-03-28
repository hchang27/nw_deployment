import asyncio
import os
import sys
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import PIL.Image as PImage
import torch
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, Cylinder, DefaultScene, Sphere, group

from lucidsim_old.dataset import Dataset
from lucidsim_old.job_queue import JobQueue
from lucidsim_old.terraria.textures.hurdle import Hurdle
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
    near = 0.05
    far = 50.0
    key = "ego"
    showFrustum = True
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = True


class RenderArgs(CameraArgs):
    width = 640
    height = 360


class EgoViewRender(PrefixProto):
    """
    Renders all marker images for the robot.
    """
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"
    dataset_prefix = "hurdle/scene_99999"

    background_texture = "assets/snowy.jpeg"
    hurdle_texture = "assets/bricks.jpeg"
    ground_texture = "assets/grass.png"

    rollout_range = [0, 10]

    render_type: Literal["rgb", "depth", "marker_mask", "background_mask", "obstacle_mask"] = "rgb"
    save_video: bool = False

    serve_root = Proto(env="$CWD")

    # only used in RGB rendering mode
    plane_color = "green"
    marker_color = "orange"
    hurdle_color = "red"

    # hard-coded to be blue at the moment. 
    # background_color = "blue"

    check_missing = False

    show_markers = False

    port = 8015

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        queries = dict(grid=False)
        if self.render_type in ["depth"]:
            queries["background"] = "000000"
        elif self.render_type in ["rgb"]:
            queries["background"] = "0000ff"
        elif self.render_type in ["object_mask", "obstacle_mask", "marker_mask"]:
            queries["background"] = "000000"
        elif self.render_type == "background_mask":
            queries["background"] = "ffffff"

        # adjusting the camera near and far

        if self.render_type == "depth":
            RenderArgs.near = 0.1
            RenderArgs.far = 3

        self.app = Vuer(static_root=self.serve_root, queries=queries, port=self.port,
                        uri=f"ws://localhost:{self.port}")

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.lengths = {}
        self.rollouts = {}

        self.goals = {}

        for rollout_id in range(*self.rollout_range):
            self.rollouts[rollout_id] = self.dataset[rollout_id]
            self.lengths[rollout_id] = len(self.rollouts[rollout_id]["obs"])
            # retrieve goals for the marker locations
            with self.logger.Prefix(os.path.join(self.dataset_prefix, "assets")):
                new_goal, = self.logger.load_pkl(f"goals_{rollout_id:04d}.pkl")
                self.goals[rollout_id] = new_goal

        self.num_goals = len(self.goals[self.rollout_range[0]])

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

    def _get_mesh(self):
        hurdle = Hurdle(dataset_prefix=self.dataset_prefix,
                        root=self.root,
                        prefix=self.prefix,
                        serve_root=self.serve_root,
                        )

        if self.render_type == "rgb":
            # whatever works best for OF 
            if self.ground_texture is not None and self.hurdle_texture is not None:
                ground_material = dict(materialType="standard",
                                       material=dict(map=f"http://localhost:{self.port}/static/{self.ground_texture}",
                                                     mapRepeat=[100, 100]))
                curb_material = dict(materialType="standard",
                                     material=dict(map=f"http://localhost:{self.port}/static/{self.hurdle_texture}",
                                                   mapRepeat=[0.1, 1]))
            else:
                curb_material = dict(materialType="standard", material=dict(color=self.hurdle_color))
                ground_material = dict(materialType="standard", material=dict(color=self.plane_color))
        elif self.render_type == "depth":
            curb_material = dict(materialType="depth", material=dict())
            ground_material = dict(materialType="depth", material=dict())
        elif self.render_type == "obstacle_mask":
            curb_material = dict(materialType="basic", material=dict(color="white"))
            ground_material = dict(materialType="basic", material=dict(color="black"))
        elif self.render_type == "background_mask":
            curb_material = dict(materialType="basic", material=dict(color="black"))
            ground_material = dict(materialType="basic", material=dict(color="white"))
        elif self.render_type == "marker_mask":
            curb_material = dict(materialType="basic", material=dict(color="black"))
            ground_material = dict(materialType="basic", material=dict(color="black"))
        else:
            raise NotImplementedError

        return hurdle(curb_material, ground_material)

    def _save(self):
        if self.save_video:
            for rollout_id in range(*self.rollout_range):
                with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                    file_list = self.logger.glob(f"{self.render_type}_{rollout_id:04d}/*.png")
                    file_list = sorted(file_list)
                    self.logger.make_video(file_list, f"{self.render_type}_{rollout_id:04d}.mp4", fps=RenderArgs.fps)

                    print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

            exit()

    def _grab_goals(self, step_num):

        if self.render_type == "rgb":
            cone_material = dict(materialType="standard", material=dict(color=self.marker_color))
        elif self.render_type == "depth":
            cone_material = dict(materialType="depth", material=dict())
        elif self.render_type == "background_mask":
            cone_material = dict(materialType="basic", material=dict(color="black"))
        elif self.render_type == "marker_mask":
            cone_material = dict(materialType="basic", material=dict(color="white"))
        elif self.render_type == "obstacle_mask":
            cone_material = dict(materialType="basic", material=dict(color="black"))
        else:
            raise NotImplementedError

        goal_updates = []
        for i, goal in enumerate(self.goals[step_num]):
            cone = Cylinder(
                args=[0.025, 0.25, 0.5, 20],
                position=goal,
                rotation=[np.pi / 2, 0, 0],
                key=f"goal_{i}",
                **cone_material,
            )
            goal_updates.append(cone)

        return goal_updates

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

        mat = get_three_mat(cam_position, [r, p, y])

        # update the goal locations
        updates = [CameraView(**vars(RenderArgs), matrix=mat)]

        if self.show_markers:
            updates.extend(self._grab_goals(rollout_id))

        sess.update @ updates

    async def _housekeeping(self):
        while True:
            self.job_queue.house_keeping()
            await sleep(5.0)

    async def main(self, sess: VuerSession):
        mesh = self._get_mesh()
        raw_children = [CameraView(**vars(RenderArgs))]

        if self.render_type == "rgb" and self.background_texture is not None:
            raw_children.append(Sphere(
                key="background",
                args=[50, 32, 32],
                position=[0, 0, 0],
                rotation=[np.pi / 2, 0, 0],
                material=dict(side=2, map=f"http://localhost:{self.port}/static/{self.background_texture}"),
            ))

        if self.show_markers:
            sess.set @ DefaultScene(
                group(
                    mesh,
                ),
                group(*self._grab_goals(self.rollout_range[0])),
                rawChildren=raw_children,
                up=[0, 0, 1],
            )
        else:
            sess.set @ DefaultScene(
                group(
                    mesh,
                ),
                rawChildren=raw_children,
                up=[0, 0, 1],
            )

        await sleep(2.0)

        while True:
            try:
                await sess.grab_render(key=RenderArgs.key, quality=0.9, ttl=30)
                print("Ready") 
                await sleep(0.1)
                break
            except asyncio.exceptions.TimeoutError:
                print("waiting for the scene to setup, timed out.")

        await sleep(2.0)

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
                event = await sess.grab_render(key=RenderArgs.key, quality=1.0, ttl=10)
                buff = event.value["frame"]
                img = np.array(PImage.open(BytesIO(buff)))

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

            await sleep(0.01)

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
