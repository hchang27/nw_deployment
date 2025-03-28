import asyncio
import os
import sys
from asyncio import sleep
from io import BytesIO
from typing import Literal

import numpy as np
import open3d as o3d
import PIL.Image as PImage
import torch
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, DefaultScene, Obj, Sphere, TriMesh, group

from lucidsim_old.dataset import Dataset
from lucidsim_old.job_queue import JobQueue
from lucidsim_old.utils import euler_from_quaternion, get_three_mat, quat_rotate


class CameraArgs(PrefixProto):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.005  # 0.4
    far = 2_000  # 7.0  
    key = "ego"
    showFrustum = False
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0, 0.02]
    movable = False
    monitor = True


class EgoViewRender(PrefixProto):
    local_prefix = Proto(env="$DATASETS/lucidsim/scenes/experiments/real2real")

    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes/experiments/real2real"
    dataset_prefix = "gap/scene_00001"
    background_img = "snowy.jpeg"

    rollout_range = None  # [0, 13]
    mesh_src = "textured.obj"
    mesh_material = "textured.mtl"

    render_type: Literal["rgb", "depth"] = "depth"
    save_video: bool = True
    serve_root = Proto(env="$HOME")

    check_missing = True

    port = 8014

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        CameraArgs._update(deps)

        queries = dict(grid=False)
        queries["background"] = "000000"

        if self.render_type == "depth":
            CameraArgs.near = 0.2
            CameraArgs.far = 7.0

        self.app = Vuer(static_root=self.serve_root, queries=queries, port=self.port, uri=f"ws://localhost:{self.port}")
        with self.logger.Prefix(self.dataset_prefix):
            gym_tf_pos = self.logger.read_params("gym_tf_pos")["value"]
            params = self.logger.read_params("labels")
            labels = self.logger.load_yaml("labels.yaml")

        self.mesh_position = params["mesh_position"]
        self.mesh_rotation = params["mesh_rotation"]
        self.gym_tf_pos = gym_tf_pos

        self.labels_list = [(label["start"], label["goal"]) for label in labels]

        self.dataset = Dataset(self.logger, os.path.join(self.dataset_prefix, "trajectories"))

        self.rollouts = {}
        self.lengths = {}
        self.frames = {}

        if self.rollout_range is None:
            self.rollout_range = (0, len(self.dataset))
            print(f"Using all the rollouts in the dataset: {self.rollout_range}")

        for rollout_id in range(*self.rollout_range):
            self.rollouts[rollout_id] = self.dataset[rollout_id]
            self.lengths[rollout_id] = len(self.rollouts[rollout_id]["obs"])
            self.frames[rollout_id] = {}

        self.job_queue = JobQueue()
        self._setup_job_queue()

        self.done = False

        self.loaded = False

        print(self.app)

    def _setup_job_queue(self):
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
        mesh = None
        src_path = os.path.join(self.serve_root, self.local_prefix, self.dataset_prefix, "assets", self.mesh_src)
        mat_path = os.path.join(self.serve_root, self.local_prefix, self.dataset_prefix, "assets", self.mesh_material)

        src_rel_path = os.path.relpath(src_path, self.serve_root)
        mat_rel_path = os.path.relpath(mat_path, self.serve_root)
        if self.render_type == "rgb":

            print(src_rel_path, mat_rel_path)

            mesh = Obj(
                key="terrain",
                src="http://localhost:8012/static/" + src_rel_path,
                mtl="http://localhost:8012/static/" + mat_rel_path,
                position=self.mesh_position,
                rotation=np.deg2rad(self.mesh_rotation),
                onLoad="loaded mesh"
            )
        elif self.render_type == "depth":
            trimesh = o3d.io.read_triangle_mesh(str(src_path))
            mesh = TriMesh(
                key="terrain",
                vertices=np.array(trimesh.vertices),
                faces=np.array(trimesh.triangles),
                position=self.mesh_position,
                rotation=np.deg2rad(self.mesh_rotation),
                materialType="depth",
                onLoad="loaded mesh"
            )

            self.loaded = True

        else:
            raise NotImplementedError

        return mesh

    def _save(self):
        if self.save_video:
            for rollout_id in range(*self.rollout_range):
                framestack = self.frames[rollout_id]
                framestack = np.array([framestack[i] for i in range(len(framestack))])  # sorted

                print(f"saving {len(self.frames[rollout_id])} total frames to video")
                with self.logger.Prefix(os.path.join(self.dataset_prefix, "ego_views")):
                    self.logger.save_video(framestack, f"{self.render_type}_{rollout_id:04d}.mp4",
                                           fps=CameraArgs.fps)

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

        mat = get_three_mat(cam_position, [r, p, y])

        sess.update @ CameraView(**vars(CameraArgs), matrix=mat)

    async def _housekeeping(self):
        while True:
            self.job_queue.house_keeping()
            await sleep(5.0)

    async def main(self, sess: VuerSession):
        print("hi")
        mesh = self._get_mesh()
        if self.render_type == "depth":
            sess.set @ DefaultScene(
                group(
                    mesh,
                    position=self.gym_tf_pos,
                ),
                rawChildren=[
                    CameraView(
                        **vars(CameraArgs),
                    ),
                ],
                up=[0, 0, 1],
            )
        else:
            sess.set @ DefaultScene(
                group(
                    mesh,
                    position=self.gym_tf_pos,
                ),
                rawChildren=[
                    CameraView(
                        **vars(CameraArgs),
                    ),
                    Sphere(
                        key="background",
                        args=[50, 32, 32],
                        position=[0, 0, 0],
                        rotation=[np.pi / 2, 0, 0],
                        material=dict(side=2, map=f"http://localhost:8012/static/{self.background_img}"),
                    )
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
            except asyncio.exceptions.TimeoutError:
                print("setup timeout")

        while not self.loaded:
            print("waiting on load")
            await sleep(0.1)

        # start housekeeping
        loop = asyncio.get_running_loop()
        loop.create_task(self._housekeeping())

        self._handle_step(self.rollout_range[0], 0, sess)
        while len(self.job_queue) > 0:
            print("jobs left:", len(self.job_queue))
            # print("total results", len(self.frames))
            job, mark_done, put_back = self.job_queue.take()

            rollout_id, step = job['job_params']

            try:
                print("Took frame", job['job_params'])
                self._handle_step(rollout_id, step, sess)
                await sleep(0.2)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)
                buff = event.value["frame"]
                img = np.array(PImage.open(BytesIO(buff)))
                self.frames[rollout_id][step] = img

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

    async def onLoad(self, event, sess):
        self.loaded = True
        print(f"mesh has been loaded: {event.value}")

    def __call__(self, *args, **kwargs):
        print("starting")
        self.app.add_handler("LOAD", self.onLoad)
        self.app.spawn(self.main)
        self.app.run()


if __name__ == "__main__":
    ego_view = EgoViewRender()
    ego_view()
