import asyncio
import os
from asyncio import sleep
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.events import GrabRender
from vuer.schemas import Obj, TriMesh, CameraView, DefaultScene, group, Sphere

from lucidsim_old.dataset import Dataset
import open3d as o3d

import torch

from lucidsim_old.job_queue import JobQueue
from lucidsim_old.utils import ISAAC_DOF_NAMES, euler_from_quaternion, get_three_mat, Go1, quat_rotate
import PIL.Image as PImage
from lucidsim_old import ROBOT_DIR


class CameraArgs(PrefixProto):
    width = 640
    height = 360
    fov = 70  # vertical
    stream = "ondemand"
    fps = 50
    near = 0.4
    far = 8.0
    key = "ego"
    showFrustum = False
    downsample = 1
    distanceToCamera = 2
    cam_to_base = [0.29, 0.0425, 0.08]
    movable = False
    monitor = False


class EgoViewRender(PrefixProto):
    prefix = Proto(env="$DATASETS/lucidsim/scenes/")
    dataset_prefix = Path("mit_stairs/stairs_0001_v1")

    mesh_src = "textured.obj"
    mesh_material = "textured.mtl"

    rollout_id = 4
    render_type: Literal["rgb", "depth"] = "rgb"
    save_images: bool = True

    serve_root = Proto(env="$HOME")

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.prefix)

        CameraArgs._update(deps)
        self.frames = {}

        queries = dict(grid=False)
        # if self.render_type == "depth":
        queries["background"] = "000000"

        self.app = Vuer(static_root=self.serve_root, queries=queries)

        with self.logger.Prefix(self.dataset_prefix):
            gym_tf_pos = self.logger.read_params("gym_tf_pos")["value"]
            params = self.logger.read_params("labels")
            labels = self.logger.load_yaml("labels.yaml")

        self.mesh_position = params["mesh_position"]
        self.mesh_rotation = params["mesh_rotation"]
        self.gym_tf_pos = gym_tf_pos

        self.labels_list = [(label["start"], label["goal"]) for label in labels]

        self.dataset = Dataset(self.logger, self.dataset_prefix / "trajectories")
        self.rollout = self.dataset[self.rollout_id]
        self.num_steps = len(self.rollout["obs"])

        self.job_queue = JobQueue()
        self._setup_job_queue()

        self.done = False

        self.loaded = False

        print(self.app)

    def _setup_job_queue(self):
        for step in range(self.num_steps):
            self.job_queue.append(step)

    def _get_mesh(self):
        mesh = None
        if self.render_type == "rgb":
            src_rel_path = os.path.relpath(os.path.join(self.prefix, self.dataset_prefix, "assets", self.mesh_src),
                                           self.serve_root)
            mat_rel_path = os.path.relpath(os.path.join(self.prefix, self.dataset_prefix, "assets", self.mesh_material),
                                           self.serve_root)
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
            trimesh = o3d.io.read_triangle_mesh(
                os.path.join(self.serve_root, self.prefix, self.dataset_prefix, "assets", self.mesh_src))
            mesh = TriMesh(
                key="terrain",
                vertices=np.array(trimesh.vertices),
                faces=np.array(trimesh.triangles),
                position=self.mesh_position,
                rotation=np.deg2rad(self.mesh_rotation),
                materialType="depth",
                onLoad="loaded mesh"
            )

        else:
            raise NotImplementedError

        return mesh

    def _save(self):
        if self.save_images:
            self.frames = [self.frames[i] for i in range(len(self.frames))]

            print(f"saving {len(self.frames)} total frames to video")
            with self.logger.Prefix(str(self.dataset_prefix / "ego_views")):
                self.logger.save_video(self.frames, f"{self.render_type}_{self.rollout_id:04d}.mp4", fps=CameraArgs.fps)

                for step, img in enumerate(self.frames):
                    self.logger.save_image(img, f"{self.render_type}_{self.rollout_id:04d}/frame_{step:04d}.png")

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

        mat = get_three_mat(cam_position, [r, p, y])
        # sess.upsert @ (
        #     Go1(
        #         f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
        #         joint_values,
        #         global_rotation=(r, p, y),
        #         position=position,
        #     ),
        # )

        sess.update @ CameraView(**vars(CameraArgs), matrix=mat)

    async def main(self, sess: VuerSession):
        print("hi")
        mesh = self._get_mesh()
        sess.set @ DefaultScene(
            group(
                group(
                    mesh,
                    rotation=[0, 0, -np.pi / 2],
                ),
                position=self.gym_tf_pos,
            ),
            rawChildren=[
                CameraView(
                    **vars(CameraArgs),
                ),
            ],
            up=[0, 0, 1],
            show_helper=False,
        )
        print(f"numsteps", self.num_steps)

        await sleep(2.0)

        while not self.loaded:
            await sleep(0.1)

        while len(self.job_queue) > 0:
            # print("jobs left:", len(self.job_queue))
            # print("total results", len(self.frames))
            job, mark_done, put_back = self.job_queue.take()

            id = job['job_params']

            try:
                # print("Took frame", job['job_params'])
                self._handle_step(id, sess)
                await sleep(0.2)
                event = await sess.grab_render(key=CameraArgs.key, quality=1.0)

                buff = event.value["frame"]
                img = np.array(PImage.open(BytesIO(buff)))
                self.frames[id] = img

                mark_done()
                self.job_queue.house_keeping()
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
