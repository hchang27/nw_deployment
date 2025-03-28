import os
from asyncio import sleep
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
import torch
from params_proto import PrefixProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, DefaultScene, Obj, TriMesh, group

from lucidsim_old import ROBOT_DIR
from lucidsim_old.dataset import Dataset
from lucidsim_old.utils import ISAAC_DOF_NAMES, Go1, euler_from_quaternion, get_three_mat, quat_rotate


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
    cam_to_base = [0.29, 0.0, 0.02]
    movable = True
    monitor = True


class EgoViewRender(PrefixProto):
    prefix = Proto(env="$DATASETS/lucidsim/scenes/experiments/real2real")
    dataset_prefix = Path("stairs/scene_00001")

    mesh_src = "textured.obj"
    mesh_material = "textured.mtl"

    rollout_id = 0
    render_type: Literal["rgb", "depth"] = "rgb"
    save_images: bool = False

    serve_root = Proto(env="$HOME")

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.prefix)

        CameraArgs._update(deps)
        self.frames = []

        queries = dict(grid=False)
        if self.render_type == "depth":
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

        self.loaded = False

        print(self.app)

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
                onLoad="loaded mesh",
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
            print(f"saving {len(self.frames)} total frames to video")
            with self.logger.Prefix(str(self.dataset_prefix / "ego_views")):
                self.logger.save_video(self.frames, f"{self.render_type}_{self.rollout_id:04d}.mp4", fps=CameraArgs.fps)
                print(f"Done saving video to the dataset folder: {self.logger.get_dash_url()}")

    def _handle_step(self, step, sess):
        joint_values = {name: angle for name, angle in zip(ISAAC_DOF_NAMES, self.rollout["dofs"][step])}

        quat_t = torch.from_numpy(self.rollout["states"][step][None, 3:7])
        global_rot = euler_from_quaternion(quat_t.float())
        r, p, y = (angle.item() for angle in global_rot)

        position = self.rollout["states"][step][:3]
        cam_position = position.astype(np.float32) + \
                       quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
                           0].numpy()

        position = position.tolist()
        cam_position = cam_position.tolist()

        mat = get_three_mat(cam_position, [r, p, y])
        sess.upsert @ (
            Go1(
                f"http://localhost:8012/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf",
                joint_values,
                global_rotation=(r, p, y),
                position=position,
            ),
        )

        sess.update @ CameraView(**vars(CameraArgs), matrix=mat)

    async def main(self, sess: VuerSession):
        print("hi")
        mesh = self._get_mesh()
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
        print("numsteps", self.num_steps)

        while not self.loaded:
            await sleep(0.1)

        while True:
            try:
                await sess.grab_render(key=CameraArgs.key, quality=0.9)
                print("Ready")
                await sleep(0.01)
                break
            except TimeoutError:
                print("setup timeout")

        self._handle_step(0, sess)
        while True:
            step = 0
            while step < self.num_steps:
                try:
                    self._handle_step(step, sess)
                    step += 1
                except TimeoutError:
                    print("timeout")
                await sleep(0.02)

        self._save()

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
