import os
from asyncio import sleep

import numpy as np
from params_proto import PrefixProto, Proto
from typing import List

from vuer import Vuer, VuerSession

import open3d as o3d
from vuer.schemas import DefaultScene, group, PointCloud, PointerControls

from lucidsim_old.utils import cube, sphere


class Labeler(PrefixProto):
    # prefix = Proto(env="$DATASETS/lucidsim/scenes/")
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"

    local_dataset_root = Proto(env="$DATASETS/lucidsim/scenes/")

    mesh_src = "textured.obj"
    mesh_position = [0, 0, 1.4]
    """The translation of the mesh in the world"""
    mesh_rotation = [90, 0, 0]
    """The rotation of the mesh in the world"""

    overwrite_existing = True

    dataset_prefix = "experiments/real2real/stairs/scene_00003"

    def __post_init__(self, _deps=None):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        if self.mesh_src.endswith(".obj"):
            mesh = o3d.io.read_triangle_mesh(
                os.path.join(self.local_dataset_root, self.dataset_prefix, "assets", self.mesh_src),
                True)
            self.point_cloud = mesh.sample_points_poisson_disk(50_000)
        else:
            self.point_cloud = o3d.io.read_point_cloud(
                os.path.join(self.local_dataset_root, self.dataset_prefix, "assets", self.mesh_src))

        with self.logger.Prefix(self.dataset_prefix):
            existing_labels = self.logger.glob("labels.yaml")
            if existing_labels:
                print(f"Found existing labels: {existing_labels[0]}. To overwrite, set overwrite_existing=True")
                params = self.logger.read_params("labels")
                self.mesh_position = params["mesh_position"]
                self.mesh_rotation = params["mesh_rotation"]
                if not self.overwrite_existing:
                    self.data = self.logger.load_yaml(existing_labels[0])
                    self.starts = [d["start"] for d in self.data]
                    self.goals = [d["goal"] for d in self.data]
                else:
                    self.starts: List[List[float]] = []
                    self.goals: List[List[float]] = []
                    self.data: List[dict] = []
            else:
                self.starts: List[List[float]] = []
                self.goals: List[List[float]] = []
                self.data: List[dict] = []

                with self.logger.Prefix(self.dataset_prefix):
                    print(f"Logging to {self.logger.get_dash_url()}")
                    self.logger.log_params(labels=vars(self))

        self.app = Vuer()

        self.counter = 0

    def process_point(self, position, vuer_session):
        """
        Adds the new point to start / end and data, and adds the geometry to the current VuerSession
        """
        if len(self.goals) < len(self.starts):
            self.goals.append([position["x"], position["y"], position["z"]])
            self.data[-1]["goal"] = self.goals[-1]
            vuer_session.upsert @ cube(key=f"goal_{len(self.goals) - 1}", position=self.goals[-1])
        elif len(self.starts) == len(self.goals):
            self.starts.append([position["x"], position["y"], position["z"]])
            self.data.append({"start": self.starts[-1]})
            vuer_session.upsert @ sphere(key=f"start_{len(self.starts) - 1}", position=self.starts[-1])
        else:
            raise ValueError

        with self.logger.Prefix(self.dataset_prefix):
            self.logger.save_yaml(self.data, "labels.yaml")

    async def set_goal(self, event, session):
        # this triggers twice :(
        self.counter += 1
        if self.counter % 2 == 1:
            return

        position = event.value["position"]
        self.process_point(position, session)

        print(self.starts, "\n", self.goals, "\n")

    async def main(self, sess: VuerSession):
        sess.set @ DefaultScene(
            group(
                PointCloud(
                    key="pc",
                    vertices=np.array(self.point_cloud.points),
                    position=self.mesh_position,
                    rotation=np.deg2rad(self.mesh_rotation),
                    size=0.01,
                ),
                # rotation=[0, 0, -np.pi / 2],
            ),
            PointerControls(),
            up=[0, 0, 1],
        )

        while True:
            await sleep(0.01)

    def __call__(self, *args, **kwargs):
        self.app.spawn(self.main)
        print('spawned')
        self.app.add_handler("ADD_MARKER", self.set_goal)
        print('added handler')
        self.app.run()
        print('ran')


if __name__ == "__main__":
    # labeler = Labeler(dataset_prefix="mit_stairs/stairs_0007_v1", mesh_position=[0, 0, 4.25],
    #                   mesh_rotation=[0, -90, -90])

    labeler = Labeler()
    labeler()
