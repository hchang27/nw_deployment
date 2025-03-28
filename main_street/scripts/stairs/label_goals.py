"""
Add markers on the terrain for auto-collection, which are saved as YAML
"""

from asyncio import sleep

import numpy as np
import open3d as o3d
from params_proto import ParamsProto
from typing import List

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, PointCloud, PointerControls, group

from main_street.scripts.utils import cube, sphere
from ml_logger import logger

counter = 0


class Args(ParamsProto):
    # mesh_src = "/Users/alanyu/Downloads/red_stairs_v2/textured.obj"
    mesh_src = "/home/exx/red_stairs_v2/textured.obj"
    mesh_tf_pos = [0, 0, 1.4]
    mesh_tf_rot = [90, 0, 0]

    # dataset_prefix = "/lucid-sim/lucid-sim/datasets/stairs/debug/00001/"
    dataset_prefix = "/lucid-sim/lucid-sim/datasets/stairs/debug/00002/"


if Args.mesh_src.endswith(".obj"):
    mesh = o3d.io.read_triangle_mesh(Args.mesh_src, True)
    point_cloud = mesh.sample_points_poisson_disk(50_000)
else:
    point_cloud = o3d.io.read_point_cloud(Args.mesh_src)


class Labels:
    starts: List[List[float]] = []
    goals: List[List[float]] = []
    data: List[dict] = []

    @staticmethod
    def process_point(position, vuer_session):
        """
        Adds the new point to start / end and data, and adds the geometry to the current VuerSession
        """
        if len(Labels.goals) < len(Labels.starts):
            Labels.goals.append([position["x"], position["y"], position["z"]])
            Labels.data[-1]["goal"] = Labels.goals[-1]
            vuer_session.upsert @ cube(key=f"goal_{len(Labels.goals) - 1}", position=Labels.goals[-1])
        elif len(Labels.starts) == len(Labels.goals):
            Labels.starts.append([position["x"], position["y"], position["z"]])
            Labels.data.append({"start": Labels.starts[-1]})
            vuer_session.upsert @ sphere(key=f"start_{len(Labels.starts) - 1}", position=Labels.starts[-1])
        else:
            raise ValueError

        with logger.Prefix(Args.dataset_prefix):
            logger.save_yaml(Labels.data, "labels.yaml")


app = Vuer()


@app.spawn
async def main(sess: VuerSession):
    with logger.Prefix(Args.dataset_prefix):
        print(f"Logging to {logger.get_dash_url()}")
        logger.log_params(labels=vars(Args))

    sess.set @ DefaultScene(
        group(
            PointCloud(
                key="pc",
                vertices=np.array(point_cloud.points),
                position=Args.mesh_tf_pos,
                rotation=np.deg2rad(Args.mesh_tf_rot),
                size=0.01,
            ),
            rotation=[0, 0, -np.pi / 2],
        ),
        PointerControls(),
        up=[0, 0, 1],
    )

    while True:
        await sleep(0.1)


async def set_goal(event, session):
    # this triggers twice :(
    global counter
    counter += 1
    if counter % 2 == 1:
        return

    position = event.value["position"]
    Labels.process_point(position, session)

    print(Labels.starts, "\n", Labels.goals, "\n")


app.add_handler("ADD_MARKER", set_goal)
app.run()
