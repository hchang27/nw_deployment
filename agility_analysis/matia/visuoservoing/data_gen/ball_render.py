import asyncio
import os
from functools import partial, lru_cache
from io import BytesIO

import numpy as np
from ml_logger import ML_Logger
from vuer import Vuer
from vuer.schemas import DefaultScene, CameraView, Sphere

from lucidsim_old.job_queue import JobQueue

logger = ML_Logger(root=os.getcwd(), prefix="assets")

PORT = 8013
app = Vuer(port=PORT, uri=f"ws://localhost:{PORT}", queries={"background": "000000"})


def transform_points(pts, matrix):
    # Convert the list of points to a numpy array with an additional dimension for the homogeneous coordinate
    pts_homogeneous = np.hstack((pts, np.ones((len(pts), 1))))

    # Apply the transformation matrix to each point
    transformed_pts = pts_homogeneous @ matrix

    # Convert back to 3D points from homogeneous coordinates
    transformed_pts = transformed_pts[:, :3]
    return transformed_pts


def get_expanded_fov(expansion_factor, fov):
    """
    Compute the expanded field of view given the expansion factor and the original field of view.
    
    Expansion factor: the ratio of the new pixel width to the old pixel width.
    """

    return 2 * np.arctan(np.tan(fov / 2) * expansion_factor)


@lru_cache(maxsize=1)
def get_logger():
    from ml_logger import ML_Logger

    data_logger = ML_Logger(
        root="http://luma01.csail.mit.edu:4000",
        # prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/{datetime.now():%Y%m%d-%H%M%S}",
        prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v7",
    )
    print(data_logger.get_dash_url())
    return data_logger


def setup_jobs() -> JobQueue:
    # fmt: off
    matrix = np.array([
        -0.9403771820302098, -0.33677144289058686, 0.04770482963301034, 0,
        0.14212405695663877, -0.26162828559882034, 0.9546472608292598, 0,
        -0.30901700268934784, 0.9045085048953463, 0.2938925936815643, 0,
        -0.47444114213044175, 1.2453493553603068, 0.5411873913841395, 1,
    ]).reshape(4, 4)
    # fmt: on

    data_logger = get_logger()
    pts, = data_logger.load_pkl("points.pkl")
    pts_cam_frame = transform_points(pts, matrix)

    get_sphere = partial(
        Sphere,
        args=[0.05],
        # position=p.tolist(),
        materialType="depth",
        # materialType="phong",
        material=dict(color="#ff0000"),
        key="ball",
    )

    queue = JobQueue()
    for i, (pt, pt_cam_frame) in enumerate(zip(pts, pts_cam_frame)):
        x, y, _ = pt.tolist()
        queue.append({
            "scene": get_sphere(position=pt_cam_frame.tolist()),
            "x": x,
            "y": y,
            "index": i,
        })

    return matrix, queue


if __name__ == '__main__':

    data_logger = get_logger()
    # data_logger.remove("ego-rgb")
    data_logger.remove("ego-depth")

    matrix, queue = setup_jobs()


    async def house_keeping():
        while queue:
            print(f"queue size: {len(queue)}")
            queue.house_keeping()
            await asyncio.sleep(5.0)


    app._add_task(house_keeping())


    # We don't auto start the vuer app because we need to bind a handler.
    @app.spawn(start=True)
    async def show_heatmap(proxy):
        print("on connection")

        async def render_fn(scene, x, y, index):
            proxy.upsert @ scene
            await asyncio.sleep(0.2)
            response = await proxy.grab_render(key="ego", downsample=2, quality=1)
            buff = response.value["frame"]
            # data_logger.upload_buffer(buffer=buff, key=f"ego-rgb/frame_{index:04d}.jpg")
            data_logger.upload_buffer(buffer=buff, key=f"ego-depth/frame_{index:04d}.jpg")

        print("on connection")
        proxy.set @ DefaultScene(
            rawChildren=[
                CameraView(
                    fov=50,
                    width=320,
                    height=240,
                    key="ego",
                    matrix=matrix.flatten().tolist(),
                    movable=False,
                    stream="ondemand",
                    fps=120,
                    near=0.4,
                    far=1.25,
                    showFrustum=True,
                    downsample=1,
                    distanceToCamera=2,
                ),
            ],
            # hide the helper to only render the objects.
            grid=False,
            show_helper=False,
        )
        await asyncio.sleep(0.01)

        while queue:
            job, done, reset = queue.take()
            if job is None:
                break

            try:
                await render_fn(**job["job_params"])
                done()
            except asyncio.exceptions.TimeoutError:
                reset()
                print("timeout")
                continue
            except Exception as e:
                reset()
                print('shit happened')
                raise e
