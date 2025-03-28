import asyncio
import os

import numpy as np
from ml_logger import ML_Logger
from typing import Tuple
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent, Upsert
from vuer.schemas import DefaultScene, CameraView, Sphere


logger = ML_Logger(root=os.getcwd(), prefix="assets")

app = Vuer()

# fmt: off
matrix = np.array([
    -0.9403771820302098, -0.33677144289058686, 0.04770482963301034, 0,
    0.14212405695663877, -0.26162828559882034, 0.9546472608292598, 0,
    -0.30901700268934784, 0.9045085048953463, 0.2938925936815643, 0,
    -0.47444114213044175, 1.2453493553603068, 0.5411873913841395, 1,
]).reshape(4, 4)
# fmt: on


@app.add_handler("CAMERA_MOVE")
async def track_movement(event: ClientEvent, sess: VuerSession):
    global matrix
    # only intercept the ego camera.
    if event.key != "ego":
        return
    if event.value["matrix"] is None:
        return
    new_matrix = np.array(event.value["matrix"]).reshape(4, 4)
    if not np.allclose(new_matrix, matrix):
        print("matrix has changed")
        matrix = new_matrix


def transform_points(pts, matrix):
    # Convert the list of points to a numpy array with an additional dimension for the homogeneous coordinate
    pts_homogeneous = np.hstack((pts, np.ones((len(pts), 1))))

    # Apply the transformation matrix to each point
    transformed_pts = pts_homogeneous @ matrix

    # Convert back to 3D points from homogeneous coordinates
    transformed_pts = transformed_pts[:, :3]
    return transformed_pts


class State:
    def __init__(self):
        self.ball = Sphere()
        self.camera = CameraView()

    def update(
        self,
        ball_pt: np.ndarray,
        camera_matrix: np.ndarray,
    ):
        self.ball.position = ball_pt.tolist()
        self.camera.matrix = camera_matrix.flatten().tolist()

        return Upsert(self.ball, self.camera)


class RenderQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.proxy = None
        self.task = None

    async def render(self, event):
        await self.queue.put(event)
        return await self.queue.get()

    async def _render(self):
        while True:
            event = await self.queue.get()
            await self.proxy.send(event)
            response = await self.proxy.grab_render(
                key=self.camera.key, downsample=1, quality=1
            )
            self.queue.task_done()

    def start(self, proxy):
        self.proxy = proxy
        self.task = asyncio.create_task(self._render())


render_queue = RenderQueue()

async def render(self, **kwargs):
    await proxy.send @ event
    response = await proxy.grab_render(key=self.camera.key, downsample=1, quality=1)

async def main():
    state = State()

    ball_pts = sample_camera_frustum_batch(
        fov=50,
        width=320,
        height=240,
        near=0.45,
        far=0.5,
        num_samples=1000,
    )

    for pt in ball_pts:

        event = state.update(pt, matrix)
        image = await render_queue.render(event)




# We don't auto start the vuer app because we need to bind a handler.
@app.spawn(start=True)
async def show_heatmap(proxy):
    proxy.set @ DefaultScene(
        rawChildren=[
            CameraView(
                fov=50,
                width=320,
                height=240,
                key="ego",
                # position=[-0.5, 1.25, 0.5],
                # rotation=[-0.4 * np.pi, -0.1 * np.pi, 0.15 + np.pi],
                matrix=matrix.flatten().tolist(),
                stream="frame",
                fps=30,
                near=0.4,
                far=1.8,
                showFrustum=True,
                downsample=1,
                distanceToCamera=2
                # dpr=1,
            ),
        ],
        # hide the helper to only render the objects.
        grid=False,
        show_helper=False,
    )
    last_id = None
    while True:

        await asyncio.sleep(0.01)
