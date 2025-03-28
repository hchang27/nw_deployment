import numpy as np
from vuer import VuerSession
from vuer.events import ClientEvent

from agility_analysis.matia.visuoservoing.so3 import transform_points


def render(ball_pts):
    import asyncio

    import numpy as np
    from vuer import Vuer
    from vuer.schemas import DefaultScene, CameraView, Sphere

    app = Vuer()

    # this is the default camera pose.
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
        nonlocal matrix
        # only intercept the ego camera.
        if event.key != "ego":
            return
        if event.value["matrix"] is None:
            return
        new_matrix = np.array(event.value["matrix"]).reshape(4, 4)
        if not np.allclose(new_matrix, matrix):
            print("matrix has changed")
            matrix = new_matrix

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
                    matrix=matrix.flatten().tolist(),
                    stream=False,
                    fps=30,
                    near=0.4,
                    far=1.8,
                    showFrustum=True,
                    downsample=1,
                    distanceToCamera=2,
                    # movable=False,
                ),
            ],
            # hide the helper to only render the objects.
            grid=False,
            show_helper=False,
        )
        last_id = None
        while True:

            if last_id and id(matrix) == last_id:
                await asyncio.sleep(0.016)
                continue

            last_id = id(matrix)

            print("original points")
            print(ball_pts[0, :5])

            pts = transform_points(ball_pts, matrix)
            print("update balls")
            print(pts[0, :5])
            print("matrix")
            print(matrix[:5])

            proxy.upsert @ [
                Sphere(
                    args=[0.05],
                    position=(pts[i]).tolist(),
                    material=dict(color="red"),
                    materialType="phong",
                    key=f"ball-{i}",
                )
                for i, p in enumerate(ball_pts)
            ]

            await asyncio.sleep(0.001)


if __name__ == "__main__":
    # from agility_analysis.matia.visuoservoing.data_gen.frustum_sampling import frustum_sampling
    #
    # ball_pts = frustum_sampling(
    #     fov=50,
    #     width=320,
    #     height=240,
    #     near=0.45,
    #     far=0.5,
    #     num_samples=100,
    # )
    from ml_logger import ML_Logger

    data_logger = ML_Logger(
        root="http://luma01.csail.mit.edu:4000",
        # prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/{datetime.now():%Y%m%d-%H%M%S}",
        prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
    )
    print(data_logger.get_dash_url())

    points, = data_logger.load_pkl("points.pkl")

    render(points)
