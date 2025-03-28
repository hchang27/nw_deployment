import numpy as np
from vuer import VuerSession
from vuer.events import ClientEvent
import torch
import torchvision.transforms.v2 as T
from vuer.schemas import SceneBackground

from agility_analysis.matia.visuoservoing.so3 import transform_points

PORT = 8014

def render(ball_pts):
    from ml_logger import logger
    import asyncio

    import numpy as np
    from vuer import Vuer
    from vuer.schemas import DefaultScene, CameraView, Sphere

    app = Vuer(port=PORT, uri=f"ws://localhost:{PORT}")

    # this is the default camera pose.
    # fmt: off
    matrix = np.array([
        -0.9403771820302098, -0.33677144289058686, 0.04770482963301034, 0,
        0.14212405695663877, -0.26162828559882034, 0.9546472608292598, 0,
        -0.30901700268934784, 0.9045085048953463, 0.2938925936815643, 0,
        -0.47444114213044175, 1.2453493553603068, 0.5411873913841395, 1,
    ]).reshape(4, 4)

    pipeline = [
        T.Resize((60, 96), interpolation=T.InterpolationMode.BILINEAR),
        # T.RandomCrop((60, 96), padding=4),
        # T.RandomAutocontrast(),
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]
    transform_train = T.Compose(pipeline)

    checkpoint = "/geyang/scratch/2024/01-22/011320/checkpoints/net_last.pt"
    print("Loading model...")
    load = logger.memoize(logger.torch_load)
    model = load(checkpoint, map_location="cuda")
    print("Model is loaded.")
    # otherwise LayerNorm throws an error
    model.eval()

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
            # print("matrix has changed")
            matrix = new_matrix

    ball_transformed = transform_points(ball_pts, matrix)
    # We don't auto start the vuer app because we need to bind a handler.
    @app.spawn(start=True)
    async def main(proxy):
        proxy.set @ DefaultScene(
            Sphere(
                args=[0.05],
                position=ball_transformed[0].tolist(),
                material=dict(color="red"),
                materialType="phong",
                key=f"ball",
            ),
            rawChildren=[
                CameraView(
                    fov=50,
                    width=320,
                    height=240,
                    key="ego",
                    matrix=matrix.flatten().tolist(),
                    stream="ondemand",
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
            from PIL import Image
            from io import BytesIO

            if last_id and id(matrix) == last_id:
                await asyncio.sleep(0.016)
                continue

            last_id = id(matrix)

            try:
                frame = await proxy.grab_render(key="ego", downsample=1, quality=1)
                img = Image.open(BytesIO(frame.value["frame"])).convert("RGB")
                img_np = np.array(img)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1)[None, ...].to("cuda")
                processed_img = transform_train(img_t)

                # display_img = processed_img.cpu().numpy()[0].transpose(1, 2, 0) + 0.5
                # img_rgb = np.clip(display_img * 255, 0, 255).astype('uint8')
                # 
                # import matplotlib.pyplot as plt
                # plt.imshow(img_rgb)
                # plt.show()
                # 
                # proxy.upsert @ SceneBackground(img_rgb, key="background")
                # await asyncio.sleep(0.01)

                with torch.no_grad():
                    x, y = model(processed_img).cpu().numpy()[0]
                    print(x, y)

                # marker = np.array([x, y, -0.45])[None, :]
                # marker = transform_points(marker, matrix)
                #
                # proxy.upsert @ Sphere(
                #     args=[0.05],
                #     position=marker.tolist(),
                #     material=dict(color="yellow", wireframe=True),
                #     materialType="standard",
                #     key=f"marker",
                # )

            except  asyncio.exceptions.TimeoutError as e:
                print("timeout")
                pass

            await asyncio.sleep(0.01)


if __name__ == "__main__":
    points = np.array([0, 0, - 0.45])[None, :]

    render(points)
