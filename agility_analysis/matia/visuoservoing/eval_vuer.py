import asyncio

import torchvision.transforms.v2 as T
import numpy as np
from PIL import Image
from ml_logger import ML_Logger
import torch
from matplotlib import pyplot as plt
from params_proto import ParamsProto
from tqdm import tqdm
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, CameraView, Sphere, TimelineControls, ImageBackground
from vuer.serdes import jpg


class Args(ParamsProto):
    log_root = "http://luma01.csail.mit.edu:4000"
    prefix = "lucidsim/experiments/matia/visuoservoing/ball_gen/ball-eval-v7"

    # model_path = "/alanyu/scratch/2024/01-25/183233/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-25/232534/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-26/001507/checkpoints/net_1000.pts"
    model_path = "/alanyu/scratch/2024/01-26/225802/checkpoints/net_500.pts"


# app = Vuer(
#     uri="ws://localhost:8112",
#     queries=dict(
#         reconnect=True,
#         grid=False,
#         backgroundColor="black",
#     ),
#     port=8112,
# )
app = Vuer()

def load_images(file_paths, loader, prefix="."):
    """This is way too much data to load into memory."""
    images = []
    with loader.Prefix(prefix):
        for file_path in tqdm(file_paths, desc="Loading images"):
            buff = loader.load_file(file_path)
            image = Image.open(buff)
            image = image.convert("RGB")
            image_np = np.array(image, dtype=np.uint8)
            images.append(image_np)

    return images


def transform():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    return T.Compose([
        T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
        *pipeline
    ])


PREDICTED = []
IMAGES = []

CAMERA_MATRIX = None
CURRENT_STEP = 0
print('yo')


@app.spawn
async def main(sess: VuerSession):
    global PREDICTED, IMAGES

    from ml_logger import logger
    loader = ML_Logger(root=Args.log_root, prefix=Args.prefix)

    print('here')

    model = logger.torch_jit_load(Args.model_path).cpu()
    model.eval()

    image_paths = sorted(loader.glob("ego-rgb/*.jpg"))
    images = load_images(image_paths, loader)

    resize_tf = T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR)
    IMAGES = [np.array(resize_tf(Image.fromarray(image))) for image in images]

    tf = transform()

    print('here')

    for i, image in enumerate(images):
        image = image[..., :3]
        data = torch.from_numpy(image)[None, ...].permute(0, 3, 1, 2).contiguous()

        data = tf(data)

        PREDICTED.append(model(data) / 2)

    sess.set @ DefaultScene(
        TimelineControls(start=0, end=len(PREDICTED) - 2),
        grid=False,
        show_helper=False,
    )

    while True:
        await asyncio.sleep(0.1)


def handle_step(step, sess):
    global PREDICTED, CAMERA_MATRIX, IMAGES

    matrix = np.array(CAMERA_MATRIX).reshape(4, 4)

    predicted_location = PREDICTED[step].detach().numpy()

    # add homogenous coordinate
    predicted_location = np.concatenate([predicted_location, np.ones((1, 1))], axis=-1)

    # project back into world frame
    pred_world = matrix.T @ predicted_location.T
    # pred_world = np.linalg.inv(matrix) @ predicted_location.T
    pred_world = pred_world[:3, 0]

    sess.upsert @ (
        Sphere(
            key="PREDICTED",
            args=[0.1, 32, 32],
            position=pred_world.tolist(),
        ),
        ImageBackground(
            src=jpg(IMAGES[step], 50),
            key='background'
        )
    )
    return pred_world


async def step_handler(event, sess):
    global CURRENT_STEP
    CURRENT_STEP = event.value["step"]
    print("step", CURRENT_STEP)
    handle_step(CURRENT_STEP, sess)


@app.add_handler("CAMERA_MOVE")
async def cam_move(event, sess):
    if event.key == "defaultCamera":
        global CAMERA_MATRIX
        CAMERA_MATRIX = event.value['camera']["matrix"]
        handle_step(CURRENT_STEP, sess)


app.add_handler("TIMELINE_STEP", step_handler)
app.run()
