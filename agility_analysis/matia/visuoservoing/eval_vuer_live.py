import asyncio

import cv2.gapi
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

import cv2


class Args(ParamsProto):
    log_root = "http://luma01.csail.mit.edu:4000"
    prefix = "lucidsim/experiments/matia/visuoservoing/ball_gen/ball-eval-v7"

    # model_path = "/alanyu/scratch/2024/01-25/183233/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-25/232534/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-26/001507/checkpoints/net_1000.pts"
    model_path = "/alanyu/scratch/2024/01-26/215708/checkpoints/net_1000.pts"


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


CAMERA_MATRIX = None
CURRENT_STEP = 0

cap = cv2.VideoCapture(0)


@app.spawn
async def main(sess: VuerSession):
    global PREDICTED, IMAGES

    from ml_logger import logger
    loader = ML_Logger(root=Args.log_root, prefix=Args.prefix)

    print('here')

    model = logger.torch_jit_load(Args.model_path, map_location="cpu")
    model.eval()

    # resize_tf = T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR)

    tf = transform()

    print('here')

    sess.set @ DefaultScene(
        grid=False,
        show_helper=False,
    )

    await asyncio.sleep(1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

        print('og shape', img.shape, img.dtype)

        img = img[..., :3]
        data = torch.from_numpy(img)[None, ...].permute(0, 3, 1, 2).contiguous()

        data = tf(data)

        print("post tf", data.shape, data.dtype)

        predicted = model(data) / 2
        print('predicted', predicted)
        global CAMERA_MATRIX

        # print("resize shape", img.shape)
        matrix = np.array(CAMERA_MATRIX).reshape(4, 4)

        predicted_location = predicted.cpu().detach().numpy()

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
                src=jpg(img, 50),
                key='background'
            )
        )

        await asyncio.sleep(0.05)


@app.add_handler("CAMERA_MOVE")
async def cam_move(event, sess):
    if event.key == "defaultCamera":
        global CAMERA_MATRIX
        CAMERA_MATRIX = event.value['camera']["matrix"]


app.run()
