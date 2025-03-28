import asyncio
import os

import numpy as np
import cv2 as cv
import torchvision.transforms.v2 as T

from params_proto import ParamsProto
from vuer import Vuer, VuerSession
from vuer.schemas import ImageBackground, DefaultScene, Arrow
from vuer.serdes import jpg
from matplotlib import pyplot as plt
import torch


def get_transform():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    return T.Compose([
        T.Resize((40, 60), interpolation=T.InterpolationMode.BILINEAR),
        *pipeline
    ])


app = Vuer()

from ml_logger import logger

prefix = prefix = "/alanyu/scratch/2024/01-24/202046/checkpoints/net_last.pts"

model = logger.torch_jit_load(prefix, map_location='cpu')
model.eval()


@app.spawn(start=True)
async def show_heatmap(session: VuerSession):
    session.set @ DefaultScene()

    cap = cv.VideoCapture(0)

    preprocess_transform = get_transform()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # cv.imshow('frame', frame)
        frame = np.array(frame, dtype=np.uint8)
        frame = torch.from_numpy(frame)[None, ...].permute(0, 3, 1, 2).contiguous()

        frame_processed = preprocess_transform(frame)

        # plt.imshow(frame_processed[0].permute(1, 2, 0));
        # plt.show()
        output = model(frame_processed)
        print(output)

        session.upsert(
            ImageBackground(
                # src=jpg(rgb, 99),
                src=jpg(img, 99),
                # depthSrc=jpg(depth),
                key="image",
            ),
            to="bgChildren",
        )

        await asyncio.sleep(0.01)

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


app.run()
