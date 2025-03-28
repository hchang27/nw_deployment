import asyncio
import time

import numpy as np
import torch
import torchvision

# zed = ZedCamera(fps=120, mode="PERFORMANCE")
# zed.share_buffers("rgb", "depth")
# zed.spin_process("rgb")
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, ImageBackground
from vuer.serdes import jpg

# from go1_gym_deploy.modules.base.zed_node import ZedCamera
from go1_gym_deploy.modules.base.usb_cam import USBCamera

# 
# rs = RealSenseCamera(res=(360, 640))
# print('hey')
# rs = RealSenseCamera()
# rs.spin_process("depth")
# print('done')
# rs = ZedCamera(mode="QUALITY")
# rs.spin_process("depth")
# 
# while True:
#     frame = rs.frame["rgb"]
#     # cv2.imshow("rgb", frame)
#     print(frame.shape)
#     if cv2.waitKey(1) == ord('q'):
#         break
# 
# exit()

usb = USBCamera(res=(1080, 1920), fps=30)
usb.spin_process()

time.sleep(1)

app = Vuer(
    uri="ws://localhost:8112",
    queries=dict(
        reconnect=True,
        grid=False,
        backgroundColor="black",
    ),
    port=8112,
)

# resize_transform = torchvision.transforms.Resize((58, 87),
#                                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
resize_transform = torchvision.transforms.Resize((45, 80),
                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

NEAR_CLIP = 0.28


def process_depth(depth):
    """
    Process the ZED depth image by:
    1. Converting into meters [ already done in zed_node init_params] 
    2. Invert 
    2. Clip distances 
    3. Resize 
    4. Normalize  
    """
    if depth is None:
        return

    depth = torch.from_numpy(depth)

    # depth = -depth

    # crop
    depth = _crop_depth_image(depth)

    # clip
    # depth = torch.clip(depth, -cfg.depth.far_clip, -cfg.depth.near_clip)
    depth = torch.clip(depth, NEAR_CLIP, 2)
    depth = resize_transform(depth[None, :]).squeeze()
    depth = _normalize_depth_image(depth)

    return depth


def _crop_depth_image(depth_image):
    # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
    return depth_image[:-2, 4:-4]


def _normalize_depth_image(depth_image):
    # depth_image = depth_image * -1
    depth_image = (depth_image - NEAR_CLIP) / (2) - 0.5
    return depth_image


@app.spawn(start=True)
async def show_heatmap(session: VuerSession):
    session.set @ DefaultScene()

    def clean_depth(depth, max=1, min=0):
        depth = depth.copy()
        depth = np.nan_to_num(depth)
        depth[depth >= max] = max
        depth[depth <= min] = min
        return depth

    i = 0
    while True:
        # rgb = rs.frame["depth"]
        # depth = rs.frame['depth']

        rgb = usb.frame["rgb"]
        # 
        # depth_inp = process_depth(depth)
        # 
        # depth = clean_depth(depth)
        # depth = (depth / 2 * 255).astype(np.uint8)

        session.upsert(
            ImageBackground(
                src=jpg(rgb, 50),
                # src=jpg(depth, 99),
                # depthSrc=jpg(depth),
                key="image",
            ),
            to="bgChildren",
        )
        await asyncio.sleep(0.01)
