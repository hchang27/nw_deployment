from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from math import ceil
from ml_logger import ML_Logger
from params_proto import Proto, Flag, PrefixProto
from PIL import Image
from torch import Tensor
from tqdm import tqdm


def batched_quat_rotate_inverse_np(q, v):
    q_w = q[:, :, -1:]
    q_vec = q[:, :, :3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0

    prod = q_vec[:, :, None, :] @ v[:, :, :, None]

    c = 2 * q_vec * prod[:, :, :, 0]

    return a - b + c


def vector_to_euler(vectors):
    rolls = np.arctan2(vectors[:, 1], vectors[:, 0])
    pitches = np.arcsin(vectors[:, 2])
    yaws = np.arctan2(np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2), vectors[:, 2])
    return rolls, pitches, yaws


def vector_to_intrinsic_euler(vec):
    norm_vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    rolls = np.arctan2(norm_vec[:, 1], norm_vec[:, 0])

    pitches = np.arcsin(norm_vec[:, 2])
    yaws = np.arctan2(
        np.sqrt(norm_vec[:, 0] ** 2 + norm_vec[:, 1] ** 2), norm_vec[:, 2]
    )
    return np.stack([rolls, pitches, yaws]).T


def center_crop(img, new_width, new_height):
    """
    Crops the given NumPy image array to the specified width and height centered around the middle of the image.

    Parameters:
    img (numpy.ndarray): The image to be cropped (assumed to be in HxWxC format).
    new_width (int): The desired width of the cropped image.
    new_height (int): The desired height of the cropped image.

    Returns:
    numpy.ndarray: The cropped image.
    """

    height, width = img.shape[:2]

    # Calculate the starting points (top-left corner) of the crop
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    # Perform the crop
    cropped_img = img[start_y:start_y + new_height, start_x:start_x + new_width]

    return cropped_img


def load_images(file_paths, *, prefix=".", loader: ML_Logger, image_size=(240, 320)):
    """This is way too much data to load into memory."""
    images = []
    with loader.Prefix(prefix):
        for file_path in tqdm(file_paths, desc="Loading images"):
            buff = loader.load_file(file_path)
            image = Image.open(buff)
            image = image.convert("RGB")

            image = image.resize((512, 512), Image.BILINEAR)

            image_np = np.array(image, dtype=np.uint8)

            # center crop the image to the desired size
            image_np = center_crop(image_np, image_size[1], image_size[0])

            images.append(image_np)

    return images


class BallXYYaw(PrefixProto, cli=False):
    data_path = Proto(
        env="$DATASETS/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
        help="path to the rollout file. Casted to a pathlib.Path object.",
        dtype=Path,
    )
    image_path = Proto("lucid_dreams")
    trajectory_path = Proto("trajectories")
    device = "cuda"
    debug = Flag(help="debug mode")

    transform: Callable[[Tensor], Tensor] = None
    """This is a transform function that is applied to the images before they are returned."""

    def __post_init__(self, _deps=None):

        if self.debug:
            print(
                "DEBUG MODE, only loads a small subset of data. Turn off Args.debug to load all data."
            )

        # load_images()
        loader = ML_Logger(root=self.data_path)
        # with loader.Prefix("ego-depth"):
        #     loader.make_video("*.jpg", "video.mp4", fps=30)
        # import time
        # print("now sleep...")
        # time.sleep(10)
        # exit()
        img_mask = None
        if self.image_path == "lucid_dreams":
            image_path = self.image_path
            img_mask = "imagen_**/*.png"
        elif self.image_path == "rgb" or self.image_path == "depth":
            image_path = f"ego-{self.image_path}"
            img_mask = "*.jpg"
        else:
            raise NotImplementedError

        image_list = loader.glob(img_mask, wd=image_path)
        image_list = sorted(image_list)

        points, = loader.load_pkl("points.pkl")
        images = load_images(image_list, prefix=image_path, loader=loader)

        # todo: change this to state, obs, action in a future iteration
        data = torch.from_numpy(np.stack(images))
        xyz = torch.from_numpy(points)

        yaw = torch.arctan(-xyz[:, 0] / xyz[:, 2])[..., None]

        # for lucid_dreams, we need to multiply the labels for however many generation trials there were .
        if image_path == "lucid_dreams":
            num_trials = len(image_list) / len(yaw)
            assert num_trials.is_integer(), "the number of trials should be an integer."
            num_trials = int(num_trials)
            yaw = yaw.repeat(num_trials, 1)
        assert len(data) == len(yaw), "the data should align."

        self.data = torch.permute(data, [0, 3, 1, 2]).contiguous()
        self.target = yaw.contiguous()

        # Testing this out
        # self.data = self.data.to(device)
        # self.target = self.target.to(device)
        self.indices = torch.arange(len(self.data))

    def sample_batch(self, batch_size, shuffle=True):
        """Returns an iterator that yields batches of images and targets."""
        if shuffle:
            self.indices = torch.randperm(len(self.data))

        n_chunks = ceil(len(self.data) / batch_size)

        for batch in torch.chunk(self.indices, n_chunks):
            images = self.data[batch].to(self.device)

            if self.transform is not None:
                images = self.transform(images)

            # INFO: cast to full precision, need to fix data @alany1
            targets = self.target[batch].to(self.device, dtype=torch.float32)
            yield images, targets


if __name__ == "__main__":
    import torchvision.transforms.v2 as T

    pipeline = [
        T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop((60, 80), padding=4),  # SVGA = (600, 960)
        # T.RandomHorizontalFlip(),
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(
            mean=[118.0995, 113.6374, 97.3486], std=[55.2961, 52.6547, 49.0021]
        ),
    ]

    transform = T.Compose(pipeline)

    dataset = BallXYYaw(transform=transform, device="cuda", debug=True)

    i = 1
    for im, t in tqdm(dataset.sample_batch(batch_size=1, shuffle=True), desc="Training..."):
        import matplotlib.pyplot as plt

        a = im[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(a / 3 + 0.5)
        plt.text(0, -5, t[0], size=20)

        plt.show()
        i += 1
        if i > 10:
            break

if __name__ == "__main__":
    from asyncio import sleep
    from vuer import Vuer, VuerSession
    from vuer.schemas import DefaultScene, Arrow

    app = Vuer(port=8014)


    @app.spawn(start=True)
    async def visualize(sess: VuerSession):
        sess.set @ DefaultScene()
        await sleep(0.1)

        for i in range(len(dataset.data)):
            await sleep(0.001)
            # if i % 10 != 0:
            #     continue

            l = dataset.target[i].tolist()

            sess.upsert @ Arrow(
                position=[0.1 * i, 0, 0],
                direction=[0, l[0], l[1]],
                key=f"heading-{i}",
            )

        while True:
            await sleep(1)
