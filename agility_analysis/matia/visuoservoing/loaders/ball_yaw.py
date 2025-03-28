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


class BallYaw(PrefixProto, cli=False):
    data_path = Proto(
        env="$DATASETS/lucidsim/scenes/ball/scene_00005",
        help="path to the rollout file. Casted to a pathlib.Path object.",
        dtype=Path,
    )
    image_path = Proto("depth")
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
        traj_list = loader.glob("*.pkl", wd=self.trajectory_path)
        traj_list = sorted(traj_list)

        if self.debug:
            traj_list = traj_list[:1]
        else:
            traj_list = traj_list[1:2]

        def load_images(file_paths, prefix="."):
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

        # todo: change this to state, obs, action in a future iteration
        data = []
        targets = []

        it = tqdm(traj_list, desc="Trajectory")

        for i, traj_file in enumerate(it):

            traj_id: str = Path(traj_file).stem.split("_")[-1]
            with loader.Prefix("trajectories"):
                (traj_data,) = loader.load_pkl(traj_file)
                df = pd.DataFrame(traj_data)
                delta_yaw = df["obs"].apply(lambda x: x[0, 7]).to_list()

            # we only load the first one for now

            image_mask = None
            if self.image_path == "lucid_dreams":
                image_mask = f"lucid_dreams/imagen_{traj_id}_0000/*.png"
            elif self.image_path == "rgb" or self.image_path == "depth":
                image_mask = f"ego_views/{self.image_path}_{traj_id}/*.png"
            else:
                raise NotImplementedError

            image_list = loader.glob(image_mask)
            image_list = sorted(image_list)
            images = load_images(image_list)

            data.append(images)
            targets.append(delta_yaw)

            assert len(images) == len(delta_yaw)

        data = np.concatenate(data)

        targets = np.concatenate(targets)

        self.data = torch.from_numpy(data)
        self.data = torch.permute(self.data, [0, 3, 1, 2]).contiguous()
        # self.target = torch.LongTensor(self.targets).contiguous()
        self.target = torch.from_numpy(targets).contiguous()

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
            targets = self.target[batch].to(self.device, dtype=torch.float32)[..., None]
            yield images, targets


if __name__ == "__main__":
    import torchvision.transforms.v2 as T

    pipeline = [
        T.Resize((60, 96), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop((60, 96), padding=4),  # SVGA = (600, 960)
        # T.RandomHorizontalFlip(),
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(
            mean=[118.0995, 113.6374, 97.3486], std=[55.2961, 52.6547, 49.0021]
        ),
    ]

    transform = T.Compose(pipeline)

    dataset = BallYaw(transform=transform, device="cuda", debug=False)

    i = 1
    for im, t in tqdm(dataset.sample_batch(batch_size=1), desc="Training..."):
        import matplotlib.pyplot as plt

        a = im[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(a / 3 + 0.5)
        plt.text(0, -5, t[0], size=20)

        plt.show()
        i += 1
        if i > 10:
            break

# if __name__ == "__main__":
#     from asyncio import sleep
#     from vuer import Vuer, VuerSession
#     from vuer.schemas import DefaultScene, Arrow
# 
#     app = Vuer()
# 
# 
#     @app.spawn(start=True)
#     async def visualize(sess: VuerSession):
#         sess.set @ DefaultScene()
#         await sleep(0.1)
# 
#         for i in range(100):
#             await sleep(0.001)
#             # if i % 10 != 0:
#             #     continue
# 
#             l = float(dataset.target[i])
# 
#             sess.upsert @ Arrow(
#                 position=[0.1 * i, 0, 0],
#                 direction=[0, l, 0],
#                 key=f"heading-{i}",
#             )
# 
#         while True:
#             await sleep(1)
