import torchvision.transforms.v2 as T
import numpy as np
from PIL import Image
from ml_logger import ML_Logger
import torch
from matplotlib import pyplot as plt
from params_proto import ParamsProto
from tqdm import tqdm


class Args(ParamsProto):
    log_root = "http://luma01.csail.mit.edu:4000"
    prefix = "lucidsim/experiments/matia/visuoservoing/ball_gen/ball-eval-v7"

    # model_path = "/alanyu/scratch/2024/01-25/183233/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-25/232534/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-26/001507/checkpoints/net_500.pts"
    # model_path = "/alanyu/scratch/2024/01-26/220519/checkpoints/net_1000.pts"
    # model_path = "/alanyu/scratch/2024/01-26/225802/checkpoints/net_500.pts"
    model_path = "/alanyu/scratch/2024/01-27/165816/checkpoints/net_1000.pts"


# load model


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
        # T.ToDtype(torch.float32),
        # T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return T.Compose([
        T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
        *pipeline
    ])


def main():
    from ml_logger import logger
    loader = ML_Logger(root=Args.log_root, prefix=Args.prefix)

    model = logger.torch_jit_load(Args.model_path).cpu()
    model.eval()

    image_paths = sorted(loader.glob("ego-rgb/*.jpg"))
    images = load_images(image_paths, loader)

    tf = transform()

    for i, image in enumerate(images):
        image = image[..., :3]
        data = torch.from_numpy(image)[None, ...].permute(0, 3, 1, 2).contiguous()

        data = tf(data)

        print(i, model(data) / 2)


if __name__ == '__main__':
    main()

