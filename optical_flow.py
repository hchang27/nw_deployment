from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as F2
from PIL import Image
from params_proto import ParamsProto
from torchtyping import TensorType
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image


class RunArgs(ParamsProto, cli=False):
    im_root: Path = Path("/home/exx/")

    source_images: List[str] = ["first.jpg"]
    target_images: List[str] = ["second.jpg"]
    test_img: str = "generated.png"

    save_path = None

    resize_to = (1024, 1024)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    debug = False


def warp(x, flo):
    """
    From RAFT: https://github.com/princeton-vl/RAFT/issues/64

    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow (1 -> 2)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size())
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.to("cuda")
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def warp_forward(x, flo, return_mask=False):
    """
    warp an image/tensor (im1) to im2, according to the optical flow
    x: [B, C, H, W] (im1)
    flo: [B, 2, H, W] flow (1 -> 2)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    mask = torch.ones(x.size())

    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.to("cuda")
    # Invert the flow by multiplying by -1
    vgrid = grid - flo  # Change here
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    if return_mask:
        return output, mask
    return output


def plot(imgs, **imshow_kwargs):
    import matplotlib.pyplot as plt

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            # img = F2.to_pil_image(img.to("cpu"))
            img = img.permute(1, 2, 0).cpu()

            if img.dtype == torch.float32:
                img = (img + 1) / 2

            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


def preprocess(
    batch: TensorType["batch", "channel", "height", "width"],
) -> TensorType["batch", "channel", "resize_height", "resize_width", torch.float32]:
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=RunArgs.resize_to),
        ]
    )
    batch = transforms(batch)
    return batch


def load_images(images: List[str], root: Path) -> TensorType["batch", "channel", "height", "width"]:
    images = np.concatenate([np.array(Image.open(root / image))[None, ..., :3] for image in images])
    frames = torch.from_numpy(images).permute(0, 3, 1, 2)
    return frames


def compute_consistency_mask(target, predicted, threshold=0.2):
    delta = torch.norm(target - predicted, dim=1)
    consistency_mask = torch.abs(delta < threshold)
    return consistency_mask.float()


def entrypoint():
    import matplotlib.pyplot as plt

    model = raft_large(pretrained=True, progress=False).to(RunArgs.device)
    model.eval()

    source_frames = load_images(RunArgs.source_images, RunArgs.im_root)
    target_frames = load_images(RunArgs.target_images, RunArgs.im_root)

    if RunArgs.debug:
        plot(source_frames)
        plt.show()
        plot(target_frames)
        plt.show()

    source_batch = preprocess(source_frames).to(RunArgs.device)
    target_batch = preprocess(target_frames).to(RunArgs.device)

    flow = model(target_batch, source_batch)[-1]

    flow_imgs = flow_to_image(flow)
    if RunArgs.debug:
        grid = [[source_img, flow_img] for (source_img, flow_img) in zip(source_batch, flow_imgs)]
        plot(grid)
        plt.show()

    warped = warp(source_batch, flow).detach()
    if RunArgs.debug:
        grid = [[source_img, warped_img] for (source_img, warped_img) in zip(source_batch, warped)]
        plot(grid)
        plt.show()

    masks = compute_consistency_mask(target_batch, warped)
    masked_warped = warped * masks[:, None, :, :]
    plot(masked_warped)
    plt.show()

    # test images now
    test_frame = load_images([RunArgs.test_img], RunArgs.im_root)
    test_batch = preprocess(test_frame).to(RunArgs.device)

    # iteratively apply the predicted flows to the test image
    current_test_batch = test_batch
    warp_results = [current_test_batch.clone()]
    masked_warp_results = [current_test_batch.clone()]
    for i in range(len(flow_imgs)):
        current_test_batch = warp(current_test_batch, flow[i]).detach()
        warp_results.append(current_test_batch.detach().clone())
        masked_warp_results.append(current_test_batch.detach().clone() * masks[i, None, :, :])

    warp_results = torch.cat(warp_results, dim=0)
    masked_warp_results = torch.cat(masked_warp_results, dim=0)

    if RunArgs.debug:
        grid = [[original_img, generated_img] for (original_img, generated_img) in zip(target_batch, warp_results)]
        plot(grid)
        plt.show()

    # save the generated images
    if RunArgs.save_path:
        for i, generated_img in enumerate(warp_results):
            generated_img = F2.to_pil_image((generated_img.to("cpu") + 1) / 2)
            generated_img.save(RunArgs.save_path / f"{i}.png")
        for i, generated_img in enumerate(masked_warp_results):
            generated_img = F2.to_pil_image((generated_img.to("cpu") + 1) / 2)
            generated_img.save(RunArgs.save_path / f"{i}_masked.png")
        for i, mask in enumerate(masks):
            mask = F2.to_pil_image(mask.to("cpu"))
            mask.save(RunArgs.save_path / f"{i}_mask.png")


if __name__ == "__main__":
    RunArgs.im_root = Path("/home/exx/source_imgs")

    folder_images = ["01.jpeg", "02.jpeg"]  # , "03.jpeg", "04.jpeg", "05.jpeg", "06.jpeg", "07.jpeg", "08.jpeg", "09.jpeg"]
    RunArgs.source_images = folder_images[:-1]
    RunArgs.target_images = folder_images[1:]

    RunArgs.test_img = "generated.png"

    RunArgs.resize_to = (512, 512)

    RunArgs.debug = True
    RunArgs.save_path = Path("/home/exx/raft_results")

    entrypoint()
