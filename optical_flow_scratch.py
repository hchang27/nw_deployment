from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.transforms as T
from PIL import Image

from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F2.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(1024, 1024)),
        ]
    )
    batch = transforms(batch)
    return batch


im_root = Path("/home/exx/")

images = ["first.jpg", "second.jpg"]

images = np.concatenate([np.array(Image.open(im_root / image))[None, ..., :3] for image in images])
frames = torch.from_numpy(images).permute(0, 3, 1, 2)
plot(frames)
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

batch = preprocess(frames).to(device)


model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

# result = model(batch[0:1], batch[1:2])
result = model(batch[1:2], batch[0:1])
result_img = flow_to_image(result[-1])
plt.imshow(result_img[0].permute(1, 2, 0).cpu())
plt.show()


def warp(x, flo):
    """
    From RAFT: https://github.com/princeton-vl/RAFT/issues/64
    
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to("cuda")
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


x = (warp(batch[-1:], result[-1]) + 1) / 2
plt.imshow(x[0].detach().cpu().permute(1, 2, 0))
plt.show()

y = (warp(batch[-2:-1], result[-1]) + 1) / 2
plt.imshow(y[0].detach().cpu().permute(1, 2, 0))
plt.show()

test_image = np.array(Image.open(im_root / 'generated.png'))[None, ..., :3]
test_frame = torch.from_numpy(test_image).permute(0, 3, 1, 2)

test_frame.shape

plot(test_frame); plt.show()

test_batch = preprocess(test_frame).to(device)
test = (warp(test_batch, result[-1]) + 1) / 2
plt.imshow(test[0].detach().cpu().permute(1, 2, 0)); plt.show()
# plot(test[0].detach().cpu().permute(1, 2, 0)); plt.show()
