import torchvision.transforms.v2 as T
import numpy as np
from PIL import Image
from ml_logger import logger
import torch
from matplotlib import pyplot as plt

# prefix = "/alanyu/scratch/2024/01-24/193401/checkpoints/net_last.pts"
# prefix = "/alanyu/scratch/2024/01-24/201001/checkpoints/net_last.pts"
# prefix = "/alanyu/scratch/2024/01-24/202046/checkpoints/net_last.pts"
# prefix = "/lucid-sim/matia/analysis/ball_xy_yaw/lucid_dreams/large_symmetry/32/checkpoints/net_1000.pts"
# prefix = "/alanyu/scratch/2024/01-25/183233/checkpoints/net_500.pts"
prefix = "/alanyu/scratch/2024/01-25/221632/checkpoints/net_500.pts"
# prefix = "/alanyu/scratch/2024/01-24/210020/checkpoints/net_last.pts"
model = logger.torch_jit_load(prefix, map_location="cpu")
model.eval()




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


img_path = "/Users/alanyu/Downloads/rs4.png"

# from PIL import Image
# from pillow_heif import register_heif_opener

# register_heif_opener()

# image = Image.open('image.heic')

# img_path = "/Users/alanyu/Downloads/IMG_6786.HEIC"
img = Image.open(img_path)

plt.imshow(img);
plt.show()

img = np.array(img, dtype=np.uint8)[..., :3]
data = torch.from_numpy(img)[None, ...].permute(0, 3, 1, 2).contiguous()

data = transform()(data)

plt.imshow(data[0].permute(1, 2, 0));
plt.show()
print(model(data) / 2)

data = torch.flip(data, dims=[3])
