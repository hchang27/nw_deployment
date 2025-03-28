import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size in {84, 100}, f"unexpected size: {size}"
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, "input must be a 4D tensor"
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f"unexpected size: {x.size(3)}"
        if self.size == 84:
            p = 8
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim), nn.Tanh()
            # nn.Linear(in_shape[0], out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SODAMLP(nn.Module):
    def __init__(self, projection_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)


class CoordConv(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert len(size) == 2, "size must be a tuple of length 2, h, w."

        h, w = size

        xs = torch.linspace(-1, 1, w)
        ys = torch.linspace(-1, 1, h)
        y_grid, x_grid = torch.meshgrid(ys, xs)

        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(self, x):
        repeated = self.coords.repeat(x.size(0), 1, 1, 1)
        x = torch.cat([x, repeated], dim=1)
        return x


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, coord_conv=False):
        super().__init__()
        assert len(obs_shape) == 3
        obs_channels, *hw = obs_shape

        self.num_layers = num_layers
        self.num_filters = num_filters

        if coord_conv:
            input_dim = obs_channels + 2

            self.layers = [
                # centercrop(size=84),
                NormalizeImg(),
                CoordConv(size=hw),
                nn.Conv2d(input_dim, num_filters, 3, stride=2),
            ]
        else:
            input_dim = obs_channels

            self.layers = [
                # centercrop(size=84),
                NormalizeImg(),
                nn.Conv2d(input_dim, num_filters, 3, stride=2),
            ]

        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
            self,
            shared_cnn: SharedCNN,
            head_cnn: HeadCNN,
            projection: RLProjection,
    ):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, image):
        x = self.shared_cnn(image)
        x = self.head_cnn(x)
        return self.projection(x)


import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.AvgPool2d(2, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 10, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print("===>", x.shape)
        x = x.view(-1, 128 * 5 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def get_simple_model(device):
    net = SimpleCNN()
    net.to(device)
    return net


def get_soda_model(
        inp_shape=(3, 60, 96),
        coord_conv=True,
        num_shared_layers=10,
        num_filters=32,
        num_head_layers=0,
        projection_dim=2,
        device="cuda:0",
) -> Encoder:
    """Returns a two-staged network. Can be fused into one."""
    shared_cnn = SharedCNN(inp_shape, num_shared_layers, num_filters, coord_conv=coord_conv)
    head_cnn = HeadCNN(shared_cnn.out_shape, num_head_layers, num_filters)

    encoder = Encoder(
        shared_cnn,
        head_cnn,
        # RLProjection(head_cnn.out_shape, projection_dim),
        SODAMLP(head_cnn.out_shape[0], num_filters, projection_dim),
    )
    encoder.to(device)

    return encoder


def get_vit():
    pass