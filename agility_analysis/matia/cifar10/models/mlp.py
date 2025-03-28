import copy

import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLP(nn.Module):
    def __init__(self, layer_dims: List[int] = [512], init_scale=1.0):
        super().__init__()

        self._n_units = copy.copy(layer_dims)
        self._layers = []
        for i in range(1, len(layer_dims)):
            layer = nn.Linear(layer_dims[i - 1], layer_dims[i], bias=False)
            variance = math.sqrt(2.0 / (layer_dims[i - 1] + layer_dims[i]))
            layer.weight.data.normal_(0.0, init_scale * variance)
            self._layers.append(layer)

            name = "fc%d" % i
            if i == len(layer_dims) - 1:
                name = "fc"  # the prediction layer is just called fc
            self.add_module(name, layer)

    def forward(self, x):
        x = x.view(-1, self._n_units[0])
        out = self._layers[0](x)
        for layer in self._layers[1:]:
            out = F.relu(out)
            out = layer(out)
        return out
