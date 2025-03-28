import torch
from torch import nn
from cxx.modules.actor_critic import get_activation


class SimpleVisionHead(nn.Module):
    def __init__(self, vision_backbone: nn.Module, activation_fn: str, n_proprio=53, n_priv=9):
        super().__init__()
        self.vision_backbone = vision_backbone
        # self.n_proprio = n_proprio
        # self.n_priv = n_priv
        activation_fn = get_activation(activation_fn)

        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio + n_priv, 128),
            activation_fn,
            nn.Linear(128, 32)
        )

    def forward(self, camera, obs_prop_priv):
        vision_latent = self.vision_backbone(camera)
        vision_latent = self.combination_mlp(torch.cat((vision_latent, obs_prop_priv), dim=-1))
        return vision_latent
