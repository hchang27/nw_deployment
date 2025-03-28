from typing import List, Tuple, Union

import torch
from torch import nn
from torchtyping import TensorType

from agility_analysis.matia.cifar10.models import MODEL_TYPES, get_model
from cxx.modules import Estimator
from cxx.modules.actor_critic import Actor, get_activation


class BallRGBActorProp(nn.Module):
    def __init__(
            self,
            n_proprio: int,
            n_scan: int,
            num_actions: int,
            scan_encoder_dims: List[int],
            actor_hidden_dims: List[int],
            priv_encoder_dims: List[int],
            estimator_hidden_dims: List[int],
            n_priv_latent: int,
            n_priv: int,
            history_len: int,
            activation_fn: str,
            tanh_encoder_output,
            vision_head: MODEL_TYPES,
            device: str,
            freeze_teacher_modules: bool,
            **kwargs
    ):
        super().__init__()

        self.n_proprio = n_proprio
        self.n_scan = n_scan
        self.num_actions = num_actions
        self.n_priv = n_priv

        self.freeze_teacher_modules = freeze_teacher_modules

        self.vision_latent_dim = scan_encoder_dims[-1]

        activation_fn = get_activation(activation_fn)

        self.device = device

        self.actor = Actor(
            n_proprio=n_proprio,
            n_scan=n_scan,
            num_actions=num_actions,
            scan_encoder_dims=scan_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            priv_encoder_dims=priv_encoder_dims,
            n_priv_latent=n_priv_latent,
            n_priv=n_priv,
            history_len=history_len,
            activation_fn=activation_fn,
            tanh_encoder_output=tanh_encoder_output,
        ).to(self.device)

        self.estimator = Estimator(input_dim=n_proprio, output_dim=n_priv,
                                   hidden_dims=estimator_hidden_dims).to(self.device)

        if isinstance(vision_head, str):
            self.vision_head = get_model(vision_head, device=self.device)
        else:
            self.vision_head = vision_head

        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio + n_priv, 128),
            activation_fn,
            nn.Linear(128, 32 + 1)
        )

    def _parse_ac_params(self, params):
        actor_params = {}
        for k, v in params.items():
            if k.startswith("actor."):
                actor_params[k[6:]] = v
        return actor_params

    def load_teacher_modules(self, logger_prefix: str):
        from ml_logger import logger
        with logger.Prefix(logger_prefix):
            state_dict = logger.torch_load("checkpoints/model_last.pt", map_location=self.device)
            model_dict = self._parse_ac_params(state_dict["model_state_dict"])

            self.actor.load_state_dict(model_dict)
            self.estimator.load_state_dict(state_dict["estimator_state_dict"])

        if self.freeze_teacher_modules:
            for param in self.estimator.parameters():
                param.requires_grad = False
            for param in self.actor.parameters():
                param.requires_grad = False

    def forward(
            self,
            camera: Union[TensorType["batch", "num_channels", "height", "width"], None],
            obs: TensorType["batch", "num_observations"],
            vision_latent: Union[TensorType["batch", "self.vision_latent_dim"], None] = None,
            hist_encoding: bool = True,
    ) -> Tuple[TensorType["batch", 12], TensorType["batch", "depth_latent_dim"]]:
        # assert camera is not None or vision_latent is not None, "Either camera or vision_latent must be provided"

        # mask out the yaw observation

        obs_prop = obs[:, :self.n_proprio].clone()
        yaw = None
        if camera is not None:
            # run inference with the vision head. If no image is provided, then we reuse the previous latent.
            vision_latent = self.vision_head(camera)

            # Avoid inplace modification by creating a new tensor
            obs_prop_masked = obs_prop.clone()
            obs_prop_masked[:, 6:8] = 0  # mask out the yaw observation without modifying obs_prop in-place
            priv_states_estimated = self.estimator(obs_prop_masked.to(torch.float32))
            obs_prop_priv = torch.cat([obs_prop_masked, priv_states_estimated], dim=-1)
            vision_latent_and_yaw = self.combination_mlp(torch.cat((vision_latent, obs_prop_priv), dim=-1))
            vision_latent = vision_latent_and_yaw[:, :-1]
            yaw = 1.5 * vision_latent_and_yaw[:, -1:]

            # Again, avoid modifying the original obs_prop in-place
            obs_prop_with_yaw = obs_prop.clone()
            obs_prop_with_yaw[:, 6:8] = yaw

            obs_prop_scan = torch.cat([obs_prop_with_yaw, obs[:, self.n_proprio: self.n_proprio + self.n_scan]], dim=-1)

        start_idx = self.n_proprio + self.n_scan
        end_idx = self.n_proprio + self.n_scan + self.n_priv
        obs_prop_scan = torch.cat([obs_prop, obs[:, self.n_proprio: start_idx]], dim=-1)
        tail = obs[:, end_idx:]
        estimated_obs = torch.cat([obs_prop_scan, priv_states_estimated, tail], dim=1)

        actions = self.actor(estimated_obs, hist_encoding=hist_encoding, scandots_latent=vision_latent)

        obs_scan = obs[:, self.n_proprio:self.n_proprio + self.n_scan]
        teacher_scandots_latent = self.actor.scan_encoder(obs_scan).detach()  # i think this is right?

        return actions, vision_latent, teacher_scandots_latent, yaw
