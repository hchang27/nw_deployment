from typing import Literal

from cxx.modules.depth_backbone import DepthOnlyFCBackbone, RecurrentDepthBackbone

MODELS = dict(
    cxx_base=DepthOnlyFCBackbone,
    cxx_rnn=RecurrentDepthBackbone
)

MODEL_TYPES = Literal[
    "cxx_base",
    "cxx_rnn",
]


def get_model(arch: MODEL_TYPES, *args, device=None, **kwargs):
    print("==> Building model..")
    try:
        model = MODELS[arch](*args, **kwargs)
    except KeyError:
        raise RuntimeError(f"Network architecture {arch} is not supported")

    if device is not None:
        model.to(device)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    return model
