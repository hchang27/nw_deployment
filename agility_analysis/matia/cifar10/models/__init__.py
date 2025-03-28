from typing import Literal

from .dla_simple import SimpleDLA
from .preact_resnet import PreActResNet18
from .vgg import VGG
from .resnet import ResNet18, ResNet50
from .googlenet import GoogLeNet
from .densenet import DenseNet121
from .resnext import ResNeXt29_2x64d
from .mlp import MLP
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .dpn import DPN92
from .shufflenet import ShuffleNetG2
from .senet import SENet18
from .shufflenetv2 import ShuffleNetV2
from .efficientnet import EfficientNetB0
from .regnet import RegNetX_200MF
from .wideresnet import WideResNet

print("==> Registering model..")
MODELS = dict(
    WideResNet=WideResNet,
    # MLP=MLP,
    VGG=VGG,
    ResNet18=ResNet18,
    ResNet50=ResNet50,
    PreActResNet18=PreActResNet18,
    GoogLeNet=GoogLeNet,
    DenseNet121=DenseNet121,
    ResNeXt29=ResNeXt29_2x64d,
    MobileNet=MobileNet,
    MobileNetV2=MobileNetV2,
    DPN92=DPN92,
    # ShuffleNetG2=ShuffleNetG2,
    SENet18=SENet18,
    ShuffleNetV2=ShuffleNetV2,
    EfficientNetB0=EfficientNetB0,
    RegNetX_200MF=RegNetX_200MF,
    SimpleDLA=SimpleDLA,
)

MODEL_TYPES = Literal[
    "WideResNet",
    "MLP",
    "VGG",
    "ResNet18",
    "ResNet50",
    "PreActResNet18",
    "GoogLeNet",
    "DenseNet121",
    "ResNeXt29",
    "MobileNet",
    "MobileNetV2",
    "DPN92",
    "ShuffleNetG2",
    "SENet18",
    "ShuffleNetV2",
    "EfficientNetB0",
    "RegNetX_200MF",
    "SimpleDLA",
]


def get_model(arch: MODEL_TYPES, device=None):
    print("==> Building model..")
    try:
        model = MODELS[arch]()
    except KeyError:
        raise RuntimeError(f"Network architecture {arch} is not supported")

    if device is not None:
        model.to(device)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    return model
