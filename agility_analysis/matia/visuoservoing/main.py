import os
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torchvision.transforms.v2 as T
from params_proto import Flag, ParamsProto
from torch.utils.data import DataLoader
from torchinfo import summary

from agility_analysis.matia.cifar10.models import MODEL_TYPES
from agility_analysis.matia.visuoservoing.loaders.ball_xy import BallXY
from agility_analysis.matia.visuoservoing.loaders.ball_xy_yaw import BallXYYaw
from agility_analysis.matia.visuoservoing.loaders.ball_xyd import BallXYD
from agility_analysis.matia.visuoservoing.loaders.ball_yaw import BallYaw
from agility_analysis.matia.visuoservoing.model import get_soda_model


class Params(ParamsProto):
    local_dataset_root: str = os.environ["DATASETS"]
    seed = 100

    # num_classes = 10
    # data_aug: List[Literal["crop", "rotate", "color", "perspective", "blur"]] = ["crop", "rotate", "color",
    #                                                                              "perspective", "blur"]
    arch = 'soda'
    data_aug = ["crop", "color", "perspective"]

    batch_size = 32
    n_epochs = 1_000

    arch: MODEL_TYPES = "SimpleDLA"
    """Model types, limited to the following: SimpleDLA, SENet, PreActResNet, ResNet, WideResNet..."""

    image_path = "lucid_dreams"
    optimizer = "adam"
    lr = 0.0001
    lr_schedule = False
    momentum = 0.7
    weight_decay = 5e-4
    # deprecated. Does not apply to the y prediction in ball_xy.
    # symmetry_coef = 0.

    dataset: Literal['xy', 'yaw', 'xyyaw', 'xyd'] = "xy"

    eval_full_trainset = Flag(
        "Whether to re-evaluate the full train set on a fixed model, or simply report "
        "the running average of training statistics"
    )

    checkpoint_stops = None
    checkpoint_interval = 5_00

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = None


def get_ball_data():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    print('using imagenet normalization')
    from ml_logger.job import RUN

    augs = {
        "crop": T.RandomCrop((60, 80), padding=4),
        "rotate": T.RandomRotation(degrees=10),
        "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
        "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    }

    aug_pipeline = [augs[aug] for aug in Params.data_aug]

    train_loader = BallXY(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-train-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *aug_pipeline,
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )
    test_loader = BallXY(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )

    return train_loader, test_loader


def get_ball_xyz_data():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    from ml_logger.job import RUN

    augs = {"crop": T.RandomCrop((60, 80), padding=4),
            "rotate": T.RandomRotation(degrees=10),
            "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
            "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            }

    aug_pipeline = [augs[aug] for aug in Params.data_aug]

    print(aug_pipeline)

    train_loader = BallXYD(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-train-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *aug_pipeline,
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )
    test_loader = BallXYD(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )

    return train_loader, test_loader


def get_ball_xyyaw_data():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    from ml_logger.job import RUN

    augs = {"crop": T.RandomCrop((60, 80), padding=4),
            "rotate": T.RandomRotation(degrees=10),
            "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
            "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            }

    aug_pipeline = [augs[aug] for aug in Params.data_aug]

    print(aug_pipeline)

    train_loader = BallXYYaw(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-train-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *aug_pipeline,
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )
    test_loader = BallXYYaw(
        data_path=f"{Params.local_dataset_root}/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )

    return train_loader, test_loader


def get_ball_yaw_data():
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    augs = {"crop": T.RandomCrop((60, 80), padding=4),
            "rotate": T.RandomRotation(degrees=10),
            "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
            "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            }

    aug_pipeline = [augs[aug] for aug in Params.data_aug]

    from ml_logger.job import RUN

    train_loader = BallYaw(
        data_path=f"{Params.local_dataset_root}/lucidsim/scenes/ball/scene_00005",
        debug=True or RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *aug_pipeline,
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )
    test_loader = BallYaw(
        data_path=f"{Params.local_dataset_root}/lucidsim/scenes/ball/scene_00005",
        debug=RUN.debug,
        transform=T.Compose([
            T.Resize((60, 80), interpolation=T.InterpolationMode.BILINEAR),
            *pipeline
        ]),
        device=Params.device,
        image_path=Params.image_path,
    )

    return train_loader, test_loader


def train(train_loader, model, criterion, optimizer):
    """Train for one epoch on the training set"""
    from ml_logger import logger

    # switch to train mode
    model.train()

    for i, (image, target) in enumerate(
            train_loader.sample_batch(Params.batch_size, True)
    ):
        output = model(image)
        optimizer.zero_grad()
        target *= 2
        loss = F.mse_loss(output, target)

        loss.backward()

        optimizer.step()

        logger.store_metrics({"train/loss": loss.item()})


def evaluate(val_loader: DataLoader, model, criterion):
    """Perform validation on the validation set"""
    from ml_logger import logger

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        for image, target in val_loader.sample_batch(Params.batch_size, True):
            output = model(image)
            target *= 2
            loss = F.mse_loss(output, target)
            logger.store_metrics({"eval/loss": loss.item()})


def main(**deps):
    from ml_logger import logger

    print(logger.get_dash_url())

    Params._update(deps)
    logger.job_started(Params=vars(Params))
    logger.upload_file(__file__)

    np.random.seed(Params.seed)
    torch.random.manual_seed(Params.seed)

    # this is not the bottleneck. Data loading is.
    torch.set_float32_matmul_precision("medium")  # or high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.log_text(
        """
        charts:
        - yKeys: ["train/loss/mean", "eval/loss/mean"]
          xKey: epoch
        """,
        dedent=True,
        filename=".charts.yml",
        overwrite=True,
    )

    logger.print("Loading data...")

    if Params.dataset == 'xy':
        train_loader, val_loader = get_ball_data()
        projection_dim = 2
    elif Params.dataset == 'xyyaw':
        train_loader, val_loader = get_ball_xyyaw_data()
        projection_dim = 1
    elif Params.dataset == 'yaw':
        train_loader, val_loader = get_ball_yaw_data()
        projection_dim = 1
    elif Params.dataset == 'xyd':
        train_loader, val_loader = get_ball_xyz_data()
        projection_dim = 3
    else:
        raise NotImplementedError

    logger.print("Loading model...")

    Params.arch = "vit"

    if Params.arch == "soda":
        model = get_soda_model(inp_shape=(3, 60, 80), device=Params.device, projection_dim=projection_dim, coord_conv=False)

    elif Params.arch == "vit":
        from vit_pytorch.vit_for_small_dataset import ViT

        model = ViT(
            image_size=(60, 80),
            patch_size=10,
            num_classes=projection_dim,
            dim=64,
            depth=3,
            heads=4,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1,
        )

    elif Params.arch == "decision-transformer":
        import gym
        from transformers import DecisionTransformerConfig, DecisionTransformerModel

        gym.make("Hopper-v3")

        config = DecisionTransformerConfig(
            state_dim=17,
            act_dim=4,
            hidden_size=128,
            max_ep_len=4096,
            action_tanh=True,
            vocab_size=50257,
            n_positions=1024,
            n_layer=3,
            n_head=1,
            n_inner=None,
            activation_function="relu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )
        model = DecisionTransformerModel(config)
        report = summary(model, input_size=(1, 1, 21))

    report = summary(model, input_size=(32, 3, 60, 80))
    logger.print(report)
    logger.log(f"Number of parameters: {sum([p.data.nelement() for p in model.parameters()]):d}")

    criterion = F.mse_loss

    if Params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            Params.lr,
        )
    elif Params.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            Params.lr,
            momentum=Params.momentum,
            weight_decay=Params.weight_decay,
        )
    else:
        raise NotImplementedError

    if Params.lr_schedule:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Params.n_epochs
        )

    logger.start("epoch")
    for epoch in range(0, Params.n_epochs + 1):
        if Params.lr_schedule:
            schedule.step(epoch)
            # logger.log(lr=schedule.get_lr(epoch))

        if Params.checkpoint_interval and epoch and epoch % Params.checkpoint_interval == 0:
            print("Saving checkpoints...")
            logger.torch_save(model, f"checkpoints/net_{epoch}.pts")
            logger.duplicate(f"checkpoints/net_{epoch}.pts", "checkpoints/net_last.pts")

        train(train_loader, model, criterion, optimizer)

        if val_loader is not None:
            evaluate(val_loader, model, criterion)

        logger.log_metrics_summary(
            key_values={"epoch": epoch, "dt_epoch": logger.split("epoch")}
        )


if __name__ == "__main__":
    main()
