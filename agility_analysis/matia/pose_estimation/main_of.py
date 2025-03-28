import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torchvision.transforms.v2 as T
from params_proto import ParamsProto, Proto, Flag, PrefixProto
from torch.utils.data import DataLoader
from typing import List, Literal

from agility_analysis.matia.cifar10.models import MODEL_TYPES

from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig
from agility_analysis.matia.pose_estimation.loaders.of_pose_loader import OFPoseLoader
from agility_analysis.matia.pose_estimation.loaders.pose_loader import PoseLoader
from agility_analysis.matia.visuoservoing.model import get_soda_model


class TrainCfg(PrefixProto):
    root = "http://luma01.csail.mit.edu:4000"

    prefix = "scenes"
    scenes: List[str] = ["mit_stairs/stairs_0001_v1"]

    trajectories_per_scene = 1
    image_type: Literal["rgb", "depth", "augmented"] = "rgb"

    # data_aug = ["crop"]
    data_aug = []

    # model
    image_size = (45, 80)

    # training params
    n_epochs = 1_000
    batch_size = 32
    shuffle = False  # can't shuffle for recurrent models 

    # optimization
    optimizer = "adam"
    lr = 0.0005
    lr_schedule = True
    """Not being used, enforced by default."""
    momentum = 0.9
    weight_decay = 5e-4

    # checkpointing
    checkpoint_interval: int = None  # n_epochs // 2

    max_grad_norm: float = 1.0

    rotation_weight = 10.0

    position_scale = 1.0
    rotation_scale = 10

    # system
    seed = 100
    device = "cuda:0"

    model_checkpoint = None

    # teacher_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"
    teacher_checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-21/17.29.34/17.29.34/1/"


def get_pose_data(shuffle_train=True):
    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
    ]

    augs = {"crop": T.RandomCrop((45, 80), padding=4),
            "rotate": T.RandomRotation(degrees=10),
            "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
            "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            }

    aug_pipeline = [augs[aug] for aug in TrainCfg.data_aug]

    train_loader = OFPoseLoader(
        **TrainCfg.__dict__,
        **Go1RGBConfig.__dict__,
        transform=T.Compose([
            T.Resize((45, 80), interpolation=T.InterpolationMode.BILINEAR),
            *aug_pipeline,
            *pipeline
        ]),
    )
    # test_loader = PoseLoader(
    #     datapath="/home/geyang/datasets/lucidsim/lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v1",
    #     debug=RUN.debug,
    #     transform=T.Compose([
    #         T.Resize((45, 80), interpolation=T.InterpolationMode.BILINEAR),
    #         *pipeline
    #     ]),
    #     device=Params.device,
    # )

    return train_loader, None


def train(train_loader, model, optimizer):
    """Train for one epoch on the training set"""
    from ml_logger import logger

    # criterion = F.mse_loss

    # switch to train mode
    model.train()

    for i, (image, flow_map, target) in enumerate(
            train_loader.sample_batch(TrainCfg.batch_size, True)
    ):
        # image_stacked = torch.cat([image[:, 0], image[:, 1]], dim=1)

        image_flow = torch.cat([image[:, 0], flow_map], dim=1)

        output = model(image_flow)

        # target *= 100
        # loss = position_loss = F.huber_loss(output[:, :1], TrainCfg.position_scale * target[:, :1])
        # rotation_loss = criterion(output[:, 3:], TrainCfg.rotation_scale * target[:, 3:])

        # yaw
        loss = F.mse_loss(output[:, 5:6], 10 * target[:, 5:6])

        # loss = position_loss + TrainCfg.rotation_weight * rotation_loss

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainCfg.max_grad_norm)

        optimizer.zero_grad()
        # loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        logger.store_metrics({"train/loss": loss.item()})


def evaluate(val_loader: DataLoader, model):
    """Perform validation on the validation set"""
    from ml_logger import logger

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        for image, target in val_loader:
            output = model(image)
            target *= 2
            loss = F.mse_loss(output, target)
            logger.store_metrics({"eval/loss": loss.item()})


def main(**deps):
    from ml_logger import logger

    print(logger.get_dash_url())

    TrainCfg._update(deps)
    logger.job_started(TrainCfg=vars(TrainCfg))

    np.random.seed(TrainCfg.seed)
    torch.random.manual_seed(TrainCfg.seed)

    # this is not the bottleneck. Data loading is.
    torch.set_float32_matmul_precision("medium")  # or high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    logger.log_text(
        """
        charts:
        - yKey: "train/loss/mean"
          xKey: epoch
        """,
        dedent=True,
        filename=".charts.yml",
        overwrite=True,
    )

    logger.print("Loading data...")
    train_loader, val_loader = get_pose_data(shuffle_train=True)

    logger.print("Loading model...")
    if TrainCfg.model_checkpoint is not None:
        from ml_logger import logger
        model = logger.torch_load(TrainCfg.model_checkpoint, map_location=TrainCfg.device)
    else:
        # model = get_soda_model(inp_shape=(2 * 1, 45, 80), projection_dim=6, device=TrainCfg.device)
        model = get_soda_model(inp_shape=(5, 45, 80), device=TrainCfg.device, projection_dim=6,
                               coord_conv=False)

    # model = get_simple_model(device=TrainCfg.device)

    # data = torch.randn(64, 3, 45, 80).to(TrainCfg.device)
    # model(data).shape
    logger.print("model has been loaded")
    logger.log(
        f"Number of parameters: {sum([p.data.nelement() for p in model.parameters()]):d}"
    )

    if TrainCfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            TrainCfg.lr,
        )
    elif TrainCfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            TrainCfg.lr,
            momentum=TrainCfg.momentum,
            weight_decay=TrainCfg.weight_decay,
        )
    else:
        raise NotImplementedError

    if TrainCfg.lr_schedule:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TrainCfg.n_epochs
        )

    logger.start("epoch")
    for epoch in range(0, TrainCfg.n_epochs + 1):
        # print('Yo')
        if TrainCfg.lr_schedule:
            schedule.step(epoch)

        if TrainCfg.checkpoint_interval and epoch % TrainCfg.checkpoint_interval == 0:
            print("Saving checkpoints...")
            logger.torch_jit_save(model, f"checkpoints/net_{epoch}.pts")
            logger.duplicate(f"checkpoints/net_{epoch}.pts", f"checkpoints/net_last.pts")

        train(train_loader, model, optimizer)

        if val_loader is not None:
            evaluate(val_loader, model)

        logger.log_metrics_summary(
            key_values={"epoch": epoch, "dt_epoch": logger.split("epoch")}
        )


if __name__ == "__main__":
    main()
