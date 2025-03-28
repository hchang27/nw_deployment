import os
from typing import List, Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
from params_proto import PrefixProto
from torch import nn

from agility_analysis.matia.go1.dataset import LucidDreams
from agility_analysis.matia.go1.models.actor import BasicRGBActor
from agility_analysis.matia.go1.stacked_dataset import StackedLucidDreams
from agility_analysis.matia.visuoservoing.model import get_soda_model
from cxx.modules.depth_backbone import RecurrentDepthBackbone

THIGH_IDXS = [1, 4, 7, 10]

AUGMENTATION_TYPES = Literal["crop", "rotate", "color", "perspective", "blur"]


class TrainCfg(PrefixProto):
    local_dataset_root: str = os.path.join(os.environ["DATASETS"], "lucidsim", "scenes")
    # , "experiments", "real2real")

    scenes = ["hurdle/alan_debug/scene_00022"]
    trajectories_per_scene = [list(range(40))]

    val_scenes = ["hurdle/alan_debug/scene_00022"]
    val_trajectories_per_scene = [list(range(40, 50))]

    image_type: Literal["rgb", "depth", "augmented"] = "depth"

    # model
    # arch = "cxx_base"
    image_size = (45, 80)
    num_filters = 64
    num_shared_layers = 10

    crop_image_size = (720, 1280)

    # training params
    batch_size = 32
    n_epochs = 500
    shuffle = True  # can't shuffle for recurrent models
    max_grad_norm: float = 1.0
    data_aug: List[AUGMENTATION_TYPES] = ["crop", "rotate", "color", "perspective"]

    # optimization
    optimizer = "adam"
    lr = 0.00005
    lr_schedule = True

    momentum = 0.9
    weight_decay = 5e-4

    # checkpointing
    checkpoint_interval: int = 250

    normalize_depth = True

    # the older frames will be fed in as differences with the most recent frame
    recurrent = False
    use_stacked_dreams = True
    compute_deltas = False
    drop_last = False
    stack_size = 10

    # system
    seed = 100
    device = "cuda:0"

    freeze_teacher_modules = False

    gpu_load = True  # whether to load all images into GPU memory

    # teacher_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"
    # teacher_checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-02-20/02.57.22/02.57.22/1"
    teacher_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-20/04.03.35/go1/300/20/0.5"

    imagenet_pipe = True


def fetch_dreams():
    from ml_logger import logger

    # TODO: make sure normalization is correct
    if TrainCfg.imagenet_pipe:
        pipeline = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        pipeline = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=False),
            T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
        ]

    logger.print(f"Using pipeline {pipeline}")
    augs = {
        "crop": T.RandomCrop(TrainCfg.image_size, padding=4),
        "rotate": T.RandomRotation(degrees=10),
        "color": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        "perspective": T.RandomPerspective(distortion_scale=0.2, p=0.3),
        "blur": T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    }

    aug_pipeline = [augs[aug_name] for aug_name in TrainCfg.data_aug]

    transform_train = T.Compose([
        T.Resize(TrainCfg.image_size, interpolation=T.InterpolationMode.BILINEAR),
        *aug_pipeline,
        *pipeline,
    ])

    transform_eval = T.Compose([
        T.Resize(TrainCfg.image_size, interpolation=T.InterpolationMode.BILINEAR),
        *pipeline,
    ])

    val_loader = None

    if TrainCfg.use_stacked_dreams or TrainCfg.recurrent:
        loader_class = StackedLucidDreams
    else:
        loader_class = LucidDreams

    train_loader = loader_class(
        stack_size=TrainCfg.stack_size,
        compute_deltas=TrainCfg.compute_deltas,
        local_dataset_root=TrainCfg.local_dataset_root,
        scenes=TrainCfg.scenes,
        trajectories_per_scene=TrainCfg.trajectories_per_scene,
        image_type=TrainCfg.image_type,
        device=TrainCfg.device,
        transform=transform_train,
        tensor_size=TrainCfg.image_size,
        gpu_load=TrainCfg.gpu_load,
        normalize_depth=TrainCfg.normalize_depth,
        crop_image_size=TrainCfg.crop_image_size,
    )

    if TrainCfg.val_scenes is not None:
        val_loader = loader_class(
            stack_size=TrainCfg.stack_size,
            compute_deltas=TrainCfg.compute_deltas,
            local_dataset_root=TrainCfg.local_dataset_root,
            scenes=TrainCfg.val_scenes,
            trajectories_per_scene=TrainCfg.val_trajectories_per_scene,
            image_type=TrainCfg.image_type,
            device=TrainCfg.device,
            transform=transform_eval,
            tensor_size=TrainCfg.image_size,
            gpu_load=TrainCfg.gpu_load,
            normalize_depth=TrainCfg.normalize_depth,
            crop_image_size=TrainCfg.crop_image_size,
        )

    return train_loader, val_loader


def train_recurrent(train_loader, model, criterion, optimizer):

    from ml_logger import logger

    model.train()

    model.vision_head.hidden_states = None

    for i, (camera, obs, action) in enumerate(train_loader.sample_batch(TrainCfg.batch_size, shuffle=TrainCfg.shuffle)):
        # camera: TensorType["batch", "buffer", 3, "height", "width"]
        batch, buffer, channels, height, width = camera.shape

        teacher_latent_buffer = []
        student_latent_buffer = []
        student_actions_buffer = []

        for t in range(buffer):
            camera_t = camera[:, t, :, :, :]
            # camera_t = camera[:, t, 0, :, :]
            obs_t = obs[:, t, :]

            pred_actions, vision_latent, teacher_scandots_latent = model(camera_t, obs_t)

            teacher_latent_buffer.append(teacher_scandots_latent)
            student_latent_buffer.append(vision_latent)
            student_actions_buffer.append(pred_actions)

        # latents are already detached in forward
        teacher_latent_buffer = torch.stack(teacher_latent_buffer, dim=1)

        student_latent_buffer = torch.stack(student_latent_buffer, dim=1)
        student_actions_buffer = torch.stack(student_actions_buffer, dim=1)

        action_diff = action - student_actions_buffer
        # action_diff[..., THIGH_IDXS] = action_diff[..., THIGH_IDXS] * TrainCfg.thigh_weight
        action_loss = action_diff.norm(p=2, dim=2).mean()
        latent_loss = (student_latent_buffer - teacher_latent_buffer).norm(p=2, dim=2).mean()

        loss = action_loss  # + latent_loss
        # loss = latent_loss

        optimizer.zero_grad()
        loss.backward()

        if TrainCfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), TrainCfg.max_grad_norm)

        optimizer.step()

        logger.store_metrics({"train/action_loss": action_loss.item()})
        logger.store_metrics({"train/latent_loss": latent_loss.item()})

        model.vision_head.detach_hidden_states()
        # model.vision_head.hidden_states = None


def evaluate_recurrent(val_loader, model, criterion):
    from ml_logger import logger

    model.vision_head.hidden_states = None

    with torch.no_grad():
        model.eval()

        for i, (camera, obs, action) in enumerate(
            val_loader.sample_batch(TrainCfg.batch_size, shuffle=TrainCfg.shuffle)
        ):
            # camera: TensorType["batch", "buffer", 3, "height", "width"]
            batch, buffer, channels, height, width = camera.shape

            teacher_latent_buffer = []
            student_latent_buffer = []
            student_actions_buffer = []

            for t in range(buffer):
                camera_t = camera[:, t, :, :, :]
                # camera_t = camera[:, t, 0, :, :]
                obs_t = obs[:, t, :]

                pred_actions, vision_latent, teacher_scandots_latent = model(camera_t, obs_t)

                teacher_latent_buffer.append(teacher_scandots_latent)
                student_latent_buffer.append(vision_latent)
                student_actions_buffer.append(pred_actions)

            teacher_latent_buffer = torch.stack(teacher_latent_buffer, dim=1)
            student_latent_buffer = torch.stack(student_latent_buffer, dim=1)
            student_actions_buffer = torch.stack(student_actions_buffer, dim=1)

            action_loss = (action - student_actions_buffer).norm(p=2, dim=2).mean()
            latent_loss = (student_latent_buffer - teacher_latent_buffer).norm(p=2, dim=2).mean()

            logger.store_metrics({"eval/action_loss": action_loss.item()})
            logger.store_metrics({"eval/latent_loss": latent_loss.item()})

            model.vision_head.detach_hidden_states()
            model.vision_head.hidden_states = None


def train(train_loader, model, criterion, optimizer):
    """Train for one epoch on the training set."""
    from ml_logger import logger

    # switch to train mode
    model.train()

    for i, (camera, obs, action) in enumerate(
        train_loader.sample_batch(TrainCfg.batch_size, shuffle=TrainCfg.shuffle)
    ):
        # form depth buffer
        # TensorType["batch", "buffer", 1, "height", "width"]

        if TrainCfg.use_stacked_dreams:
            batch, stack, channels, height, width = camera.shape

            if TrainCfg.drop_last:
                # drop the most recent frame, and only keep the deltas
                camera = camera[:, :-1, :, :, :]

            camera = camera.reshape(
                batch, (stack - int(TrainCfg.drop_last)) * channels, height, width
            )

            obs = obs[:, -1, :]
            action = action[:, -1, :]

        pred_actions, vision_latent, teacher_scandots_latent = model(camera, obs)

        # TODO: maybe scale the actions here ?
        # Can also supervise on yaw, from the vision head output. not sure if that makes sense though
        # loss = criterion(pred_actions, action)

        # used by cxx; around 0.6 - 0.7 is good
        action_loss = (action - pred_actions).norm(p=2, dim=1).mean()
        latent_loss = (vision_latent - teacher_scandots_latent).norm(p=2, dim=1).mean()

        # loss = latent_loss  # + 0.1 * action_loss
        loss = latent_loss

        optimizer.zero_grad()
        loss.backward()

        if TrainCfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), TrainCfg.max_grad_norm)

        optimizer.step()

        # model.vision_head.detach_hidden_states()
        logger.store_metrics({"train/action_loss": action_loss.item()})
        logger.store_metrics({"train/latent_loss": latent_loss.item()})


def evaluate(val_loader, model, criterion):
    """Perform validation on the validation set."""
    from ml_logger import logger

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        for i, (camera, obs, action) in enumerate(
            val_loader.sample_batch(TrainCfg.batch_size, shuffle=TrainCfg.shuffle)
        ):
            if TrainCfg.use_stacked_dreams:
                batch, stack, channels, height, width = camera.shape

                if TrainCfg.drop_last:
                    # drop the most recent frame, and only keep the deltas
                    camera = camera[:, :-1, :, :, :]

                camera = camera.reshape(
                    batch, (stack - int(TrainCfg.drop_last)) * channels, height, width
                )

                obs = obs[:, -1, :]
                action = action[:, -1, :]

            pred_actions, vision_latent, teacher_scandots_latent = model(camera, obs)

            # used by cxx; around 0.6 - 0.7 is good
            action_loss = (action - pred_actions).norm(p=2, dim=1).mean()
            latent_loss = (vision_latent - teacher_scandots_latent).norm(p=2, dim=1).mean()

            logger.store_metrics({"eval/action_loss": action_loss.item()})
            logger.store_metrics({"eval/latent_loss": latent_loss.item()})


def main(**deps):
    from ml_logger import logger

    from agility_analysis.matia.go1.configs.go1_rgb_config import Go1RGBConfig

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

    logger.log_text(
        """
        charts:
        - yKeys: ["train/action_loss/mean", "eval/action_loss/mean"]
          xKey: epoch
        - yKeys: ["train/latent_loss/mean", "eval/latent_loss/mean"]
          xKey: epoch
        """,
        dedent=True,
        filename=".charts.yml",
        overwrite=True,
    )

    logger.print("Loading data...")
    train_loader, val_loader = fetch_dreams()
    logger.print("Loading model...")

    logger.upload_file(__file__)

    # vision_head = get_model(TrainCfg.arch, prop_dim=Go1RGBConfig.n_proprio,
    #                             scandots_output_dim=Go1RGBConfig.scan_encoder_dims[-1],
    #                             hidden_state_dim=None,
    #                             image_size=TrainCfg.image_size,
    #                             output_activation=get_activation(Go1RGBConfig.activation_fn),
    #                             )

    num_channels = 3
    if TrainCfg.use_stacked_dreams:
        num_channels = 3 * (TrainCfg.stack_size - int(TrainCfg.drop_last))

    if TrainCfg.recurrent:
        num_channels = 3

    vision_head = get_soda_model(
        inp_shape=(num_channels, *TrainCfg.image_size),
        num_filters=TrainCfg.num_filters,
        device=TrainCfg.device,
        projection_dim=Go1RGBConfig.scan_encoder_dims[-1],
        num_shared_layers=TrainCfg.num_shared_layers,
        coord_conv=False,
    )

    # vision_head = DepthOnlyFCBackbone(
    #     None,
    #     32,
    #     None,
    #     image_size=TrainCfg.image_size,
    # )
    #
    # # vision_head = SimpleVisionHead(vision_backbone, n_proprio=Go1RGBConfig.n_proprio,
    # #                                n_priv=Go1RGBConfig.n_priv, activation_fn=Go1RGBConfig.activation_fn
    # #                                )
    #

    if TrainCfg.recurrent:
        vision_head = RecurrentDepthBackbone(
            vision_head, n_proprio=Go1RGBConfig.n_proprio + Go1RGBConfig.n_priv, ignore_yaw=True
        )

    model = BasicRGBActor(
        **vars(Go1RGBConfig),
        vision_head=vision_head,
        freeze_teacher_modules=TrainCfg.freeze_teacher_modules,
        device=TrainCfg.device,
    )

    model.load_teacher_modules(TrainCfg.teacher_checkpoint)

    logger.print("model has been loaded")
    logger.print(f"Number of parameters: {sum([p.data.nelement() for p in model.parameters()]):d}")

    criterion = nn.MSELoss().to(TrainCfg.device)
    if TrainCfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            TrainCfg.lr,
            weight_decay=TrainCfg.weight_decay,
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
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainCfg.n_epochs)
    logger.start("epoch")

    model.to(TrainCfg.device)

    train_fn = train_recurrent if TrainCfg.recurrent else train
    eval_fn = evaluate_recurrent if TrainCfg.recurrent else evaluate

    for epoch in range(0, TrainCfg.n_epochs + 1):
        if TrainCfg.lr_schedule:
            schedule.step(epoch)

        if TrainCfg.checkpoint_interval and epoch % TrainCfg.checkpoint_interval == 0:
            logger.print("Saving checkpoints...")
            logger.save_torch(model, f"checkpoints/net_{epoch}.pt")
            logger.duplicate(f"checkpoints/net_{epoch}.pt", "checkpoints/net_last.pt")

        train_fn(train_loader, model, criterion, optimizer)
        if val_loader:
            eval_fn(val_loader, model, criterion)
        logger.log_metrics_summary(key_values={"epoch": epoch, "dt_epoch": logger.split("epoch")})


if __name__ == "__main__":
    main()
