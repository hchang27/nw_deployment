import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms.v2 as T
from math import ceil
from params_proto import ParamsProto, Proto, Flag
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from agility_analysis.matia.cifar10.models import get_model, MODEL_TYPES


class Params(ParamsProto):
    dataset_root: str = Proto("/tmp/datasets", env="DATASETS")
    seed = 100
    num_classes = 10
    data_aug = Flag()

    batch_size = 128
    n_epochs = 200

    arch: MODEL_TYPES = "SimpleDLA"
    """Model types, limited to the following: SimpleDLA, SENet, PreActResNet, ResNet, WideResNet..."""

    optimizer = "sgd"
    lr = 0.1
    lr_schedule = None
    momentum = 0.9
    weight_decay = 5e-4

    eval_full_trainset = Flag(
        "Whether to re-evaluate the full train set on a fixed model, or simply report "
        "the running average of training statistics"
    )

    checkpoint_stops = None
    checkpoint_interval = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = None


class Ge10(CIFAR10):
    def __init__(
        self, batch_size, device=None, shuffle=False, transform=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.data = torch.from_numpy(self.data)
        self.data = torch.permute(self.data, [0, 3, 1, 2]).contiguous()
        self.target = torch.LongTensor(self.targets).contiguous()
        self.device = device

        # Testing this out
        self.data = self.data.to(device)
        self.target = self.target.to(device)
        self.indices = torch.arange(len(self.data))

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        for batch in torch.chunk(self.indices, ceil(len(self.data) / 128)):
            images = self.data[batch]
            if self.transform is not None:
                images = self.transform(images)
            yield images, self.target[batch]


def get_cifar(shuffle_train=True):
    normalize = T.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    pipeline = [
        T.ToImage(),
        T.ToDtype(torch.float32),
        normalize,
    ]
    transform_test = T.Compose(pipeline)

    if Params.data_aug:
        pipeline = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            *pipeline,
        ]

    transform_train = T.Compose(pipeline)

    train_loader = Ge10(
        batch_size=Params.batch_size,
        shuffle=shuffle_train,
        transform=transform_train,
        device=Params.device,
        root=Params.dataset_root,
        train=True,
        download=True,
    )
    val_loader = Ge10(
        device=Params.device,
        root=Params.dataset_root,
        train=False,
        transform=transform_test,
        batch_size=Params.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer):
    """Train for one epoch on the training set"""
    from ml_logger import logger

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.store_metrics({"train/loss": loss.item()})

        # measure accuracy and record loss
        prec1 = top_k(output.data, target, topk=(1,))[0]
        logger.store_metrics({"train/top_1": prec1.item()})


def evaluate(val_loader: DataLoader, model, criterion):
    """Perform validation on the validation set"""
    from ml_logger import logger

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        for input, target in val_loader:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            logger.store_metrics({"eval/loss": loss.item()})

            # measure accuracy and record loss
            prec1 = top_k(output.data, target, topk=(1,))[0]
            logger.store_metrics({"eval/top_1": prec1.item()})


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ExpDecay:
    def __init__(self, optimizer, lr, n_epochs=None, steps=None, fracs=None):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
        self.optimizer = optimizer
        self.lr = lr

        if n_epochs:
            steps = np.array(steps, dtype=float)
            steps *= (n_epochs + 1.0) / steps.sum()
            steps = [round(s) for s in steps]
            assert sum(steps) == n_epochs + 1, "the steps needs to be equal to n"

        lrs = lr * np.array(fracs)
        self.lrs = np.concatenate([[lr] * int(step) for lr, step in zip(lrs, steps)])

    def get_lr(self, epoch):
        return self.lrs[epoch]

    def step(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(epoch)


def top_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


def main(**deps):
    from ml_logger import logger

    print(logger.get_dash_url())

    Params._update(deps)
    logger.job_started(Params=vars(Params))

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
        - yKeys: ["train/top_1/mean", "eval/top_1/mean"]
          xKey: epoch
        - yKey: "train/loss/mean"
          xKey: epoch
        """,
        dedent=True,
        filename=".charts.yml",
        overwrite=True,
    )

    logger.print("Loading data...")
    train_loader, val_loader = get_cifar(shuffle_train=True)
    logger.print("Loading model...")

    model = get_model(Params.arch, device=Params.device)
    logger.print("model has been loaded")
    logger.log(
        f"Number of parameters: {sum([p.data.nelement() for p in model.parameters()]):d}"
    )

    criterion = nn.CrossEntropyLoss().to(Params.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        Params.lr,
        momentum=Params.momentum,
        weight_decay=Params.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Params.n_epochs
    )

    logger.start("epoch")
    for epoch in range(0, Params.n_epochs + 1):
        if Params.lr_schedule is not None:
            schedule.step(epoch)
            logger.log(lr=schedule.get_lr(epoch))

        if (
            Params.checkpoint_interval
            and epoch % Params.checkpoint_interval == 0
            or Params.checkpoint_stops
            and epoch in Params.checkpoint_stops
        ):
            print("Saving checkpoints...")
            logger.save_torch(model, f"checkpoints/net_{epoch}.pt")
            logger.duplicate(f"checkpoints/net_{epoch}.pt", f"checkpoints/net_last.pt")

        train(train_loader, model, criterion, optimizer)
        evaluate(val_loader, model, criterion)
        logger.log_metrics_summary(
            key_values={"epoch": epoch, "dt_epoch": logger.split("epoch")}
        )


if __name__ == "__main__":
    main()
