from pathlib import Path

if __name__ == "__main__":
    from pandas import DataFrame
    from params_proto import PrefixProto
    from params_proto.hyper import Sweep
    from agility_analysis.matia.cifar10.main import Params
    from ml_logger import logger
    from ml_logger.job import RUN

    machine_list = logger.load_pkl("/lucid-sim/infra/vision_cluster/metrics.pkl")
    df = DataFrame(machine_list)
    available = df[df["status"] == True].filter(items=["host", "gpu_id"])

    print(f"There are {len(available)} GPUs available.")

    class Host(PrefixProto):
        """For specifying the Machine address and GPU id."""

        ip: int = "vision01"
        gpu_id: int = 0

    with Sweep(Params, Host, RUN) as sweep:
        """Setting up the sweep"""

        Params.data_aug = True
        with sweep.zip:
            Host.ip = available["host"].tolist()
            Host.gpu_id = available["gpu_id"].tolist()

            with sweep.product:
                Params.arch = [
                    # "SimpleDLA",
                    # "WideResNet",
                    # "ResNet18",
                    # "PreActResNet18",
                    # "GoogLeNet",
                    # "DenseNet121",
                    # "ResNeXt29",
                    # "MobileNet",
                    # "MobileNetV2",
                    # "DPN92",
                    # "SENet18",
                    # "EfficientNetB0",
                    # "RegNetX_200MF",
                    "VGG",
                    "MLP",
                    "ShuffleNetV2",
                    "ShuffleNetG2",
                ]
                Params.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def tail(Params, Host, RUN):
        RUN.prefix = f"/lucid-sim/matia/analysis/cifar10/{Params.arch}/{Params.seed}"
        Params.device = f"cuda:{Host.gpu_id}"

    sweep.save(f"{Path(__file__).stem}.jsonl")
