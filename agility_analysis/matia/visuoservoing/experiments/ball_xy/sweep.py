"""
Read prompts from JSON, and simply zip with the rollout number.
"""

from pathlib import Path

from pandas import DataFrame
from params_proto import PrefixProto, Proto, ParamsProto
from params_proto.hyper import Sweep
from ml_logger import logger
from ml_logger.job import RUN

from agility_analysis.matia.visuoservoing.main import Params


class Host(PrefixProto):
    """For specifying the Machine address and GPU id."""

    ip: int = "vision01"
    gpu_id: int = 0


if __name__ == '__main__':
    machine_list = logger.load_pkl("/lucid-sim/infra/vision_cluster/metrics.pkl")
    df = DataFrame(machine_list)
    available = df[df["status"] == True].filter(items=["host", "gpu_id"])

    print(f"There are {len(available)} GPUs available.")

    with Sweep(Params, RUN) as sweep:
        """Setting up the sweep"""

        with sweep.zip:
            RUN.job_counter = list(range(1_000))

            with sweep.product:
                Params.seed = [100, 200]
                Params.batch_size = [24, 32, 64, 128]
                Params.n_epochs = [1000]
                Params.optimizer = ["sgd", "adam"]
                Params.lr = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
                Params.lr_schedule = [True, False]
                Params.momentum = [0.7, 0.8, 0.9]

        print("done")

    print("out")


    @sweep.each
    def tail(Params, RUN):
        RUN.prefix = f"/lucid-sim/matia/analysis/ball_xy/{Params.image_path}/small/{{job_counter}}/"


    print("tail")

    sweep.save(f"{Path(__file__).stem}.jsonl")

    print("saveds")
