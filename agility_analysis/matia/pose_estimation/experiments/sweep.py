"""
Read prompts from JSON, and simply zip with the rollout number.
"""

from pathlib import Path

from pandas import DataFrame
from params_proto import PrefixProto, Proto, ParamsProto
from params_proto.hyper import Sweep
from ml_logger import logger
from ml_logger.job import RUN

from agility_analysis.matia.pose_estimation.main_of import TrainCfg


class Host(PrefixProto):
    """For specifying the Machine address and GPU id."""

    ip: int = "vision01"
    gpu_id: int = 0


if __name__ == '__main__':
    with Sweep(TrainCfg, RUN) as sweep:
        """Setting up the sweep"""

        with sweep.zip:
            RUN.job_counter = list(range(1_000))

            with sweep.product:
                TrainCfg.seed = [100]
                TrainCfg.batch_size = [32, 64, 128]
                TrainCfg.n_epochs = [5_00]
                TrainCfg.optimizer = ["adam"]
                TrainCfg.lr = [0.00001, 0.00005, 0.0001, 0.0005]
                TrainCfg.lr_schedule = [True, False]
                TrainCfg.image_type = ["rgb"]


    @sweep.each
    def tail(TrainCfg, RUN):
        RUN.prefix = f"/lucid-sim/matia/analysis/pose_estimation/{TrainCfg.image_type}/small/{{job_counter}}/"


    print("tail")

    sweep.save(f"{Path(__file__).stem}.jsonl")

    print("saveds")
