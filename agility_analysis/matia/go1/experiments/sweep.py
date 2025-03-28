"""
Read prompts from JSON, and simply zip with the rollout number.
"""

from pathlib import Path

from pandas import DataFrame
from params_proto import PrefixProto, Proto, ParamsProto
from params_proto.hyper import Sweep
from ml_logger import logger
from ml_logger.job import RUN

from agility_analysis.matia.go1.main_tracking import TrainCfg


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
                TrainCfg.num_shared_layers = [8, 10]
                TrainCfg.n_epochs = [500]
                TrainCfg.optimizer = ["adam"]
                TrainCfg.lr = [0.00005, 0.0005]
                TrainCfg.lr_schedule = [True, False]
                TrainCfg.image_type = ["augmented"]
                TrainCfg.use_stacked_dreams = [False]
                TrainCfg.num_filters = [32, 64]
                TrainCfg.checkpoint_interval = [None]
                TrainCfg.symmetry_coef = [0.0, 0.1]

    @sweep.each
    def tail(TrainCfg, RUN):
        RUN.prefix = f"/lucid-sim/matia/analysis/go1/tracking/{TrainCfg.image_type}/debug/{{job_counter}}/"


    print("tail")

    sweep.save(f"{Path(__file__).stem}.jsonl")

    print("saveds")
