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
    with Sweep(Params, RUN) as sweep:
        """Setting up the sweep"""

        with sweep.zip:
            RUN.job_counter = list(range(1_000))

            with sweep.product:
                Params.seed = [100]
                Params.batch_size = [32, 64]
                Params.n_epochs = [1_000]
                Params.optimizer = ["adam"]
                Params.lr = [0.00005, 0.0001, 0.0005]
                Params.lr_schedule = [True, False]
                Params.image_path = ["lucid_dreams"]
                Params.data_aug = [["crop", "rotate", "color"],
                                   ["crop", "color", "perspective"]]
                Params.dataset = ["xyyaw"]
                Params.symmetry_coef = [0.0, 0.25, 0.5]

        print("done")

    print("out")


    @sweep.each
    def tail(Params, RUN):
        RUN.prefix = f"/lucid-sim/matia/analysis/ball_{Params.dataset}/{Params.image_path}/very_large/{{job_counter}}/"


    print("tail")

    sweep.save(f"{Path(__file__).stem}.jsonl")

    print("saveds")
