import jaynes
from params_proto.hyper import Sweep

if __name__ == '__main__':

    from itertools import product
    from more_itertools import chunked
    import math
    from ml_logger import logger
    from ml_logger.job import instr, RUN
    from pandas import DataFrame
    from agility_analysis.matia.pose_estimation.main_of import TrainCfg, main

    MACHINE_LIMIT = 16

    machine_list = logger.load_pkl("/lucid-sim/infra/vision_cluster/metrics.pkl")
    df = DataFrame(machine_list)
    available = df[df["status"] == True].filter (items=["host", "gpu_id"])
    print(f"There are {len(available)} GPUs available, but we will limit to using {MACHINE_LIMIT} machines.")
    available = available[:MACHINE_LIMIT]

    ips = available["host"].tolist()
    devices = available["gpu_id"].tolist()

    sweep = Sweep(TrainCfg, RUN).load("sweep.jsonl")

    for i, deps in zip(range(len(available)), sweep):
        # jaynes.config(mode="local")
        jaynes.config(
            runner=dict(
                envs=f"CUDA_VISIBLE_DEVICES={int(devices[i])}"),
            launch={"ip": ips[i]},
        )
        thunk = instr(main, **deps, __diff=False)
        jaynes.run(thunk)

    jaynes.listen()
