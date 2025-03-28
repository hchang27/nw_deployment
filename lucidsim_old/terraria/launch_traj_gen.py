import jaynes
from params_proto.hyper import Sweep


def main(**deps):
    from lucidsim_old.traj_generation.traj_gen import TrajGenerator
    traj_gen = TrajGenerator(**deps)
    traj_gen()


if __name__ == '__main__':

    from ml_logger import logger
    from ml_logger.job import instr
    from pandas import DataFrame

    from lucidsim_old.traj_generation.traj_gen import TrajGenerator

    MACHINE_LIMIT = 16

    machine_list = logger.load_pkl("/lucid-sim/infra/vision_cluster/metrics.pkl")
    df = DataFrame(machine_list)
    available = df[df["status"] == True].filter(items=["host", "gpu_id"])
    print(f"There are {len(available)} GPUs available, but we will limit to using {MACHINE_LIMIT} machines.")
    available = available[:MACHINE_LIMIT]

    ips = available["host"].tolist()
    devices = available["gpu_id"].tolist()

    with Sweep(TrajGenerator) as sweep:
        with sweep.product:
            TrajGenerator.rollout_range = [(0, 20)]
            TrajGenerator.terrain_type = ["flat", "hurdle", "gap", "stairs"]

            with sweep.zip:
                TrajGenerator.dataset_prefix = ["scene_00005", "scene_00006", "scene_00007"]
                TrajGenerator.seed = [5, 6, 7]

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
