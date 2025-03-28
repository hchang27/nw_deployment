import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    dict(ip="vision27", gpu_id=3),
    dict(ip="vision27", gpu_id=4),
    dict(ip="vision27", gpu_id=5),
]

if __name__ == "__main__":
    from agility_analysis import RUN, instr
    from main_street.train import train

    with Sweep(RUN, RunArgs) as sweep:
        # RunArgs.debug = True
        with sweep.zip:
            RunArgs.seed = [100, 200, 300]
        with sweep.product:
            RunArgs.task = ["go1"]
            RunArgs.exptid = ["000-00-debug"]

        RunArgs.max_iterations = 25_000

    @sweep.each
    def tail(RUN, RunArgs):
        RUN.job_name = f"{RunArgs.task}/{RunArgs.seed}"

    for i, (machine, deps) in enumerate(zip(machines, sweep)):
        host = machine["ip"]
        visible_devices = f'{machine["gpu_id"]}'
        jaynes.config(
            launch=dict(ip=host),
            runner=dict(envs=f"CUDA_VISIBLE_DEVICES={visible_devices}"),
            # shell="screen -dm /bin/bash --norc",
            verbose=True,
        )
        # print(f"Setting up config {i} on machine {host}")
        thunk = instr(train, **deps)
        jaynes.run(thunk)

    # jaynes.execute()
    jaynes.listen(300)
