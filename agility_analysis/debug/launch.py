import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    # dict(ip="visiongpu55", gpu_id=0),
    # dict(ip="visiongpu55", gpu_id=1),
    # dict(ip="visiongpu55", gpu_id=2),
    # dict(ip="visiongpu55", gpu_id=3),
    # dict(ip="vision42", gpu_id=2),
    # dict(ip="vision42", gpu_id=3),
    dict(ip="vision44", gpu_id=0),
    # dict(ip="vision44", gpu_id=1),
    # dict(ip="vision44", gpu_id=2),
    # dict(ip="vision42", gpu_id=7),
    # dict(ip="vision44", gpu_id=3),
    # dict(ip="vision44", gpu_id=4),
    # dict(ip="vision44", gpu_id=5),
    # dict(ip="vision44", gpu_id=6),
    # dict(ip="vision44", gpu_id=7),
    # ===================================
    # dict(ip="visiongpu54", gpu_id=0),
    # dict(ip="visiongpu54", gpu_id=1),
    # dict(ip="visiongpu54", gpu_id=2),
    # dict(ip="visiongpu54", gpu_id=3),
    # dict(ip="visiongpu54", gpu_id=4),
    # dict(ip="visiongpu54", gpu_id=5),
    # dict(ip="visiongpu54", gpu_id=6),
    # dict(ip="visiongpu54", gpu_id=7),
]

if __name__ == "__main__":
    from main_street.train import train
    from agility_analysis import instr, RUN

    with Sweep(RUN, RunArgs) as sweep:
        with sweep.zip:
            RunArgs.seed = [100, 200, 300]
            
        RunArgs.exptid = "000-30-debug" # , "000-32-debug"]
        RunArgs.task = "go1_ball"

    @sweep.each
    def tail(RUN, RunArgs):
        RUN.job_name = f"{RUN.now:%Y-%m-%d/%H.%M.%S}/{RunArgs.seed}"


    for i, (machine, deps) in enumerate(zip(machines, sweep)):
        host = machine["ip"]
        visible_devices = f'{machine["gpu_id"]}'
        # jaynes.config(mode='local')
        jaynes.config(
            launch=dict(ip=host),
            runner=dict(
                envs=f"CUDA_VISIBLE_DEVICES={visible_devices}"
            ),
            verbose=True,
        ) 
        print(f"Setting up config {i} on machine {host}")
        thunk = instr(train, RunArgs, **deps)
        jaynes.add(thunk)

    jaynes.execute()
    jaynes.listen(300)
