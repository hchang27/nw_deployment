import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=1),
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=2),
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=3),
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=4),
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=5),
    dict(ip="isola-v100-2.csail.mit.edu", gpu_id=6),
]

if __name__ == "__main__":
    from agility_analysis import RUN, instr
    from main_street.scripts.train import train
    from main_street.envs.go1.go1_config import Go1RoughCfg

    with Sweep(RUN, RunArgs, Go1RoughCfg.domain_rand) as sweep:
        # RunArgs.debug = True
        with sweep.zip:
            RunArgs.seed = [100, 200, 300]
        with sweep.product:
            Go1RoughCfg.domain_rand.action_curr_step_scratch = [[0, 6], [0, 4]]
            RunArgs.task = ["go1"]
            RunArgs.exptid = ["000-00-debug"]

        RunArgs.max_iterations = 25_000

    @sweep.each
    def tail(RUN, RunArgs, dr):
        RUN.job_name = f"{RunArgs.task}/delay_{dr.action_curr_step_scratch[-1]}/{RunArgs.seed}"

    for i, (machine, deps) in enumerate(zip(machines, sweep)):
        host = machine["ip"]
        visible_devices = f'{machine["gpu_id"]}'
        # jaynes.config(mode='local')
        jaynes.config(
            launch=dict(ip=host),
            runner=dict(envs=f"CUDA_VISIBLE_DEVICES={visible_devices}"),
            verbose=True,
        )
        print(f"Setting up config {i} on machine {host}")
        thunk = instr(train, RunArgs, **deps)
        jaynes.add(thunk)

    jaynes.execute()
    jaynes.listen(300)
