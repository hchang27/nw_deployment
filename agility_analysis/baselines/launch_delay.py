import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    dict(ip="vision09.csail.mit.edu", gpu_id=0),
    dict(ip="vision09.csail.mit.edu", gpu_id=2),
    dict(ip="vision09.csail.mit.edu", gpu_id=4),
    dict(ip="vision09.csail.mit.edu", gpu_id=5),
    dict(ip="vision09.csail.mit.edu", gpu_id=7),
    dict(ip="vision03.csail.mit.edu", gpu_id=0),
    dict(ip="vision03.csail.mit.edu", gpu_id=1),
    dict(ip="vision03.csail.mit.edu", gpu_id=2),
]

if __name__ == "__main__":
    from agility_analysis import RUN, instr
    from main_street.envs.go1.go1_config import Go1RoughCfg
    from main_street.train import train

    with Sweep(RUN, RunArgs, Go1RoughCfg.domain_rand) as sweep:
        # RunArgs.debug = True

        with sweep.zip:
            Go1RoughCfg.domain_rand.action_curr_step = [[2, 2], [4, 4], [6, 6], [7, 7]]
            Go1RoughCfg.domain_rand.action_delay_view = [2, 4, 6, 7]
        with sweep.product:
            RunArgs.seed = [200, 300]
            RunArgs.task = ["go1"]
            RunArgs.exptid = ["000-00-debug"]
            RunArgs.max_iterations = [25_000]


    @sweep.each
    def tail(RUN, RunArgs, dr):
        RUN.job_name = f"{RunArgs.task}/{RunArgs.seed}/{dr.action_delay_view}"


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
        jaynes.run(thunk)

    # jaynes.execute()
    jaynes.listen(300)
