import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    # dict(ip="vision23.csail.mit.edu", gpu_id=0),
    # dict(ip="vision23.csail.mit.edu", gpu_id=1),
    # dict(ip="vision23.csail.mit.edu", gpu_id=2),
    dict(ip="vision23.csail.mit.edu", gpu_id=0),
    # dict(ip="vision23.csail.mit.edu", gpu_id=4),
    # dict(ip="vision23.csail.mit.edu", gpu_id=7),
    # dict(ip="vision03.csail.mit.edu", gpu_id=1),
    # dict(ip="vision03.csail.mit.edu", gpu_id=2),
]

if __name__ == "__main__":
    from agility_analysis import RUN, instr
    from main_street.envs.go1.go1_config import Go1RoughCfg
    from main_street.train import train

    with Sweep(RUN, RunArgs, Go1RoughCfg.depth) as sweep:
        # RunArgs.debug = True

        RunArgs.use_camera = True
        RunArgs.resume = True
        RunArgs.load_checkpoint = "/lucid-sim/lucid-sim/baselines/launch/2024-04-15/16.33.55/go1/100"

        with sweep.zip:
            RunArgs.device = [f"cuda:{machines[i]['gpu_id']}" for i in range(len(machines))]
            with sweep.product:
                Go1RoughCfg.depth.buffer_len = [3, 4]
                RunArgs.seed = [300]
                RunArgs.task = ["go1"]
                RunArgs.exptid = ["000-00-debug"]
                RunArgs.max_iterations = [25_000]

    @sweep.each
    def tail(RUN, RunArgs, depth):
        RUN.job_name = f"{RunArgs.task}/{RunArgs.seed}/{depth.buffer_len}"

    for i, (machine, deps) in enumerate(zip(machines, sweep)):
        host = machine["ip"]
        jaynes.config(mode="local")
        # jaynes.config(
        #     launch=dict(ip=host),
        #     verbose=True,
        # )
        # print(f"Setting up config {i} on machine {host}")
        thunk = instr(train, RunArgs, **deps)
        jaynes.run(thunk)

    # jaynes.execute()
    jaynes.listen(300)
