import jaynes
from isaacgym import gymapi
from params_proto.hyper import Sweep

assert gymapi, "force import first to avoid torch error."

from cxx.runners.on_policy_runner import RunnerArgs
from main_street.config import RunArgs

machines = [
    dict(ip="vision23.csail.mit.edu", gpu_id=1),
    dict(ip="vision23.csail.mit.edu", gpu_id=2),
    dict(ip="vision23.csail.mit.edu", gpu_id=3),
    dict(ip="vision23.csail.mit.edu", gpu_id=4),
    dict(ip="vision23.csail.mit.edu", gpu_id=5),
    dict(ip="vision23.csail.mit.edu", gpu_id=6),
    dict(ip="vision23.csail.mit.edu", gpu_id=7),
]

if __name__ == "__main__":
    from agility_analysis import RUN, instr
    from main_street.train import train

    sweep = Sweep(RUN, RunArgs, RunnerArgs).load("sweeps/expert_vanilla.jsonl")

    for i, deps in sweep.items():
        machine = machines[i % len(machines)]

        host = machine["ip"]
        gpu_id = f'cuda:{machine["gpu_id"]}'

        jaynes.config(
            launch=dict(ip=host),
            runner=dict(shell="screen -dm /bin/bash --norc"),
        )
        print(f"Setting up config {i} on machine {host}")
        thunk = instr(train, **{"run.device": gpu_id}, **deps)
        jaynes.add(thunk)

    jaynes.execute()
    jaynes.listen(300)
