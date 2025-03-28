import jaynes
from params_proto.hyper import Sweep

from main_street.config import RunArgs

machines = [
    dict(ip="vision09", gpu_id=0),
]


if __name__ == "__main__":
    from agility_analysis import instr, RUN


    def examine_environment():
        import isaacgym
        assert isaacgym
        print("this is done!")

    thunk = instr(examine_environment)
    jaynes.run(thunk)
    jaynes.listen()
    exit()

    # from main_street.train import train
    from agility_analysis import instr, RUN

    # with Sweep(RUN) as sweep:
    #
    # @sweep.each
    # def tail(RUN, RunArgs):
    #     RUN.job_name = f"{RUN.now:%Y-%m-%d/%H.%M.%S}/{RunArgs.seed}"


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
        thunk = instr(examine_environment, RunArgs, **deps)
        jaynes.add(thunk)

    jaynes.execute()
    jaynes.listen(300)
