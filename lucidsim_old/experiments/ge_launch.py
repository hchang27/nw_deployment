import jaynes

machines = [
    dict(ip="vision19", gpu_id=0),
    # dict(ip="vision19", gpu_id=2),
    # dict(ip="vision19", gpu_id=3),
    # dict(ip="vision19", gpu_id=4),
    # dict(ip="vision19", gpu_id=5),
    dict(ip="vision25", gpu_id=0),
    dict(ip="vision26", gpu_id=0),
    dict(ip="vision28", gpu_id=0),
    #
    # dict(ip="vision30", gpu_id=0),
]

if __name__ == "__main__":
    from params_proto import ParamsProto
    from params_proto.hyper import Sweep

    from agility_analysis import instr
    from lucidsim_old.traj_generation.traj_gen import TrajGenerator

    class Args(ParamsProto):
        sweep = "data_generation/parkour_noisy.jsonl"


    def examine_environment(deps):
        TrajGenerator._update(deps)
        traj_gen = TrajGenerator()
        traj_gen()
        print(traj_gen.terrain_type, "domain is finished!")


    sweep = Sweep(TrajGenerator).load(Args.sweep)

    for deps, m in zip(sweep, machines):
        host = m["ip"]
        visible_devices = f'{m["gpu_id"]}'
        jaynes.config(
            launch=dict(ip=host),
            runner=dict(
                envs=f"CUDA_VISIBLE_DEVICES={visible_devices}"
            ),
        )
        print(f"Setting up machine {host}")
        thunk = instr(examine_environment, deps, __diff=False)
        jaynes.run(thunk)

    jaynes.listen()
    exit()
