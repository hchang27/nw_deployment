if __name__ == "__main__":
    import jaynes
    from ml_logger import logger
    from ml_logger.job import instr, RUN
    from params_proto import PrefixProto
    from params_proto.hyper import Sweep

    from agility_analysis.matia.cifar10.main import main, Params

    class Host(PrefixProto):
        """For specifying the Machine address and GPU id."""

        ip: int = "vision01"
        gpu_id: int = 0

    sweep = Sweep(Params, Host, RUN).load("sweep.jsonl")

    for deps in sweep:
        jaynes.config(
            # runner={"shell": "/bin/bash --norc"},
            launch={"ip": Host.ip},
        )
        with logger.Prefix(RUN.prefix):
            logger.remove("metrics.pkl", "output.log", "traceback.err")
        thunk = instr(main, **deps, __diff=False)

        jaynes.run(thunk)

    jaynes.listen()
