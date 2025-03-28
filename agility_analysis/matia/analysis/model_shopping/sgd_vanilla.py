from agility_analysis.matia.cifar10.main import main

if __name__ == "__main__":
    import jaynes

    from ml_logger.job import instr

    envs = "LANG=utf-8 LC_CTYPE=en_US.UTF-8"

    for seed, gpu in zip([100, 200, 300, 400, 500], [6, 5, 4, 3, 2]):
        thunk = instr(main, __diff=False)

        jaynes.config(
            runner=dict(
                # shell="/bin/bash --norc",
                # envs=envs + f" CUDA_VISIBLE_DEVICES={gpu}",
            )
        )

        jaynes.run(thunk)

    jaynes.listen()
