from agility_analysis.matia.cifar10.main import main

if __name__ == "__main__":
    import jaynes

    from ml_logger.job import instr

    envs = "LANG=utf-8 LC_CTYPE=en_US.UTF-8"

    for seed, gpu in zip([100, 200, 300, 400, 500], [6, 5, 4, 3, 2]):
    # for seed, gpu in zip([600, 700], [6, 5]):
        thunk = instr(
            main,
            seed=seed,
            device=f"cuda:{gpu}",
            data_aug=True,
            optimizer="sgd",
            # num_workers=0,
            # dataset_root=f"/tmp/datasets/cifar-{gpu}",
            __diff=False,
        )

        jaynes.config(
            runner=dict(
                # shell="/bin/bash --norc",
                # envs=envs + f" CUDA_VISIBLE_DEVICES={gpu}",
            )
        )

        jaynes.run(thunk, )

    jaynes.listen()
