import jaynes

# Tommi's machines
# haukka.csail.mit.edu
# peuras.csail.mit.edu
# karhu.csail.mit.edu
# susi.csail.mit.edu
# molauk.csail.mit.edu
# moltern.csail.mit.edu

machines = """
vision01.csail.mit.edu vision02.csail.mit.edu vision03.csail.mit.edu
vision04.csail.mit.edu vision05.csail.mit.edu vision06.csail.mit.edu
vision07.csail.mit.edu vision08.csail.mit.edu vision09.csail.mit.edu
vision22.csail.mit.edu vision23.csail.mit.edu vision24.csail.mit.edu
vision25.csail.mit.edu vision26.csail.mit.edu vision27.csail.mit.edu
vision28.csail.mit.edu vision33.csail.mit.edu
vision34.csail.mit.edu vision41.csail.mit.edu
vision42.csail.mit.edu vision43.csail.mit.edu 
vision47.csail.mit.edu vision48.csail.mit.edu
"""

machines = " ".join(machines.strip().split("\n")).split(" ")


class Args:
    max_load = 0.1
    max_memory = 0.02


import shutil

DESIRED_MEMORY = 10_000


def check_docker():
    return shutil.which("docker") is not None


def show_hostname(prefix, host):
    import GPUtil
    from ml_logger import logger

    logger.configure(prefix=prefix)

    try:
        gpus = GPUtil.getGPUs()
        gpu_availability = GPUtil.getAvailability(
            gpus,
            maxLoad=Args.max_load,
            maxMemory=Args.max_memory,
            includeNan=False,
            excludeID=[],
            excludeUUID=[],
        )
        for gpu_id, availability in enumerate(gpu_availability):
            availability &= gpus[gpu_id].memoryTotal > DESIRED_MEMORY
            print(f"host: {host}")
            logger.log(
                # host can be different from hostname.
                host=host,
                hostname=logger.hostname,
                gpu_id=gpu_id,
                status=availability,
                flush=True,
            )

        logger.log_text(
            f"- {host}, gpu_available: {gpu_availability}\n", filename="README.md"
        )

        logger.log_metrics_summary()

    except Exception as e:
        print(logger.hostname, "no gpus available", e)


def launch():
    from ml_logger import logger
    from time import sleep

    logger.configure(prefix="lucid-sim/infra/vision_cluster")
    logger.job_started()
    with logger.Sync():
        logger.remove("metrics.pkl")
        logger.remove("README.md")
        sleep(1)

    for host in machines:
        jaynes.config(launch=dict(ip=host))
        jaynes.run(show_hostname, prefix=logger.prefix, host=host)
        # jaynes.launcher = None


if __name__ == "__main__":
    launch()
    jaynes.listen()
