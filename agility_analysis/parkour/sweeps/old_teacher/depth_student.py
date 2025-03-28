from pathlib import Path

from params_proto.hyper import Sweep

from cxx.runners.on_policy_runner import RunnerArgs
from main_street.config import RunArgs

if __name__ == "__main__":
    from agility_analysis import RUN
    from main_street.envs.go1.go1_config import Go1RoughCfg

    with Sweep(RUN, RunArgs, RunnerArgs, Go1RoughCfg.depth) as sweep:
        # RunArgs.debug = True

        RunArgs.task = "go1"
        RunArgs.max_iterations = 25_000
        RunArgs.use_camera = True
        # mark headless to use the depth.xxx override
        RunArgs.headless = True

        RunArgs.load_checkpoint = "/lucid-sim/lucid-sim/baselines/launch/2024-04-15/16.33.55/go1/100"
        RunnerArgs.save_intermediate_checkpoints = True

        with sweep.zip:
            with sweep.product:
                Go1RoughCfg.depth.buffer_len = [2, 3]
                RunArgs.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def tail(RUN, RunArgs,  RunnerArgs, depth):
        RUN.prefix, RUN.job_name, _ = RUN(
            script_path=__file__,
            job_name=f"{RunArgs.task}/buff-{depth.buffer_len}/seed-{RunArgs.seed}",
        )

    sweep.save(f"{Path(__file__).stem}.jsonl")
