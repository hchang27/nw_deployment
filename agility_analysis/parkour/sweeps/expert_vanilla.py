from pathlib import Path

from params_proto.hyper import Sweep

from cxx.runners.on_policy_runner import RunnerArgs
from main_street.config import RunArgs

if __name__ == "__main__":
    from agility_analysis import RUN

    with Sweep(RUN, RunArgs, RunnerArgs) as sweep:
        # RunArgs.debug = True
        RunArgs.task = "go1"
        RunArgs.max_iterations = 25_000
        RunnerArgs.save_intermediate_checkpoints = True

        with sweep.product:
            RunArgs.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def tail(RUN, RunArgs, RunnerArgs):
        RUN.prefix, RUN.job_name, _ = RUN(
            script_path=__file__,
            job_name=f"{RunArgs.task}/seed-{RunArgs.seed}",
        )


sweep.save(f"{Path(__file__).stem}.jsonl")
