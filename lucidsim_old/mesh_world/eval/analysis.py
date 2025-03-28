from params_proto import ParamsProto
import pandas as pd


class Args(ParamsProto):
    experiment_log = "/lucid-sim/lucid-sim/analysis/eval/stairs/"
    keys = ["rgb_frame_diff", "depth"]


def main():
    from ml_logger import logger

    with logger.Prefix(Args.experiment_log):
        results = {}
        for key in Args.keys:
            results[key] = logger.read_metrics(key)

        df = pd.DataFrame(results)


if __name__ == '__main__':
    main()
