from dataclasses import dataclass
from ml_logger import logger


@dataclass
class BallDataset:
    """
    Contain the observations, actions, states for collected rollouts
    """

    logger_prefix: str

    def __getitem__(self, item):
        rollout = None
        with logger.Prefix(self.logger_prefix):
            try:
                (rollout,) = logger.load_pkl(f"rollout_{item:04}.pkl")
            except:
                print(f"Failed to load rollout {item}.")

        return rollout

    def __len__(self):
        return len(logger.glob("rollout_*.pkl"))


if __name__ == "__main__":
    dataset = BallDataset(logger_prefix="/lucid-sim/lucid-sim/datasets/ball/debug/00001/rollouts/")
    rollout = dataset[0]
    print("Num samples:", len(rollout))
    print("Keys:", rollout[0].keys())
    print("Observation shape:", rollout[0]["obs"][0].shape)
