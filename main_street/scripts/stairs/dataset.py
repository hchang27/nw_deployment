from dataclasses import dataclass
from ml_logger import logger


@dataclass
class Dataset:
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
    dataset = Dataset(logger_prefix="/alanyu/scratch/2024/01-13/195458/rollout_0000")
    rollout = dataset[0]
    print(rollout.keys(), rollout["obs"][0].shape)
