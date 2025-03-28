from dataclasses import dataclass
from ml_logger import ML_Logger


@dataclass
class Dataset:
    """
    Contain the observations, actions, states for collected rollouts
    
    params:
        logger: ML_Logger instance
        dataset_prefix: the relative path to the dataset folder from logger root
    """

    logger: ML_Logger
    dataset_prefix: str

    def __getitem__(self, item):
        rollout = None
        with self.logger.Prefix(self.dataset_prefix):
            try:
                (rollout,) = self.logger.load_pkl(f"trajectory_{item:04}.pkl")
            except:
                # print(f"Failed to load rollout {item}.")
                raise RuntimeError(f"Failed to load rollout {item}, check server status and network connection")

        return rollout

    def __len__(self):
        with self.logger.Prefix(self.dataset_prefix):
            return len(self.logger.glob("trajectory_*.pkl"))