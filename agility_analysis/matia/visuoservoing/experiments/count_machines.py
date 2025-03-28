from ml_logger import logger
from pandas import DataFrame

if __name__ == '__main__':
    machine_list = logger.load_pkl("/lucid-sim/infra/vision_cluster/metrics.pkl")
    df = DataFrame(machine_list)
    available = df[df["status"] == True].filter(items=["host", "gpu_id"])

    print(f"There are {len(available)} GPUs available.")
