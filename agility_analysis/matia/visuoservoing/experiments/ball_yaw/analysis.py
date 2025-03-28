prefix = "/lucid-sim/matia/analysis/ball_yaw/lucid_dreams/small/"

from ml_logger import logger

logger.configure(prefix=prefix)

all_experiments = logger.glob("*/metrics.pkl")
print(f"Found {len(all_experiments)} experiments.")

experiment = all_experiments[0]

results = {}
params = {}

died = []

for experiment in all_experiments:
    number = experiment.split("/")[-2]

    try:
        with logger.Prefix(number):
            train_loss, eval_loss = logger.read_metrics('train/loss/mean', 'eval/loss/mean')
            parameters = logger.read_params("Params")
    except ValueError:
        died.append(number)

    params[number] = parameters
    results[number] = (eval_loss.min(), train_loss.min())

print(len(died))

sorted_results = sorted(results, key=lambda x: results[x])

best = sorted_results[0]

results[sorted_results[0]]
params[best]
    
