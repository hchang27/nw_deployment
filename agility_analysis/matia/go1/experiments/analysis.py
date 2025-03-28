prefix = "/lucid-sim/matia/analysis/go1/hurdle/augmented/very_large/"

from ml_logger import logger

logger.configure(prefix=prefix)

all_experiments = logger.glob("*/metrics.pkl")
print(f"Found {len(all_experiments)} experiments.")

experiment = all_experiments[0]

params = {}
results = {}

died = []

for experiment in all_experiments:
    number = experiment.split("/")[-2]

    try:
        with logger.Prefix(number):
            train_loss = logger.read_metrics('eval/action_loss/mean')
            parameters = logger.read_params("TrainCfg")
    except ValueError:
        died.append(number)

    params[number] = parameters
    results[number] = train_loss.min()

print(len(died))

sorted_results = sorted(results, key=lambda x: results[x])

best = sorted_results[0]

sorted([key for key in results if params[key]["stack_size"]==3])

results['81']
params['48']
