# Decision Transformer

# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Dependency Versions.

On linux I use

```
gym==0.23.1
mujoco-py==2.0.2.8
mujoco200
```

Note: this does not work with gynasium. Also, this mujoco-py version is too low for mac.

On mac I use
```
mujoco-py==2.1.2.14
```

Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.

## Downloading datasets

Datasets are stored in the `data` directory. Run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt
```

Adding `-w True` will log results to Weights and Biases.
