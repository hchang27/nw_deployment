"""
Dependency Versions:

- linux:
    ```
    gym==0.23.1
    mujoco-py==2.0.2.8
    mujoco200
    ```
- On mac I use
    ```
    mujoco-py==2.1.2.14
    ```
    """
import collections
import os
import pickle

import d4rl
import gym
import numpy as np

assert d4rl, "need to register the environments."

datasets = []

for env_name in ['halfcheetah', 'hopper', 'walker2d']:
    for dataset_type in ['medium', 'medium-replay', 'expert']:
        name = f'{env_name}-{dataset_type}-v2'
        env = gym.make(name)
        dataset = env.get_dataset()

        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        paths = []
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000 - 1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

        returns = np.array([np.sum(p['rewards']) for p in paths])
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

        os.makedirs("../datasets", exist_ok=True)
        with open(f'../datasets/{name}.pkl', 'wb') as f:
            pickle.dump(paths, f)
