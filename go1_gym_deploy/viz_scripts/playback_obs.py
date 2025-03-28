import lcm

from go1_gym_deploy import task_registry
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING
from go1_gym_deploy.utils.deployment_runner import DeploymentRunner

import time

import numpy as np
import pandas as pd

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

if __name__ == '__main__':
    from ml_logger import logger

    # checkpoint = "/lucid-sim/lucid-sim/baselines/launch_flat/2023-12-26/16.16.14/go1_flat/300"
    # prefix = "/alanyu/scratch/2023/12-27/192746"

    checkpoint = "/lucid-sim/lucid-sim/2023-12-27/14.27.54/scripts/train/14.27.54/1"  # actuator net
    prefix = "/alanyu/scratch/2023/12-27/194724"

    with logger.Prefix(prefix):
        data, = logger.load_pkl("log_dict.pkl")

    obs = torch.cat(data['obs'], dim=0)

    # fmt: off
    train_cfg, env_cfg, state_estimator, lcm_agent, actor = task_registry.make_agents( \
        "go1_flat", lc, "cpu", checkpoint)
    # fmt: on

    state_estimator.spin()

    deployment_runner = DeploymentRunner(actor, lcm_agent)

    max_steps = 10000000

    time.sleep(1)

    deployment_runner.run(max_steps=max_steps, obs_replay_log=obs[11:])
