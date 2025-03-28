import lcm

from go1_gym_deploy import task_registry
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING
from go1_gym_deploy.utils.deployment_runner import DeploymentRunner

import time

import numpy as np
import pandas as pd

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

JOINT_MODE = False

if __name__ == '__main__':
    from ml_logger import logger

    checkpoint = "/lucid-sim/lucid-sim/baselines/launch_flat/2023-12-26/16.16.14/go1_flat/300"

    # prefix = "/alanyu/scratch/2023/12-26/224237"
    # prefix = "/alanyu/scratch/2023/12-27/151009"

    with logger.Prefix(prefix):
        data = logger.read_metrics()
        df = pd.DataFrame(data['metrics.pkl'])
        df.columns = df.columns.str.replace('/mean', '')

    if JOINT_MODE:
        df = df[[col for col in df.columns if "dof" in col]]
    else:
        df = df[[col for col in df.columns if "dof" not in col]]

    data = df.to_numpy()
    data = torch.from_numpy(data).float()

    # fmt: off
    train_cfg, env_cfg, state_estimator, lcm_agent, actor = task_registry.make_agents( \
        "go1_flat", lc, "cpu", checkpoint)
    # fmt: on
    state_estimator.spin()

    deployment_runner = DeploymentRunner(actor, lcm_agent)

    max_steps = 10000000

    time.sleep(1)

    # default_joint_angles_isaac = lcm_agent.default_dof_pos
    # default_joint_angles_unitree = default_joint_angles_isaac[:, JOINT_IDX_MAPPING]

    # centered = default_joint_angles_unitree - lcm_agent.default_dof_pos[:, JOINT_IDX_MAPPING]

    # dof_log_unitree = [centered] * 1000

    log_isaac = data[11:, ]
    log_unitree = log_isaac[:, JOINT_IDX_MAPPING]
    centered = (log_unitree - lcm_agent.default_dof_pos[:, JOINT_IDX_MAPPING]) / lcm_agent.cfg.control.action_scale

    deployment_runner.run(max_steps=max_steps, replay_log=centered)
