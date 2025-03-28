import lcm

from go1_gym_deploy import task_registry
from go1_gym_deploy.utils.deployment_runner import DeploymentRunner

import time

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

if __name__ == '__main__':
    from ml_logger import logger

    device = "cuda:0"

    # Vision 
    checkpoint = "/lucid-sim/lucid-sim/2023-12-28/20.44.12/scripts/train/distillation/2023-12-28/20.44.12/go1/200"

    with logger.Prefix(checkpoint):
        data, = logger.load_pkl("log_dict.pkl")

    actions = data['actions_isaac'][15:]

    observations = data["obs"][15:]
    depths = data["depth"][15:]

    env_cfg, train_cfg, state_estimator, lcm_agent, actor, zed_node = task_registry.make_agents(
        "go1",
        lc,
        device,
        checkpoint,
    )

    state_estimator.spin()

    # zed_node.spin("rgb", "depth")

    # print("starting camera")
    # while len(zed_node.frame) < 2:
    #     time.sleep(0.01)
    # 
    # print('camera is ready')

    deployment_runner = DeploymentRunner(actor, lcm_agent)

    max_steps = 10000000

    time.sleep(1)

    # deployment_runner.run(max_steps=max_steps, actions_replay_log=actions)
    deployment_runner.run(max_steps=max_steps, obs_replay_log=list(zip(observations, depths)))
