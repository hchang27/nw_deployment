from main_street import task_registry


def test_env(args):
    # use local import to make sure ordering is correct
    import torch

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(10 * env.max_episode_length)):
        actions = 0.0 * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, info = env.step(actions)
    print("Done")


if __name__ == "__main__":
    from main_street.utils import get_args

    args = get_args()
    test_env(args)
