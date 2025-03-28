import pickle

import gym
import numpy as np
import torch
from params_proto import ParamsProto, Proto
from tqdm import tqdm

from decision_transformer.act_trainer import ActTrainer
from decision_transformer.buffer import Buffer
from decision_transformer.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


import d4rl

assert d4rl, "no need to import this explicitly, because the pkl files implicitly loads these environments"


class Args(ParamsProto):
    env = 'maze2d-umaze'
    dataset = Proto('dense', help="Dataset size. Options: medium, medium-replay, medium-expert, expert")
    mode = Proto('normal', help="Mode of operation. Options: normal for standard setting, delayed for sparse")
    model_type = Proto('dt', help="Model type. Options: dt for decision transformer, bc for behavior cloning")
    K = 20
    pct_traj = 1.
    batch_size = 64
    embed_dim = 128
    n_layer = 3
    n_head = 1
    activation_function = 'relu'
    dropout = 0.1
    learning_rate = 1e-4
    weight_decay = 1e-4
    warmup_steps = 10000
    num_eval_episodes = 100
    max_iters = 10
    num_steps_per_iter = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    from ml_logger import logger

    logger.job_started(Args=Args.__dict__)
    print(logger.get_dash_url())

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    # torch.set_float32_matmul_precision('medium')  # or high

    if Args.env == 'maze2d-umaze':
        env = gym.make('maze2d-umaze-v1')
        max_ep_len = 240
        env_targets = [25, 50, 65]  # max is 66.8 per epoch
        rtg_scale = 100.  # normalization for rewards/returns
    elif Args.env == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        rtg_scale = 1000.  # normalization for rewards/returns
    elif Args.env == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        rtg_scale = 1000.
    elif Args.env == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        rtg_scale = 1000.
    elif Args.env == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        rtg_scale = 10.
    else:
        raise NotImplementedError

    if Args.model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'../datasets/{Args.env}-{Args.dataset}-v1.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if Args.mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    states_mean, states_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {Args.env} {Args.dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    num_eval_episodes = Args.num_eval_episodes
    pct_traj = Args.pct_traj or 1.

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    # sorted_inds = sorted_inds[-num_trajectories:]

    # preprocess: we precompute the returns to go so that
    # we don't need to compute this on the fly.
    for traj in tqdm(trajectories, desc="pre-compute rtg"):
        # Needed to pad last bit to 0 (because the last step should be termination)
        rtg_head = discount_cumsum(traj['rewards'], gamma=1.)
        traj['return-to-go'] = np.concatenate([rtg_head, np.zeros([1])])
        # move to GPU
        for key in traj:
            traj[key] = torch.Tensor(traj[key]).to(Args.device)

    all_states = torch.cat([t['observations'] for t in trajectories])
    states_mean = all_states.mean(axis=0)
    states_std = all_states.std(axis=0)
    del all_states

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if Args.model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=rtg_scale,
                            target_return=target_rew / rtg_scale,
                            mode=Args.mode,
                            state_mean=states_mean,
                            state_std=states_std,
                            device=Args.device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / rtg_scale,
                            mode=Args.mode,
                            state_mean=states_mean,
                            state_std=states_std,
                            device=Args.device,
                        )
                returns.append(ret)
                lengths.append(length)

            logger.store_metrics(**{
                f'R{target_rew}/return': np.mean(returns),
                f'R{target_rew}/length': np.mean(lengths),
            })

        return fn

    if Args.model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=Args.K,
            max_ep_len=max_ep_len,
            hidden_size=Args.embed_dim,
            n_layer=Args.n_layer,
            n_head=Args.n_head,
            n_inner=4 * Args.embed_dim,
            activation_function=Args.activation_function,
            n_positions=1024,
            resid_pdrop=Args.dropout,
            attn_pdrop=Args.dropout,
        )
    elif Args.model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=Args.K,
            hidden_size=Args.embed_dim,
            n_layer=Args.n_layer,
        )
    else:
        raise NotImplementedError

    model = model.to(device=Args.device)
    model = torch.compile(model)

    warmup_steps = Args.warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Args.learning_rate,
        weight_decay=Args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    # hardcode the obs_dim and act_dim for now
    buffer = Buffer(trajectories, Args.batch_size, Args.K, obs_dim=state_dim, act_dim=act_dim,
                    max_ep_len=max_ep_len, rtg_scale=rtg_scale, states_mean=states_mean,
                    states_std=states_std, device=Args.device)

    if Args.model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=Args.batch_size,
            buffer=buffer,
            scheduler=scheduler,
            states_mean=states_mean,
            states_std=states_std,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif Args.model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=Args.batch_size,
            buffer=buffer,
            scheduler=scheduler,
            states_mean=states_mean,
            states_std=states_std,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    from ml_logger import logger
    logger.job_started(config=Args.__dict__)
    print(logger)
    print(logger.get_dash_url())

    logger.start("start")
    for iter in range(Args.max_iters):
        trainer.train_iteration(num_steps=Args.num_steps_per_iter)
        logger.log_metrics_summary(key_values={"wallclock_time": logger.since("start"), "epoch": iter, "step": trainer.step})
        logger.save_torch(model, "checkpoints/model.pt")


if __name__ == '__main__':
    from ml_logger import logger

    logger.log_text("""
    charts:
    - yKey: loss/mean
      xKey: step
      yFormat: log
    - yKey: action_error/mean
      xKey: step
    - yKey:  R1800/return/mean
      xKey: step
    - yKey:  R3600/return/mean
      xKey: step
    """, ".charts.yml", dedent=True, overwrite=True)
    main()
