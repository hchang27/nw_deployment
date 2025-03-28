import time

import torch
from tqdm import tqdm, trange

from decision_transformer.buffer import Buffer


class Trainer:

    def __init__(self, model, *, optimizer, batch_size, buffer: Buffer, loss_fn, states_mean, states_std, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        # self.get_batch = get_batch
        self.states_mean = states_mean
        self.states_std = states_std
        self.buffer = buffer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.step = 0

    def train_iteration(self, num_steps):
        from ml_logger import logger

        logger.start('train')
        self.model.train()
        for step in trange(num_steps, desc='Training', leave=True):
            self.train_step()
            if self.scheduler is not None:
                self.scheduler.step()
            if step % 100 == 0:
                logger.log_metrics_summary(key_values={"step": self.step}, silent=True)
        logger.store_metrics(train_time=logger.since('train'))

        logger.start('eval')
        self.model.eval()
        loop = tqdm(self.eval_fns, desc='Evaluating')
        for eval_fn in loop:
            loop.set_description(f'Evaluating {eval_fn.__name__}')
            eval_fn(self.model)

        logger.store_metrics(eval_time=logger.since('eval'))

    def train_step(self):
        self.step += 1
        states, actions, rewards, dones, attention_mask, returns = self.buffer.get_batch()
        # I don't get this.
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:, 1:], action_target, reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
