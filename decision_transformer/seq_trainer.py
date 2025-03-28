import torch

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer


class SequenceTrainer(Trainer):
    model: DecisionTransformer

    def train_step(self):
        from ml_logger import logger

        self.step += 1

        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.buffer.get_batch()
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        # print the shape and dtype of the returned objects from the self.loss_fn call
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        logger.store_metrics(loss=loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # calculate the action error
        with torch.no_grad():
            logger.store_metrics(
                action_error=torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )
