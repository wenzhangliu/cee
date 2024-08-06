from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical


class ActionModelTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, action_model, discrete, new_logger, cat_dim=1):
        self.action_model = action_model
        self.cat_dim = cat_dim
        self.discrete = discrete
        self.new_logger = new_logger
        self.nupdates = 0

    def train_step(self, batch, **kwargs):
        self.nupdates += 1
        if self.discrete:
            loss_item = self.train_step_discrete(batch, **kwargs)
        else:
            loss_item = self.train_step_continuous(batch)
        return loss_item

    def train_step_discrete(self, batch, pi, max_grad_norm):
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # we use the "q_net" output to get action probabilities
        new_x = {"image":x, "vector":pi.float()}
        predicted = self.action_model.q_net(new_x)
        loss = torch.nn.CrossEntropyLoss()(predicted, batch.actions.view(-1))
        m = Categorical(logits=predicted)
        # Optimize the action model
        self.action_model.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_grad_norm)
        self.action_model.optimizer.step()
        acc = (predicted.argmax(dim=1) == batch.actions[:, 0]).float().mean().item()
        new_logger = self.new_logger
        new_logger.record("action model/a_loss", loss.item())
        new_logger.record("action model/a_accuracy", acc)
        new_logger.record("action model/a_entropy", torch.mean(m.entropy()).item())
        new_logger.record("action model/a_hist", torch.histc(batch.actions.float(), bins=predicted.shape[1]).tolist())
        new_logger.record("action model/a_n_updates", self.nupdates)
        return loss.item()

    def train_step_continuous(self, batch):
        # 1. build s,s'=f(s,a) distribution function
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # 1.1 calculate mu, sigma of the Gaussian action model
        mu, log_std, _ = self.action_model.actor.get_action_dist_params(x)
        # 1.2 update probability distribution with calculated mu, log_std
        self.action_model.actor.action_dist.proba_distribution(mu, log_std)
        # 2 use N(ss') to calculate the probability of actually played action
        a_logp = self.action_model.actor.action_dist.log_prob(batch.actions)

        # Optimize the action model
        loss = -a_logp.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.record("action model/loss", loss.item())
        return loss.item()