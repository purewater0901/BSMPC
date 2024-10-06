import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import agent.networks as networks
from agent.world_model import TOLD


class BaseAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.model = TOLD(cfg).to(self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()

        self.std = networks.linear_schedule(
            cfg.std_scheduler_init, cfg.min_std, cfg.std_scheduler_duration, 0
        )
        self.aug = networks.RandomShiftsAug(cfg)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
        }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d["model"])
        self.model_target.load_state_dict(d["model_target"])

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z = self.model.next(z, actions[t])
            reward = self.model.reward(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    def update_actor(self, zs):
        assert zs.shape == (
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
        )

        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)  # disable gradient for Q

        # Loss is a weighted sum of Q-values
        pis = self.model.pi(zs, self.cfg.min_std)
        assert pis.shape == (
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.action_dim,
        )

        qs = torch.min(*self.model.Q(zs, pis)).squeeze(-1)
        assert qs.shape == (self.cfg.horizon + 1, self.cfg.batch_size, )

        qs = torch.mean(qs, dim=-1)
        assert qs.shape == (self.cfg.horizon + 1,)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = torch.mean(-qs * rho) # TODO: Maybe summation

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def get_action(self, obs, step, eval_mode=False, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(
                self.cfg.action_dim, dtype=torch.float32, device=self.device
            ).uniform_(-1.0, 1.0)

        # Sample policy trajectories
        obs = obs.to(self.device).unsqueeze(0)
        horizon = int(
            min(
                self.cfg.horizon,
                networks.linear_schedule(
                    self.cfg.horizon_scheduler_init,
                    self.cfg.horizon,
                    self.cfg.horizon_scheduler_duration,
                    step,
                ),
            )
        )
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                horizon, num_pi_trajs, self.cfg.action_dim, device=self.device
            )
            z = self.model.encode(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.encode(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(
            horizon, self.cfg.action_dim, device=self.device
        )  # (horizon, action_dim)
        std = 2 * torch.ones(
            horizon, self.cfg.action_dim, device=self.device
        )  # (horizon, action_dim)
        if not t0 and hasattr(self, "_prev_mean") and len(mean) == len(self._prev_mean):
            mean[:-1, :] = self._prev_mean[1:, :]

        # Iterate CEM
        for i in range(self.cfg.mppi_iterations):
            noises = torch.randn(
                horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device
            )
            actions = (
                mean.unsqueeze(1) + std.unsqueeze(1) * noises
            )  # (horizon, num_samples, action_dim)
            actions = torch.clamp(actions, -1.0, 1.0)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = (
                self.estimate_value(z, actions, horizon).nan_to_num_(0).squeeze(-1)
            )  # (total_sample_num, )
            elite_idxs = torch.topk(value, self.cfg.num_elites, dim=0).indices
            elite_value = value[elite_idxs]  # (elite_num, )
            elite_actions = actions[:, elite_idxs]  # (horizon, elite_num, action_dim)

            # Update parameters
            max_value = elite_value.max(0)[0]
            weights = torch.softmax(
                self.cfg.temperature * (elite_value - max_value), dim=0
            )  # (elite_num, )
            _mean = torch.sum(
                weights.view(1, -1, 1) * elite_actions, dim=1
            )  # (horizon, action_dim)
            _std = torch.sqrt(
                torch.sum(
                    weights.view(1, -1, 1) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
            )
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs (actions: (horizon, action_dim))
        weights = weights.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(weights.shape[0]), p=weights)
        ]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]

        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a

    def update(self, obs, action, reward, step):
        pass
