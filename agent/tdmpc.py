import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import agent.networks as networks
from agent.base_agent import BaseAgent


class TDMPC(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def calc_td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.encode(next_obs)
        return reward + self.cfg.discount * torch.min(
            *self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std))
        )

    def update(self, obs, action, reward, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        self.optim.zero_grad(set_to_none=True)
        self.std = networks.linear_schedule(
            self.cfg.std_scheduler_init,
            self.cfg.min_std,
            self.cfg.std_scheduler_duration,
            step,
        )
        self.model.train()

        # Representation
        z = self.model.encode(self.aug(obs[0]))
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        zs[0] = z.detach()

        consistency_loss, reward_loss, value_loss = 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z = self.model.next(z, action[t])  # (batch_size, latent_size)
            reward_pred = self.model.reward(z, action[t])
            zs[t+1] = z.detach()

            # calc target value
            with torch.no_grad():
                next_obs = self.aug(obs[t + 1])
                next_z = self.model_target.encode(next_obs)
                td_target = self.calc_td_target(next_obs, reward[t])

            # Losses
            rho = self.cfg.rho**t
            consistency_loss += rho * torch.mean(
                F.mse_loss(z, next_z, reduction="none"), dim=1, keepdim=True
            )
            reward_loss += rho * F.mse_loss(reward_pred, reward[t], reduction="none")
            value_loss += rho * (
                F.mse_loss(Q1, td_target, reduction="none")
                + F.mse_loss(Q2, td_target, reduction="none")
            )

        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (2.0 * self.cfg.horizon)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss.clamp(max=1e4)
            + self.cfg.reward_coef * reward_loss.clamp(max=1e4)
            + self.cfg.value_coef * value_loss.clamp(max=1e4)
        ).mean()

        # Optimize model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()

        # Update policy
        pi_loss = self.update_actor(zs)

        # update target network
        if step % self.cfg.soft_update_freq == 0:
            for p, p_target in zip(
                self.model.parameters(), self.model_target.parameters()
            ):
                p_target.data.lerp_(p.data, self.cfg.soft_tau)

        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "Q1": float(Q1.mean().item()),
            "Q2": float(Q2.mean().item()),
            "td_target": float(td_target.mean().item()),
        }
