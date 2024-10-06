import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import agent.networks as networks
from agent.base_agent import BaseAgent


class BSMPC(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def calc_encoder_loss(self, obs, z, action, reward):
        horizon = z.shape[0]
        batch_size = self.cfg.batch_size

        perm = np.random.permutation(batch_size)
        z2 = z[:, perm, :]
        reward2 = reward[:, perm, :]
        assert z2.shape == (horizon, batch_size, self.cfg.latent_dim)
        assert reward2.shape == (horizon, batch_size, 1)

        with torch.no_grad():
            target_z = self.model_target.encode(self.aug(obs))
            pred_next_z = self.model_target.next(target_z, action)
            pred_next_z2 = pred_next_z[:, perm, :]
            assert pred_next_z.shape == (horizon, batch_size, self.cfg.latent_dim)
            assert pred_next_z2.shape == (horizon, batch_size, self.cfg.latent_dim)

        z_dist = torch.sum(F.smooth_l1_loss(z, z2, reduction="none"), dim=-1)
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)
        transition_dist = torch.norm(pred_next_z - pred_next_z2, dim=-1)

        assert z_dist.shape == (horizon, batch_size), z_dist.shape
        assert r_dist.shape == (
            horizon,
            batch_size,
        ), r_dist.shape
        assert transition_dist.shape == (
            horizon,
            batch_size,
        ), transition_dist.shape

        target_bisimilarity = r_dist + self.cfg.discount * transition_dist
        loss = (z_dist - target_bisimilarity).pow(2)
        assert loss.shape == (
            horizon,
            batch_size,
        ), loss.shape

        return loss

    def calc_transition_model_loss(self, z, action, next_z):
        horizon = z.shape[0]
        batch_size = self.cfg.batch_size

        assert z.shape == (horizon, batch_size, self.cfg.latent_dim)

        # calculate consistency of the dynamics model
        pred_next_z = self.model.next(z, action)
        assert pred_next_z.shape == z.shape, pred_next_z.shape

        transition_loss = torch.sum(
            F.mse_loss(pred_next_z, next_z, reduction="none"), dim=-1
        )
        assert transition_loss.shape == (
            horizon,
            batch_size,
        ), transition_loss.shape

        return transition_loss

    def calc_reward_loss(self, z, action, reward):
        batch_size = self.cfg.batch_size
        horizon = z.shape[0]
        assert reward.shape == (horizon, batch_size, 1)

        # calculate reward loss
        pred_reward = self.model.reward(z, action)
        assert pred_reward.shape == reward.shape, pred_reward.shape

        reward_loss = torch.sum(
            F.mse_loss(pred_reward, reward, reduction="none"), dim=-1
        )
        assert reward_loss.shape == (
            horizon,
            batch_size,
        ), reward_loss.shape

        return reward_loss

    def calc_value_loss(self, z, action, next_z, reward):
        horizon = z.shape[0]
        batch_size = self.cfg.batch_size

        Q1, Q2 = self.model.Q(z, action)
        assert Q1.shape == (horizon, batch_size, 1)
        assert Q2.shape == (horizon, batch_size, 1)

        with torch.no_grad():
            td_target = reward + self.cfg.discount * torch.min(
                *self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std))
            )
            assert td_target.shape == (horizon, batch_size, 1)

        loss = F.mse_loss(Q1, td_target, reduction="none") + F.mse_loss(
            Q2, td_target, reduction="none"
        )
        loss = loss.squeeze(-1)
        assert loss.shape == (
            horizon,
            batch_size,
        ), loss.shape

        return loss, Q1, Q2, td_target

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
        zs = self.model.encode(self.aug(obs))
        assert zs.shape == (
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
        )

        with torch.no_grad():
            next_zs = self.model_target.encode(self.aug(obs[1:]))
            assert next_zs.shape == (
                self.cfg.horizon,
                self.cfg.batch_size,
                self.cfg.latent_dim,
            )

        rhos = torch.pow(
            self.cfg.rho, torch.arange(self.cfg.horizon, device=self.device)
        )
        encoder_loss = rhos.unsqueeze(1) * self.calc_encoder_loss(
            obs[:-1], zs[:-1], action, reward
        )
        transition_loss = rhos.unsqueeze(1) * self.calc_transition_model_loss(
            zs[:-1], action, next_zs
        )
        reward_loss = rhos.unsqueeze(1) * self.calc_reward_loss(zs[:-1], action, reward)
        value_loss, Q1, Q2, td_target = self.calc_value_loss(zs[:-1], action, next_zs, reward)
        value_loss = rhos.unsqueeze(1) * value_loss

        # TODO: Maybe summation
        encoder_loss = torch.mean(encoder_loss, dim=0)
        transition_loss = torch.mean(transition_loss, dim=0)
        reward_loss = torch.mean(reward_loss, dim=0)
        value_loss = torch.mean(value_loss, dim=0) / 2.0

        assert encoder_loss.shape == (self.cfg.batch_size,), encoder_loss.shape
        assert transition_loss.shape == (self.cfg.batch_size,), transition_loss.shape
        assert reward_loss.shape == (self.cfg.batch_size,), reward_loss.shape
        assert value_loss.shape == (self.cfg.batch_size,), value_loss.shape

        total_loss = (
            self.cfg.bisim_coef * encoder_loss.clamp(max=1e4)
            + self.cfg.consistency_coef * transition_loss.clamp(max=1e4)
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
        pi_loss = self.update_actor(zs.detach())

        # update target network
        if step % self.cfg.soft_update_freq == 0:
            for p, p_target in zip(
                self.model.parameters(), self.model_target.parameters()
            ):
                p_target.data.lerp_(p.data, self.cfg.soft_tau)

        self.model.eval()
        return {
            "encoder_loss": float(encoder_loss.mean().item()),
            "consistency_loss": float(transition_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "Q1": float(Q1.mean().item()),
            "Q2": float(Q2.mean().item()),
            "td_target": float(td_target.mean().item()),
        }
