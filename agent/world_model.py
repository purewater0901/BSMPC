import torch
import torch.nn as nn
import agent.networks as networks

class TOLD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._encoder = networks.build_encoder(cfg)
        self._dynamics = networks.build_mlp(
            cfg.latent_dim + cfg.action_dim,
            cfg.hidden_dim,
            cfg.latent_dim,
            num_hidden_layers=1,
        )
        self._reward = networks.build_mlp(
            cfg.latent_dim + cfg.action_dim, cfg.hidden_dim, 1, num_hidden_layers=1
        )
        self._pi = networks.build_mlp(
            cfg.latent_dim, cfg.hidden_dim, cfg.action_dim, num_hidden_layers=1
        )
        self._Q1 = networks.build_q(
            cfg.latent_dim + cfg.action_dim, cfg.hidden_dim, output_dim=1
        )
        self._Q2 = networks.build_q(
            cfg.latent_dim + cfg.action_dim, cfg.hidden_dim, output_dim=1
        )

        self.apply(networks.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for network in [self._Q1, self._Q2]:
            for param in network.parameters():
                param.requires_grad_(enable)

    def encode(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def reward(self, z, a):
        """Predicts single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return networks.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)
