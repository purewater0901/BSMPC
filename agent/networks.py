import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def build_mlp(
    input_dim, hidden_dim, output_dim, num_hidden_layers, activate_fn=nn.ELU()
):
    layers = [nn.Linear(input_dim, hidden_dim), activate_fn]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), activate_fn]
    layers += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*layers)


def build_q(input_dim, hidden_dim, output_dim):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = int(3 * cfg.frame_stack)
        layers = [
            NormalizeImg(),
            nn.Conv2d(C, cfg.num_channels, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
        ]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        horizon = None
        if x.dim() == 5:
            horizon, batch_size, channel, height, width = x.size()
            x = x.view(horizon * batch_size, channel, height, width)

        x = self.network(x)

        if horizon:
            x = x.view(horizon, batch_size, -1)

        return x

def build_encoder(cfg):
    """Returns a TOLD encoder."""
    if cfg.modality == "pixels":
        model = ImageEncoder(cfg)
        return model
    else:
        layers = [
            nn.Linear(cfg.obs_shape[0], cfg.encoder_hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.encoder_hidden_dim, cfg.latent_dim),
        ]
        return nn.Sequential(*layers)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def linear_schedule(init, final, duration, step):
    mix = np.clip(step / duration, 0.0, 1.0)
    return (1.0 - mix) * init + mix * final

class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    def __init__(self, cfg):
        super().__init__()
        self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

    def forward(self, x):
        if not self.pad:
            return x

        # change value type
        x = x.float()

        has_reshaped = False
        if x.dim() == 5:
            horizon, n, c, h, w = x.size()
            x = x.reshape(-1, c, h, w)
            has_reshaped = True
        else:
            n, c, h, w = x.size()
            horizon = 1
        assert h == w


        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(horizon * n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(horizon * n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift

        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        if has_reshaped:
            x = x.reshape(horizon, n, c, h, w)

        return x
