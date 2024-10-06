import torch
import numpy as np

import torch
import torchrl.data.replay_buffers as buffers
from torchrl.data.replay_buffers.samplers import SliceSampler
from tensordict.tensordict import TensorDict


def convert_dict(env, obs, action=None, reward=None):
    """Creates a TensorDict for a new episode."""
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device="cpu")
    else:
        obs = obs.unsqueeze(0).cpu()

    if action is None:
        action = torch.full_like(env.rand_act(), float("nan"))

    if reward is None:
        reward = torch.tensor(float("nan"))

    data = TensorDict(
        dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
        ),
        batch_size=(1,),
    )
    return data


class ReplayBuffer:
    """
    Replay buffer based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = cfg.device
        self._capacity = min(int(cfg.max_buffer_size), int(cfg.train_steps))
        self._sampler = SliceSampler(
            num_slices=cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return buffers.ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, episodes: TensorDict):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info(device=self._device)
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in episodes.values()
            ]
        ) / len(episodes)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes/1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = self._device if 1.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        return self._reserve_buffer(
            buffers.LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, episode):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = episode["obs"]
        action = episode["action"][:-1]
        reward = episode["reward"][:-1].unsqueeze(-1)
        return self._to_device(obs, action, reward)

    def add(self, episode):
        """Add an episode to the buffer."""
        episode["episode"] = (
            torch.ones_like(episode["reward"], dtype=torch.int64) * self._num_eps
        )
        if self._num_eps == 0:
            self._buffer = self._init(episode)
        self._buffer.extend(episode)
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        episode = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(episode)
