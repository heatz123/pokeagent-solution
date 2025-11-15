"""
Replay buffer for IL/RL training
Based on cleanrl_utils/buffers.py
"""

import numpy as np
import torch
from typing import Dict, NamedTuple, Optional
import glob
import os


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms (BC, DAgger, DQN)

    Stores: (obs, action, reward, next_obs, done)

    Based on cleanrl's ReplayBuffer but simplified for Pokemon
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,  # gym.Space (Box)
        action_space,       # gym.Space (Discrete)
        device: str = "cpu",
        optimize_memory_usage: bool = False,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.optimize_memory_usage = optimize_memory_usage

        # Extract shapes
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # Allocate memory
        self.observations = np.zeros((buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        infos: Optional[Dict] = None,
    ) -> None:
        """Add a transition to the buffer"""
        # Store transition
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # Update pointer
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of transitions"""
        # Sample indices
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)

        # Convert to torch tensors
        return ReplayBufferSamples(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
        )

    def size(self) -> int:
        """Return current size of buffer"""
        return self.buffer_size if self.full else self.pos

    def __len__(self) -> int:
        return self.size()

    def save(self, path: str) -> None:
        """
        Save buffer to disk

        Args:
            path: Path to save buffer (will create .npz file)
        """
        np.savez_compressed(
            path,
            observations=self.observations,
            next_observations=self.next_observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            pos=self.pos,
            full=self.full,
        )

    def load(self, path: str) -> None:
        """
        Load buffer from disk

        Args:
            path: Path to load buffer from (.npz file)
        """
        data = np.load(path)

        # Restore arrays
        self.observations = data['observations']
        self.next_observations = data['next_observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.pos = int(data['pos'])
        self.full = bool(data['full'])

        print(f"✓ Loaded buffer from {path}")
        print(f"  Buffer size: {self.size():,} / {self.buffer_size:,}")


def load_expert_data_to_buffer(
    expert_data_dir: str,
    buffer: ReplayBuffer,
    max_episodes: Optional[int] = None,
    preprocess_fn=None,
) -> int:
    """
    Load pre-collected expert demonstrations into replay buffer

    Args:
        expert_data_dir: Directory containing episode_*.npz files
        buffer: ReplayBuffer to load data into
        max_episodes: Maximum number of episodes to load
        preprocess_fn: Function to preprocess observations (e.g., resize)

    Returns:
        Number of transitions loaded
    """
    episode_files = sorted(glob.glob(os.path.join(expert_data_dir, "episode_*.npz")))

    if len(episode_files) == 0:
        print(f"Warning: No episode files found in {expert_data_dir}")
        return 0

    if max_episodes:
        episode_files = episode_files[:max_episodes]

    total_transitions = 0

    print(f"Loading {len(episode_files)} episode files...")

    for episode_file in episode_files:
        data = np.load(episode_file, allow_pickle=True)

        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        dones = data['dones']

        # Add each transition
        for i in range(len(observations) - 1):
            obs = observations[i]
            next_obs = observations[i + 1]
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            # Preprocess if needed
            if preprocess_fn:
                obs = preprocess_fn(obs)
                next_obs = preprocess_fn(next_obs)

            buffer.add(obs, next_obs, action, reward, done)
            total_transitions += 1

    return total_transitions
