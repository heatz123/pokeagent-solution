"""
Utility functions for Pokemon IL training
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple
import torchvision.transforms as T
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Preprocessing transform (resize + normalize)
preprocess_transform = T.Compose([
    T.Resize((84, 84)),
    T.ToTensor(),  # Also converts to (C, H, W) and [0, 1]
])


def preprocess_observation(obs: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Preprocess observation for network input

    Args:
        obs: PIL Image (160, 240, 3) or numpy array

    Returns:
        np.ndarray (3, 84, 84) float32 [0, 1]
    """
    if isinstance(obs, np.ndarray):
        obs = Image.fromarray(obs.astype(np.uint8))

    # Apply transform
    obs_tensor = preprocess_transform(obs)

    # Convert to numpy
    obs_np = obs_tensor.numpy()

    return obs_np


def evaluate_policy(
    env,
    policy_net: torch.nn.Module,
    n_episodes: int,
    device: torch.device,
    max_steps: int = 10000,
    deterministic: bool = True,
) -> Tuple[float, float, dict]:
    """
    Evaluate policy on environment

    Args:
        env: PokemonEnv instance
        policy_net: Policy network
        n_episodes: Number of evaluation episodes
        device: torch device
        max_steps: Max steps per episode
        deterministic: Use argmax action (True) or sample (False)

    Returns:
        mean_return: Average episode return
        std_return: Std of episode returns
        info_dict: Additional info (episode lengths, milestones, etc.)
    """
    policy_net.eval()

    episode_returns = []
    episode_lengths = []
    milestones_reached = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            # Preprocess and get action
            obs_preprocessed = preprocess_observation(obs)
            obs_tensor = torch.FloatTensor(obs_preprocessed).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy_net(obs_tensor)

                if deterministic:
                    action = logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Extract milestone info if available
        if 'milestones' in info:
            completed = sum(1 for m in info['milestones'].values() if m.get('completed', False))
            milestones_reached.append(completed)

    policy_net.train()

    info_dict = {
        'episode_lengths': episode_lengths,
        'mean_length': np.mean(episode_lengths),
        'milestones_reached': milestones_reached,
        'mean_milestones': np.mean(milestones_reached) if milestones_reached else 0,
    }

    return np.mean(episode_returns), np.std(episode_returns), info_dict


def make_env(rom_path: str, headless: bool = True, base_fps: int = 240):
    """Create PokemonEnv instance"""
    # Import pokemon_env from parent directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from pokemon_env.gym_env import PokemonEnv

    env = PokemonEnv(
        rom_path=rom_path,
        headless=headless,
        base_fps=base_fps,
        enable_milestones=True,
    )

    return env
