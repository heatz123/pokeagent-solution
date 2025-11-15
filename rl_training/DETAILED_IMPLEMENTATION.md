# Detailed Implementation Plan - Pokemon IL

## File Structure (Final)

```
rl_training/
├── bc_pokemon.py               # Single-file Offline BC (cleanrl style)
├── dagger_pokemon.py           # Single-file Online DAgger (cleanrl style)
├── common/
│   ├── __init__.py
│   ├── networks.py            # ImpalaCNN architecture
│   ├── buffers.py             # ReplayBuffer (from cleanrl)
│   └── utils.py               # Preprocessing, evaluation, etc.
├── expert_data/               # Pre-collected demonstrations (for BC)
├── models/                    # Checkpoints
└── runs/                      # TensorBoard logs
```

---

## Part 1: Common Modules

### `common/buffers.py` (Based on cleanrl)

```python
"""
Replay buffer for IL/RL training
Based on cleanrl_utils/buffers.py
"""

import numpy as np
import torch
from typing import Dict, NamedTuple, Optional

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
    import glob
    import os

    episode_files = sorted(glob.glob(os.path.join(expert_data_dir, "episode_*.npz")))

    if max_episodes:
        episode_files = episode_files[:max_episodes]

    total_transitions = 0

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
```

### `common/networks.py` (ImpalaCNN)

```python
"""
Neural network architectures for Pokemon IL
Based on ImpalaCNN from cleanrl/qdagger_dqn_atari_jax_impalacnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        x = F.relu(x)
        return x


class ConvSequence(nn.Module):
    """Convolutional sequence: Conv -> MaxPool -> ResBlock x2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """
    ImpalaCNN architecture for Pokemon

    Input: (B, 3, 84, 84) - preprocessed RGB images
    Output: (B, action_dim) - action logits or Q-values

    Architecture:
    - 3x ConvSequence blocks (16 -> 32 -> 32 channels)
    - Flatten
    - FC(hidden_dim) -> FC(action_dim)
    """

    def __init__(self, action_dim: int = 10, hidden_dim: int = 256):
        super().__init__()

        # Three conv sequences
        self.conv_seq1 = ConvSequence(3, 16)   # 84x84 -> 42x42
        self.conv_seq2 = ConvSequence(16, 32)  # 42x42 -> 21x21
        self.conv_seq3 = ConvSequence(32, 32)  # 21x21 -> 11x11

        # Calculate flattened size: 32 * 11 * 11 = 3872
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32 * 11 * 11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, 3, 84, 84) or (B, 84, 84, 3) preprocessed observations

        Returns:
            (B, action_dim) logits/Q-values
        """
        # Ensure channel-first format (B, C, H, W)
        if x.shape[-1] == 3:  # If channel-last
            x = x.permute(0, 3, 1, 2)

        # Conv blocks
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)

        # FC layers
        x = self.fc(x)

        return x
```

### `common/utils.py` (Utilities)

```python
"""
Utility functions for Pokemon IL training
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple
import torchvision.transforms as T


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
    import sys
    sys.path.append('..')  # Add parent dir to import pokemon_env

    from pokemon_env.gym_env import PokemonEnv

    env = PokemonEnv(
        rom_path=rom_path,
        headless=headless,
        base_fps=base_fps,
        enable_milestones=True,
    )

    return env
```

---

## Part 2: `bc_pokemon.py` (Offline Behavioral Cloning)

```python
#!/usr/bin/env python3
"""
Offline Behavioral Cloning for Pokemon Emerald

Based on cleanrl style single-file implementation
Trains policy on pre-collected expert demonstrations

Usage:
    python bc_pokemon.py --expert-data-dir rl_training/expert_data --total-steps 100000
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Import common modules
from common.networks import ImpalaCNN
from common.buffers import ReplayBuffer, load_expert_data_to_buffer
from common.utils import evaluate_policy, make_env, preprocess_observation


@dataclass
class Args:
    # Experiment
    exp_name: str = "bc_pokemon"
    """Experiment name"""
    seed: int = 1
    """Random seed"""
    torch_deterministic: bool = True
    """Use deterministic PyTorch operations"""
    cuda: bool = True
    """Use CUDA if available"""
    track: bool = False
    """Track with Weights & Biases"""
    wandb_project_name: str = "pokemon-il"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity"""

    # Environment
    rom_path: str = "Emerald-GBAdvance/rom.gba"
    """Path to Pokemon ROM"""

    # Expert data
    expert_data_dir: str = "rl_training/expert_data"
    """Directory containing expert demonstrations (episode_*.npz)"""
    max_expert_episodes: Optional[int] = None
    """Maximum number of expert episodes to load (None = all)"""

    # Training
    total_steps: int = 100_000
    """Total training steps"""
    batch_size: int = 64
    """Batch size"""
    learning_rate: float = 1e-4
    """Learning rate"""
    weight_decay: float = 1e-5
    """Weight decay (L2 regularization)"""

    # Network
    hidden_dim: int = 256
    """Hidden dimension for FC layers"""

    # Evaluation
    eval_frequency: int = 5_000
    """Evaluate every N steps"""
    eval_episodes: int = 5
    """Number of episodes for evaluation"""

    # Checkpointing
    save_frequency: int = 10_000
    """Save checkpoint every N steps"""
    model_dir: str = "rl_training/models"
    """Directory to save models"""
    save_best: bool = True
    """Save best model by eval return"""


def train(args: Args):
    """Main training function"""

    # Setup
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment (for evaluation)
    env = make_env(args.rom_path, headless=True, base_fps=240)

    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=1_000_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load expert data
    print(f"Loading expert data from {args.expert_data_dir}...")
    n_transitions = load_expert_data_to_buffer(
        expert_data_dir=args.expert_data_dir,
        buffer=buffer,
        max_episodes=args.max_expert_episodes,
        preprocess_fn=preprocess_observation,
    )
    print(f"Loaded {n_transitions} expert transitions from {buffer.size()} total buffer entries")

    if buffer.size() == 0:
        raise ValueError(f"No expert data found in {args.expert_data_dir}")

    # Create policy network
    policy_net = ImpalaCNN(action_dim=env.action_space.n, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"Policy network: {sum(p.numel() for p in policy_net.parameters())} parameters")

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Training loop
    print("\n" + "="*60)
    print("Starting Behavioral Cloning training")
    print("="*60 + "\n")

    best_eval_return = -float('inf')
    start_time = time.time()

    for step in range(args.total_steps):
        # Sample batch from expert buffer
        batch = buffer.sample(args.batch_size)

        # Forward pass
        logits = policy_net(batch.observations)

        # Compute BC loss (cross-entropy)
        loss = F.cross_entropy(logits, batch.actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=-1) == batch.actions).float().mean()

            writer.add_scalar("train/bc_loss", loss.item(), step)
            writer.add_scalar("train/accuracy", accuracy.item(), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], step)

            if step % 1000 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step:6d}: Loss={loss.item():.4f}, Acc={accuracy.item():.2%}, SPS={steps_per_sec:.1f}")

        # Evaluation
        if step % args.eval_frequency == 0 and step > 0:
            print(f"\nEvaluating at step {step}...")
            eval_return, eval_std, info_dict = evaluate_policy(
                env, policy_net, args.eval_episodes, device
            )

            writer.add_scalar("eval/mean_return", eval_return, step)
            writer.add_scalar("eval/std_return", eval_std, step)
            writer.add_scalar("eval/mean_episode_length", info_dict['mean_length'], step)
            writer.add_scalar("eval/mean_milestones", info_dict['mean_milestones'], step)

            print(f"Eval: Return={eval_return:.2f}±{eval_std:.2f}, "
                  f"Length={info_dict['mean_length']:.1f}, "
                  f"Milestones={info_dict['mean_milestones']:.1f}")

            # Save best model
            if args.save_best and eval_return > best_eval_return:
                best_eval_return = eval_return
                save_path = os.path.join(args.model_dir, f"{args.exp_name}_best.pth")
                torch.save({
                    'step': step,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_return': eval_return,
                    'args': vars(args),
                }, save_path)
                print(f"✓ Saved best model: {save_path} (return={eval_return:.2f})")

        # Periodic checkpoint
        if step % args.save_frequency == 0 and step > 0:
            save_path = os.path.join(args.model_dir, f"{args.exp_name}_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"✓ Saved checkpoint: {save_path}")

    # Final save
    final_path = os.path.join(args.model_dir, f"{args.exp_name}_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation:")
    eval_return, eval_std, info_dict = evaluate_policy(
        env, policy_net, args.eval_episodes * 2, device
    )
    print(f"Final Return: {eval_return:.2f}±{eval_std:.2f}")
    print(f"Final Episode Length: {info_dict['mean_length']:.1f}")
    print(f"Final Milestones: {info_dict['mean_milestones']:.1f}")

    env.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
```

---

## Part 3: `dagger_pokemon.py` (Online DAgger with Code Agent)

```python
#!/usr/bin/env python3
"""
Online DAgger for Pokemon Emerald

Based on cleanrl style single-file implementation
Uses code_agent as expert oracle during online rollouts

Usage:
    python dagger_pokemon.py --n-iterations 10 --episodes-per-iter 5
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Import common modules
from common.networks import ImpalaCNN
from common.buffers import ReplayBuffer
from common.utils import evaluate_policy, make_env, preprocess_observation

# Import code agent
import sys
sys.path.append('..')
from agent.code_agent import CodeAgent


@dataclass
class Args:
    # Experiment
    exp_name: str = "dagger_pokemon"
    """Experiment name"""
    seed: int = 1
    """Random seed"""
    torch_deterministic: bool = True
    """Use deterministic PyTorch operations"""
    cuda: bool = True
    """Use CUDA if available"""
    track: bool = False
    """Track with Weights & Biases"""
    wandb_project_name: str = "pokemon-il"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity"""

    # Environment
    rom_path: str = "Emerald-GBAdvance/rom.gba"
    """Path to Pokemon ROM"""

    # DAgger
    n_iterations: int = 10
    """Number of DAgger iterations"""
    episodes_per_iter: int = 5
    """Rollout episodes per DAgger iteration"""
    train_steps_per_iter: int = 10_000
    """Training steps per DAgger iteration"""
    max_episode_steps: int = 1000
    """Max steps per episode"""

    # Training
    batch_size: int = 64
    """Batch size"""
    learning_rate: float = 1e-4
    """Learning rate"""
    weight_decay: float = 1e-5
    """Weight decay (L2 regularization)"""

    # Network
    hidden_dim: int = 256
    """Hidden dimension for FC layers"""

    # Expert (Code Agent)
    expert_model: str = "claude-sonnet-4-5-20250929"
    """Expert model for code_agent"""
    expert_timeout: int = 60
    """Timeout for expert queries (seconds)"""

    # Evaluation
    eval_frequency: int = 1
    """Evaluate every N iterations"""
    eval_episodes: int = 5
    """Number of episodes for evaluation"""

    # Checkpointing
    model_dir: str = "rl_training/models"
    """Directory to save models"""
    save_best: bool = True
    """Save best model by eval return"""


class ExpertOracle:
    """Wrapper for querying code_agent as expert"""

    def __init__(self, code_agent: CodeAgent):
        self.code_agent = code_agent
        self.action_map = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]

    def get_action(self, obs, info: dict, timeout: int = 60) -> int:
        """
        Query code_agent for expert action at current state

        Args:
            obs: PIL Image (screenshot)
            info: Dict with state information
            timeout: Query timeout in seconds

        Returns:
            action: int (0-9, GBA button index)
        """
        try:
            # Query code agent (this will call LLM and execute generated code)
            action_str = self.code_agent.get_action(obs, info, timeout=timeout)

            # Convert action string to index
            if action_str in self.action_map:
                action = self.action_map.index(action_str)
            else:
                print(f"Warning: Invalid action '{action_str}' from expert, using random")
                action = random.randint(0, 9)

            return action

        except Exception as e:
            print(f"Error querying expert: {e}")
            print("Using random action as fallback")
            return random.randint(0, 9)


def rollout_with_expert(
    env,
    policy_net: torch.nn.Module,
    expert: ExpertOracle,
    buffer: ReplayBuffer,
    device: torch.device,
    n_episodes: int,
    max_steps: int,
    use_student_action: bool = True,
) -> dict:
    """
    Rollout episodes using student policy, query expert for labels

    Args:
        env: PokemonEnv
        policy_net: Student policy
        expert: Expert oracle (code_agent)
        buffer: ReplayBuffer to store (obs, expert_action) pairs
        device: torch device
        n_episodes: Number of episodes to rollout
        max_steps: Max steps per episode
        use_student_action: Use student action for rollout (True) or expert (False)

    Returns:
        stats: Dict with rollout statistics
    """
    policy_net.eval()

    episode_returns = []
    episode_lengths = []
    expert_queries = 0
    expert_failures = 0

    for ep in range(n_episodes):
        print(f"\nRollout Episode {ep + 1}/{n_episodes}")
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            # Get student action
            obs_preprocessed = preprocess_observation(obs)
            obs_tensor = torch.FloatTensor(obs_preprocessed).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy_net(obs_tensor)
                student_action = logits.argmax(dim=-1).item()

            # Query expert for label
            print(f"  Step {episode_length}: Querying expert...")
            expert_queries += 1
            try:
                expert_action = expert.get_action(obs, info, timeout=60)
                print(f"  Expert action: {expert_action} (Student would do: {student_action})")
            except Exception as e:
                print(f"  Expert query failed: {e}")
                expert_action = student_action  # Fallback to student
                expert_failures += 1

            # Execute action (student or expert depending on flag)
            execute_action = student_action if use_student_action else expert_action
            next_obs, reward, terminated, truncated, info = env.step(execute_action)
            done = terminated or truncated

            # Add (obs, EXPERT_action) to buffer (DAgger key idea)
            buffer.add(
                obs=obs_preprocessed,
                next_obs=preprocess_observation(next_obs),
                action=expert_action,  # Always label with expert action
                reward=reward,
                done=done,
            )

            obs = next_obs
            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        print(f"  Episode {ep + 1} done: Return={episode_return:.2f}, Length={episode_length}")

    policy_net.train()

    stats = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'expert_queries': expert_queries,
        'expert_failures': expert_failures,
    }

    return stats


def train(args: Args):
    """Main DAgger training function"""

    # Setup
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environments
    env = make_env(args.rom_path, headless=True, base_fps=120)  # Training env
    eval_env = make_env(args.rom_path, headless=True, base_fps=240)  # Eval env (faster)

    # Create policy network
    policy_net = ImpalaCNN(action_dim=env.action_space.n, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"Policy network: {sum(p.numel() for p in policy_net.parameters())} parameters")

    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=1_000_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Create expert oracle (code_agent)
    print(f"\nInitializing expert oracle (code_agent with {args.expert_model})...")
    code_agent = CodeAgent(model=args.expert_model)
    expert = ExpertOracle(code_agent)
    print("✓ Expert oracle ready")

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # DAgger iterations
    print("\n" + "="*60)
    print("Starting DAgger training")
    print("="*60 + "\n")

    best_eval_return = -float('inf')
    global_step = 0

    for iteration in range(args.n_iterations):
        iter_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"DAgger Iteration {iteration + 1}/{args.n_iterations}")
        print(f"{'='*60}\n")

        # Phase 1: Rollout with current policy, query expert for labels
        print(f"Phase 1: Rollout ({args.episodes_per_iter} episodes)")
        print("-" * 60)

        rollout_stats = rollout_with_expert(
            env=env,
            policy_net=policy_net,
            expert=expert,
            buffer=buffer,
            device=device,
            n_episodes=args.episodes_per_iter,
            max_steps=args.max_episode_steps,
            use_student_action=True,  # Use student policy for rollout
        )

        print(f"\nRollout complete:")
        print(f"  Mean Return: {rollout_stats['mean_return']:.2f}±{rollout_stats['std_return']:.2f}")
        print(f"  Mean Length: {rollout_stats['mean_length']:.1f}")
        print(f"  Expert Queries: {rollout_stats['expert_queries']}")
        print(f"  Expert Failures: {rollout_stats['expert_failures']}")
        print(f"  Buffer Size: {buffer.size()}")

        # Log rollout stats
        writer.add_scalar("rollout/mean_return", rollout_stats['mean_return'], iteration)
        writer.add_scalar("rollout/mean_length", rollout_stats['mean_length'], iteration)
        writer.add_scalar("rollout/buffer_size", buffer.size(), iteration)

        # Phase 2: Train on aggregated dataset
        print(f"\nPhase 2: Training ({args.train_steps_per_iter} steps)")
        print("-" * 60)

        train_losses = []
        train_accs = []

        for train_step in range(args.train_steps_per_iter):
            # Sample batch from buffer
            batch = buffer.sample(args.batch_size)

            # Forward pass
            logits = policy_net(batch.observations)

            # Compute BC loss
            loss = F.cross_entropy(logits, batch.actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                accuracy = (logits.argmax(dim=-1) == batch.actions).float().mean()

            train_losses.append(loss.item())
            train_accs.append(accuracy.item())

            global_step += 1

            # Log
            if train_step % 100 == 0:
                writer.add_scalar("train/bc_loss", loss.item(), global_step)
                writer.add_scalar("train/accuracy", accuracy.item(), global_step)

            if train_step % 1000 == 0:
                print(f"  Step {train_step:5d}/{args.train_steps_per_iter}: "
                      f"Loss={loss.item():.4f}, Acc={accuracy.item():.2%}")

        print(f"\nTraining complete:")
        print(f"  Mean Loss: {np.mean(train_losses):.4f}")
        print(f"  Mean Accuracy: {np.mean(train_accs):.2%}")

        # Phase 3: Evaluate
        if (iteration + 1) % args.eval_frequency == 0:
            print(f"\nPhase 3: Evaluation ({args.eval_episodes} episodes)")
            print("-" * 60)

            eval_return, eval_std, info_dict = evaluate_policy(
                eval_env, policy_net, args.eval_episodes, device
            )

            print(f"Evaluation results:")
            print(f"  Return: {eval_return:.2f}±{eval_std:.2f}")
            print(f"  Episode Length: {info_dict['mean_length']:.1f}")
            print(f"  Milestones: {info_dict['mean_milestones']:.1f}")

            writer.add_scalar("eval/mean_return", eval_return, iteration)
            writer.add_scalar("eval/std_return", eval_std, iteration)
            writer.add_scalar("eval/mean_length", info_dict['mean_length'], iteration)
            writer.add_scalar("eval/mean_milestones", info_dict['mean_milestones'], iteration)

            # Save best model
            if args.save_best and eval_return > best_eval_return:
                best_eval_return = eval_return
                save_path = os.path.join(args.model_dir, f"{args.exp_name}_best.pth")
                torch.save({
                    'iteration': iteration,
                    'global_step': global_step,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_return': eval_return,
                    'args': vars(args),
                }, save_path)
                print(f"\n✓ Saved best model: {save_path} (return={eval_return:.2f})")

        # Save checkpoint
        save_path = os.path.join(args.model_dir, f"{args.exp_name}_iter{iteration+1}.pth")
        torch.save({
            'iteration': iteration,
            'global_step': global_step,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"✓ Saved checkpoint: {save_path}")

        iter_time = time.time() - iter_start_time
        print(f"\nIteration {iteration + 1} completed in {iter_time:.1f}s")

    # Final save
    final_path = os.path.join(args.model_dir, f"{args.exp_name}_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\n✓ DAgger training complete! Final model saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation (10 episodes):")
    eval_return, eval_std, info_dict = evaluate_policy(
        eval_env, policy_net, 10, device
    )
    print(f"Final Return: {eval_return:.2f}±{eval_std:.2f}")
    print(f"Final Episode Length: {info_dict['mean_length']:.1f}")
    print(f"Final Milestones: {info_dict['mean_milestones']:.1f}")

    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
```

---

## Usage Examples

### 1. Offline BC Training

```bash
# Collect expert data first (separate script needed)
python collect_expert_data.py --n-episodes 20

# Train BC
python bc_pokemon.py \
    --expert-data-dir rl_training/expert_data \
    --total-steps 100000 \
    --batch-size 64 \
    --eval-frequency 5000
```

### 2. Online DAgger Training

```bash
# Train DAgger (no pre-collected data needed)
python dagger_pokemon.py \
    --n-iterations 10 \
    --episodes-per-iter 5 \
    --train-steps-per-iter 10000 \
    --expert-model "claude-sonnet-4-5-20250929"
```

### 3. Load and Evaluate Model

```python
from common.networks import ImpalaCNN
from common.utils import make_env, evaluate_policy
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = ImpalaCNN(action_dim=10).to(device)
checkpoint = torch.load("rl_training/models/bc_pokemon_best.pth")
policy_net.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
env = make_env("Emerald-GBAdvance/rom.gba")
mean_return, std_return, info = evaluate_policy(env, policy_net, 10, device)
print(f"Return: {mean_return:.2f}±{std_return:.2f}")
```

---

## Next Steps

1. **Implement common modules** (`common/networks.py`, `common/buffers.py`, `common/utils.py`)
2. **Implement BC** (`bc_pokemon.py`)
3. **Implement data collection script** (for BC)
4. **Test BC training** on small dataset
5. **Implement DAgger** (`dagger_pokemon.py`)
6. **Test DAgger** with code_agent integration
7. **Compare BC vs DAgger** performance

## Implementation Priority

Week 1:
- ✅ Design complete
- ⏳ Implement `common/` modules
- ⏳ Implement `bc_pokemon.py`
- ⏳ Test BC end-to-end

Week 2:
- ⏳ Implement `dagger_pokemon.py`
- ⏳ Integrate code_agent
- ⏳ Test DAgger end-to-end

Week 3:
- ⏳ Run full experiments
- ⏳ Compare and analyze
- ⏳ Optimize and iterate
