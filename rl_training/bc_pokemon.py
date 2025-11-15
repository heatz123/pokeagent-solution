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
    print("\nCreating evaluation environment...")
    env = make_env(args.rom_path, headless=True, base_fps=240)
    print(f"Environment created: {env.observation_space}, {env.action_space}")

    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=1_000_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load expert data
    print(f"\nLoading expert data from {args.expert_data_dir}...")
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

    print(f"\nPolicy network: {sum(p.numel() for p in policy_net.parameters())} parameters")

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
