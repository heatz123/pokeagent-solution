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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            # Note: code_agent.get_action returns button string directly
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
    print("\nCreating environments...")
    env = make_env(args.rom_path, headless=True, base_fps=120)  # Training env
    eval_env = make_env(args.rom_path, headless=True, base_fps=240)  # Eval env (faster)
    print("Environments created")

    # Create policy network
    policy_net = ImpalaCNN(action_dim=env.action_space.n, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"\nPolicy network: {sum(p.numel() for p in policy_net.parameters())} parameters")

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
