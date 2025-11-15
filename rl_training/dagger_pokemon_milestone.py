#!/usr/bin/env python3
"""
DAgger for specific Pokemon Emerald milestone (RIVAL_HOUSE)

Uses successful policy as expert (no LLM calls, very fast)
Trains on single milestone to verify that policy network can fit expert behavior

Usage:
    python dagger_pokemon_milestone.py \
        --milestone RIVAL_HOUSE \
        --n-iterations 10 \
        --episodes-per-iter 10
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import tyro

# Import Pokemon environment (BEFORE torch to avoid mgba-CUDA deadlock)
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokemon_env.gym_env import PokemonEnv
from utils.tool_loader import load_tools
from utils.custom_milestone_definitions import CUSTOM_MILESTONES
from utils.server_milestone_definitions import SERVER_MILESTONES

# NOTE: torch and related modules will be imported AFTER environment initialization
# This prevents mgba-CUDA/PyTorch conflict that causes load_raw_state to hang


@dataclass
class Args:
    # Experiment
    exp_name: str = "dagger_milestone"
    """Experiment name"""
    seed: int = 1
    """Random seed"""
    torch_deterministic: bool = True
    """Use deterministic PyTorch operations"""
    cuda: bool = True
    """Use CUDA if available"""
    track: bool = False
    """Track with Weights & Biases"""
    wandb_project_name: str = "pokemon-dagger-milestone"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity"""

    # Environment
    rom_path: str = "Emerald-GBAdvance/rom.gba"
    """Path to Pokemon ROM (using separate copy to avoid conflicts)"""
    milestone: str = "RIVAL_HOUSE"
    """Milestone to train on"""
    starting_state: str = ".milestone_trainer_cache/milestone_states/CLOCK_SET_completed.state"
    """Starting state for milestone"""
    max_episode_steps: int = 50
    """Max steps per episode"""

    # DAgger
    n_iterations: int = 10
    """Number of DAgger iterations"""
    episodes_per_iter: int = 10
    """Rollout episodes per DAgger iteration"""
    train_steps_per_iter: int = 100
    """Training steps per DAgger iteration"""
    pre_collect_episodes: int = 1
    """Number of expert-only episodes to collect before training (exits if expert fails)"""

    # Training
    batch_size: int = 64
    """Batch size"""
    learning_rate: float = 3e-4
    """Learning rate"""
    weight_decay: float = 1e-5
    """Weight decay (L2 regularization)"""

    # Network
    hidden_dim: int = 256
    """Hidden dimension for FC layers"""

    # Expert Policy
    expert_policy_path: str = ".milestone_trainer_cache/successful_policies/RIVAL_HOUSE.py"
    """Path to successful expert policy"""
    expert_verbose: bool = True
    """Print expert policy logs during execution (for debugging)"""

    # Evaluation
    eval_frequency: int = 1
    """Evaluate every N iterations"""
    eval_episodes: int = 1
    """Number of episodes for evaluation"""

    # Checkpointing
    model_dir: str = "models"
    """Directory to save models"""
    save_best: bool = True
    """Save best model by eval success rate"""
    resume_from: Optional[str] = None
    """Resume from checkpoint path (e.g., models/dagger_milestone_RIVAL_HOUSE_iter5.pth)"""
    skip_collection: bool = False
    """Skip episode collection, only train on existing buffer (use with resume_from)"""

    # Video Recording
    record_video: bool = True
    """Record episode videos"""
    video_dir: str = "dagger_videos"
    """Directory to save videos"""
    record_rollout_videos: bool = True
    """Record videos during rollout (for debugging)"""
    max_videos_per_iter: int = 2
    """Max number of videos to save per iteration (to save space)"""

    # Expert Evaluation
    eval_expert_only: bool = False
    """Only evaluate expert policy (no training)"""
    eval_expert_episodes: int = 1
    """Number of episodes for expert evaluation"""


def save_frames_as_video(frames, save_path, fps=30):
    """
    Save list of PIL Images as video file

    Args:
        frames: List of PIL Images
        save_path: Path to save video (.mp4)
        fps: Frames per second

    Returns:
        save_path if successful, None otherwise
    """
    try:
        import imageio
        import numpy as np

        # Convert PIL Images to numpy arrays
        frames_np = [np.array(frame) for frame in frames]

        # Save as video
        imageio.mimsave(save_path, frames_np, fps=fps)
        return save_path
    except Exception as e:
        print(f"Warning: Failed to save video to {save_path}: {e}")
        return None


class ExpertPolicyOracle:
    """
    Wrapper for pre-written successful policy as expert

    Much faster than CodeAgent (no LLM calls)
    Executes the deterministic successful policy code

    Features:
    - VLM support: Policies can use add_to_state_schema() for visual queries
    - Tool injection: All tools available to policies
    - State object: Policies receive State object with lazy VLM evaluation
    """

    def __init__(self, policy_path: str, tools: dict, verbose: bool = False):
        """
        Args:
            policy_path: Path to successful policy .py file
            tools: Dict of available tools (find_path_action, navigate_ui, etc.)
            verbose: If True, print expert logs during execution
        """
        self.policy_path = policy_path
        self.tools = tools
        self.action_map = ["a", "b", "start", "select", "up", "down", "left", "right", "l", "r"]
        self.verbose = verbose
        self.policy_logs = []  # Store logs from policy execution

        # VLM support (same as CodeAgentEnvWrapper)
        from utils.vlm_caller import VLMCaller
        from utils.vlm_state import get_global_schema_registry, add_to_state_schema

        vlm_model = os.getenv("VLM_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
        self.vlm_caller = VLMCaller(model=vlm_model)
        self.schema_registry = get_global_schema_registry()
        print(f"🔍 VLM enabled with model: {vlm_model}")

        # Load policy code
        print(f"Loading expert policy from {policy_path}")
        with open(policy_path, "r") as f:
            policy_code = f.read()

        # Compile policy with tools and VLM support
        exec_globals = {
            **tools,  # Inject tool functions
            "log": self._log,
            "add_to_state_schema": add_to_state_schema,  # Inject VLM schema function
        }
        exec(policy_code, exec_globals)

        if "run" not in exec_globals:
            raise ValueError(f"No run() function found in {policy_path}")

        self.policy_fn = exec_globals["run"]
        print("✓ Expert policy loaded successfully")

    def _log(self, msg: str):
        """Log function for policy execution"""
        self.policy_logs.append(str(msg))
        if self.verbose:
            print(f"    [EXPERT] {msg}")

    def get_action(self, obs, info: dict) -> int:
        """
        Query expert policy for action at current state

        Args:
            obs: PIL Image (screenshot)
            info: Dict with state information (raw from env)

        Returns:
            action: int (0-9, GBA button index)
        """
        try:
            # Clear policy logs for this step
            self.policy_logs = []

            # Convert state to dict format with ASCII map (same as milestone_trainer)
            # This generates the ASCII map that pathfinding tools need
            from utils.state_formatter import convert_state_to_dict
            from utils.vlm_state import State

            formatted_state = convert_state_to_dict(info)

            # Create State object with VLM support (same as CodeAgentEnvWrapper)
            state_obj = State(
                base_data=formatted_state,
                schema_registry=self.schema_registry,
                vlm_caller=lambda screenshot, prompt, return_type: self.vlm_caller.call(
                    screenshot, prompt, return_type
                ),
                screenshot=obs,  # PIL Image
            )

            # Execute policy with State object (VLM support!)
            action_str = self.policy_fn(state_obj)

            # Log VLM accesses if verbose
            vlm_log = state_obj.get_vlm_access_log()
            if vlm_log and self.verbose:
                print(f"    🔍 VLM queries made: {len(vlm_log)}")
                for entry in vlm_log:
                    print(f"       {entry['key']}: {entry['result']} ({entry['return_type']})")

            # Convert to int
            action_str_lower = str(action_str).lower()
            if action_str_lower in self.action_map:
                action = self.action_map.index(action_str_lower)
            else:
                # Fallback
                print(f"Warning: Unknown action '{action_str}' from expert, using 'b'")
                action = 1  # B button

            return action

        except Exception as e:
            print(f"Error executing expert policy: {e}")
            import traceback

            traceback.print_exc()
            return 1  # Default: B


def check_milestone_completion(milestone_id: str, info: dict) -> bool:
    """
    Check if milestone is completed based on game state

    Args:
        milestone_id: Milestone ID (e.g., "RIVAL_HOUSE", "MAY_ROUTE103_INTERACTION")
        info: Game state dict

    Returns:
        True if milestone completed
    """
    # Search in both custom and server milestone lists
    for milestones in [CUSTOM_MILESTONES, SERVER_MILESTONES]:
        for milestone in milestones:
            if milestone["id"] == milestone_id:
                check_fn = milestone.get("check_fn")
                if check_fn:
                    try:
                        return check_fn(info, None)
                    except Exception as e:
                        print(f"Warning: Error checking milestone {milestone_id}: {e}")
                        return False

    # Milestone not found
    print(f"Warning: No check function found for milestone {milestone_id}")
    return False


def check_party_fainted(info: dict) -> bool:
    """
    Check if all Pokemon in the party have fainted (HP = 0)

    Args:
        info: Game state dict

    Returns:
        True if all Pokemon in party have 0 HP (party wiped out)
    """
    try:
        party = info.get("player", {}).get("party", [])

        # If no party exists, consider it as not fainted (avoid false positives)
        if not party:
            return False

        # Check if all Pokemon have 0 current HP
        all_fainted = all(pokemon.get("current_hp", 1) == 0 for pokemon in party)

        return all_fainted
    except Exception as e:
        print(f"Warning: Error checking party HP: {e}")
        return False


def collect_expert_demonstrations(
    env: PokemonEnv,
    expert: ExpertPolicyOracle,
    buffer: Any,  # ReplayBuffer (imported later)
    n_episodes: int,
    max_steps: int,
    milestone_id: str,
    video_dir: str,
) -> None:
    """
    Collect expert-only demonstrations to initialize buffer

    If expert fails any episode, saves video and exits program.
    This ensures the expert policy works before starting DAgger.

    Args:
        env: PokemonEnv
        expert: Expert oracle
        buffer: ReplayBuffer to store demonstrations
        n_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        milestone_id: Milestone ID for completion check
        video_dir: Directory to save failure videos
    """
    print("\n" + "=" * 70)
    print("Phase 0: Collecting Expert Demonstrations")
    print("=" * 70 + "\n")
    print(f"Collecting {n_episodes} expert episodes...")
    print("(Will exit if expert fails - this validates expert policy)\n")

    for ep in range(n_episodes):
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        obs, info = env.reset()

        # Print initial state
        player = info.get("player", {})
        location = player.get("location", "UNKNOWN")
        position = player.get("position", {})
        x, y = position.get("x", -1), position.get("y", -1)
        print(f"  Starting location: {location}")
        print(f"  Starting position: ({x}, {y})")

        # Check if already completed at start (should not happen)
        if check_milestone_completion(milestone_id, info):
            print("  ⚠️  WARNING: Milestone already completed at start!")

        episode_length = 0
        done = False
        success = False
        frames = [obs.copy()]  # Always record for failure case

        while not done and episode_length < max_steps:
            # Query expert for action
            expert_action = expert.get_action(obs, info)

            # Log first few actions
            if episode_length < 3:
                action_name = ["a", "b", "start", "select", "up", "down", "left", "right", "l", "r"][expert_action]
                print(f"  Step {episode_length}: expert chose action={expert_action} ({action_name})")

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(expert_action)
            done = terminated or truncated

            # Check if all Pokemon fainted
            if check_party_fainted(info):
                print(f"  ❌ All Pokemon fainted at step {episode_length}")
                done = True
                reward -= 50.0  # Penalty for party wipeout

            # Check milestone completion
            if check_milestone_completion(milestone_id, info):
                success = True
                done = True
                reward += 100.0

            # Store in buffer
            buffer.add(
                obs=preprocess_observation(obs),
                next_obs=preprocess_observation(next_obs),
                action=expert_action,
                reward=reward,
                done=done,
            )

            # Record frame
            frames.append(next_obs.copy())
            obs = next_obs
            episode_length += 1

        # Check if expert failed
        if not success:
            print(f"\n❌ EXPERT FAILED at episode {ep + 1}/{n_episodes}!")
            print(f"   Episode length: {episode_length}/{max_steps}")
            print("   This means the expert policy is not working correctly.")

            # Save failure video
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"expert_failure_ep{ep + 1}.mp4")
            saved_path = save_frames_as_video(frames, video_path, fps=30)

            if saved_path:
                print(f"\n📹 Failure video saved: {saved_path}")

            print("\nExiting program - please fix expert policy before training.")
            sys.exit(1)

        print(f"  ✓ SUCCESS | Length={episode_length}")

    print("\n✓ Expert demonstration collection complete!")
    print(f"  Success Rate: 100% ({n_episodes}/{n_episodes})")
    print(f"  Buffer Size: {buffer.size():,} transitions\n")


def rollout_with_expert(
    env: PokemonEnv,
    policy_net: Any,  # torch.nn.Module (imported later)
    expert: ExpertPolicyOracle,
    buffer: Any,  # ReplayBuffer (imported later)
    device: Any,  # torch.device (imported later)
    n_episodes: int,
    max_steps: int,
    milestone_id: str,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    iteration: int = 0,
    max_videos: int = 2,
) -> dict:
    """
    Rollout episodes using student policy, query expert for labels

    DAgger algorithm:
    1. Execute student policy
    2. For each state, query expert for optimal action
    3. Store (state, expert_action) in dataset
    4. Train student to match expert

    Args:
        env: PokemonEnv
        policy_net: Student policy
        expert: Expert oracle (successful policy)
        buffer: ReplayBuffer to store (obs, expert_action) pairs
        device: torch device
        n_episodes: Number of episodes to rollout
        max_steps: Max steps per episode
        milestone_id: Milestone ID for completion check

    Returns:
        stats: Dict with rollout statistics
    """
    policy_net.eval()

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    expert_queries = 0
    episode_videos = []  # Store video paths

    for ep in range(n_episodes):
        print(f"\n  Episode {ep + 1}/{n_episodes}")
        print(f"    Resetting environment...")
        obs, info = env.reset()
        print(f"    Environment reset complete")
        episode_return = 0
        episode_length = 0
        done = False
        success = False

        # Video recording
        should_record = record_video and video_dir and ep < max_videos
        frames = [] if should_record else None

        location = info.get("player", {}).get("location", "unknown")
        print(f"    Initial location: {location}")
        print(f"    Starting rollout (max {max_steps} steps)...")

        while not done and episode_length < max_steps:
            # Record frame for video
            if should_record:
                frames.append(obs.copy())

            # Get student action
            obs_preprocessed = preprocess_observation(obs)
            obs_tensor = torch.FloatTensor(obs_preprocessed).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy_net(obs_tensor)
                student_action = logits.argmax(dim=-1).item()

            # Query expert for label
            if episode_length == 0:
                print(f"    Querying expert for first action...")
            expert_queries += 1
            expert_action = expert.get_action(obs, info)
            if episode_length == 0:
                print(f"    Expert returned action: {expert_action}")

            # Execute student action (DAgger: visit student distribution)
            next_obs, reward, terminated, truncated, info = env.step(student_action)
            done = terminated or truncated

            # Check if all Pokemon fainted
            if check_party_fainted(info):
                if episode_length % 10 == 0 or episode_length < 3:
                    print(f"    ❌ All Pokemon fainted at step {episode_length}")
                done = True
                reward -= 50.0  # Penalty for party wipeout

            # Check milestone completion
            if check_milestone_completion(milestone_id, info):
                success = True
                done = True
                reward += 100.0  # Bonus for completion

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

            # Progress log every 10 steps
            if episode_length % 10 == 0:
                location = info.get("player", {}).get("location", "unknown")
                print(f"    Step {episode_length}/{max_steps}: location={location}, reward={reward:.1f}")

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(1.0 if success else 0.0)

        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"    {status} | Return={episode_return:.1f} | Length={episode_length}")

        # Save video if recorded
        if should_record and frames:
            video_filename = f"rollout_iter{iteration}_ep{ep}_{milestone_id}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            saved_path = save_frames_as_video(frames, video_path, fps=30)
            if saved_path:
                episode_videos.append(saved_path)
                print(f"    📹 Video saved: {video_filename}")

    policy_net.train()

    stats = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(episode_successes),
        "expert_queries": expert_queries,
        "videos": episode_videos,
    }

    return stats


def evaluate_policy(
    env: PokemonEnv,
    policy_net: Any,  # torch.nn.Module (imported later)
    device: Any,  # torch.device (imported later)
    n_episodes: int,
    max_steps: int,
    milestone_id: str,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    iteration: int = 0,
    max_videos: int = 2,
) -> dict:
    """
    Evaluate policy on environment

    Args:
        env: PokemonEnv instance
        policy_net: Policy network
        device: torch device
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        milestone_id: Milestone ID for completion check

    Returns:
        stats: Dict with evaluation statistics
    """
    policy_net.eval()

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episode_videos = []  # Store video paths
    print("[doing eval for ]", n_episodes, "episodes")
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        success = False

        # Video recording
        should_record = record_video and video_dir and ep < max_videos
        frames = [] if should_record else None

        while not done and episode_length < max_steps:
            # Record frame for video
            if should_record:
                frames.append(obs.copy())
            # Preprocess and get action
            obs_preprocessed = preprocess_observation(obs)
            obs_tensor = torch.FloatTensor(obs_preprocessed).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy_net(obs_tensor)
                action = logits.argmax(dim=-1).item()  # Deterministic

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Check if all Pokemon fainted
            if check_party_fainted(info):
                if episode_length % 20 == 0:
                    print(f"      ❌ All Pokemon fainted at step {episode_length}")
                done = True
                reward -= 50.0  # Penalty for party wipeout

            # Check milestone completion
            if check_milestone_completion(milestone_id, info):
                success = True
                done = True
                reward += 100.0

            episode_return += reward
            episode_length += 1

            # Progress log every 20 steps for eval
            if episode_length % 20 == 0:
                location = info.get("player", {}).get("location", "unknown")
                print(f"      [Eval] Step {episode_length}/{max_steps}: location={location}")

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(1.0 if success else 0.0)

        # Save video if recorded
        if should_record and frames:
            video_filename = f"eval_iter{iteration}_ep{ep}_{milestone_id}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            saved_path = save_frames_as_video(frames, video_path, fps=30)
            if saved_path:
                episode_videos.append(saved_path)
                print(f"      📹 Video saved: {video_filename}")

    policy_net.train()

    stats = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(episode_successes),
        "videos": episode_videos,
    }

    return stats


def evaluate_expert_policy(
    env: PokemonEnv,
    expert: ExpertPolicyOracle,
    n_episodes: int,
    max_steps: int,
    milestone_id: str,
    record_video: bool = False,
    video_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate expert policy only (no student policy)

    Args:
        env: PokemonEnv instance
        expert: Expert oracle
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        milestone_id: Milestone ID for completion check
        record_video: Whether to record videos
        video_dir: Directory to save videos

    Returns:
        stats: Dict with evaluation statistics including per-episode results
    """
    print("\n" + "=" * 70)
    print(f"Expert Policy Evaluation on {milestone_id}")
    print("=" * 70 + "\n")

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episode_details = []  # Store per-episode details

    for ep in range(n_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"{'=' * 60}")

        obs, info = env.reset()

        # Print initial state
        player = info.get("player", {})
        location = player.get("location", "UNKNOWN")
        position = player.get("position", {})
        x, y = position.get("x", -1), position.get("y", -1)
        print(f"Starting location: {location}")
        print(f"Starting position: ({x}, {y})")

        episode_return = 0
        episode_length = 0
        done = False
        success = False

        # Video recording
        should_record = record_video and video_dir
        frames = [obs.copy()] if should_record else None

        while not done and episode_length < max_steps:
            # Get expert action
            expert_action = expert.get_action(obs, info)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(expert_action)
            done = terminated or truncated

            # Record frame
            if should_record:
                frames.append(obs.copy())

            # Check if all Pokemon fainted
            if check_party_fainted(info):
                print(f"  ❌ All Pokemon fainted at step {episode_length}")
                done = True
                reward -= 50.0  # Penalty for party wipeout

            # Check milestone completion
            if check_milestone_completion(milestone_id, info):
                success = True
                done = True
                reward += 100.0

            episode_return += reward
            episode_length += 1

            # Progress log every 10 steps
            if episode_length % 10 == 0:
                location = info.get("player", {}).get("location", "unknown")
                position = info.get("player", {}).get("position", {})
                x, y = position.get("x", -1), position.get("y", -1)
                print(f"  Step {episode_length:3d}/{max_steps}: location={location}, pos=({x},{y})")

        # Store results
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(1.0 if success else 0.0)
        episode_details.append(
            {
                "episode": ep + 1,
                "success": success,
                "steps": episode_length,
                "return": episode_return,
            }
        )

        # Print episode result
        status_emoji = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"
        print(f"\n{status_emoji} Episode {ep + 1}: {status_text}")
        print(f"   Steps taken: {episode_length}/{max_steps}")
        print(f"   Total return: {episode_return:.2f}")

        # Save video if recorded
        if should_record and frames:
            os.makedirs(video_dir, exist_ok=True)
            status_suffix = "success" if success else "failed"
            video_filename = f"expert_eval_ep{ep + 1}_{status_suffix}_steps{episode_length}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            saved_path = save_frames_as_video(frames, video_path, fps=30)
            if saved_path:
                print(f"   📹 Video saved: {video_filename}")

    # Print summary
    print("\n" + "=" * 70)
    print("Expert Policy Evaluation Summary")
    print("=" * 70 + "\n")

    success_rate = np.mean(episode_successes)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    mean_return = np.mean(episode_returns)

    print(f"Overall Results:")
    print(f"  Success Rate: {success_rate:.1%} ({int(sum(episode_successes))}/{n_episodes})")
    print(f"  Mean Steps: {mean_length:.1f} ± {std_length:.1f}")
    print(f"  Mean Return: {mean_return:.2f}")

    # Print per-episode table
    print(f"\nPer-Episode Results:")
    print(f"  {'Episode':<10} {'Status':<10} {'Steps':<10} {'Return':<10}")
    print(f"  {'-' * 40}")
    for detail in episode_details:
        status = "SUCCESS" if detail["success"] else "FAILED"
        emoji = "✅" if detail["success"] else "❌"
        print(f"  {emoji} {detail['episode']:<8} {status:<10} {detail['steps']:<10} {detail['return']:<10.2f}")

    # Calculate success-only stats
    successful_episodes = [d for d in episode_details if d["success"]]
    if successful_episodes:
        success_steps = [d["steps"] for d in successful_episodes]
        print(f"\nSuccessful Episodes Only:")
        print(f"  Mean Steps: {np.mean(success_steps):.1f} ± {np.std(success_steps):.1f}")
        print(f"  Min Steps: {min(success_steps)}")
        print(f"  Max Steps: {max(success_steps)}")

    print("\n" + "=" * 70 + "\n")

    stats = {
        "mean_return": mean_return,
        "std_return": np.std(episode_returns),
        "mean_length": mean_length,
        "std_length": std_length,
        "success_rate": success_rate,
        "num_episodes": n_episodes,
        "num_success": int(sum(episode_successes)),
        "episode_details": episode_details,
    }

    return stats


def train(args: Args):
    """Main DAgger training function"""

    # Setup
    run_name = f"{args.exp_name}_{args.milestone}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    # Create video directory if recording
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"✓ Video directory created: {args.video_dir}")

    # Create environments BEFORE CUDA initialization to avoid mgba-CUDA deadlock
    print("\nCreating environments...")

    def make_milestone_env(headless: bool, base_fps: int):
        """Create environment that starts from milestone state"""
        env = PokemonEnv(
            rom_path=args.rom_path,
            headless=headless,
            base_fps=base_fps,
            enable_milestones=True,
        )

        # Wrap reset to always load milestone starting state
        original_reset = env.reset

        def reset_to_milestone(**kwargs):
            return original_reset(options={"load_state": args.starting_state})

        env.reset = reset_to_milestone
        return env

    # Use single environment for both training and evaluation
    # (Creating multiple envs causes ROM access conflicts)
    env = make_milestone_env(headless=True, base_fps=240)
    print(f"✓ Environment created (starting from {args.starting_state})")

    # IMPORTANT: Initialize emulator BEFORE PyTorch by calling reset once
    # This avoids mgba-PyTorch/CUDA deadlock in load_raw_state
    print("Initializing emulator (before PyTorch)...")
    _, _ = env.reset()
    print("✓ Emulator initialized")

    # NOW import PyTorch and related modules (AFTER emulator initialization)
    # Import into global scope so all functions can use them (no duplicate imports)
    print("Importing PyTorch...")
    global torch, F, optim, ImpalaCNN, ReplayBuffer, preprocess_observation
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from rl_training.common.networks import ImpalaCNN
    from rl_training.common.buffers import ReplayBuffer
    from rl_training.common.utils import preprocess_observation

    print("✓ PyTorch imported")

    # Set random seeds (after PyTorch import)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create policy network
    policy_net = ImpalaCNN(action_dim=10, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"\n✓ Policy network: {sum(p.numel() for p in policy_net.parameters()):,} parameters")

    # Create replay buffer (use preprocessed observation size)
    from gymnasium import spaces

    preprocessed_obs_space = spaces.Box(
        low=0,
        high=1,
        shape=(3, 84, 84),  # Preprocessed size
        dtype=np.float32,
    )
    buffer = ReplayBuffer(
        buffer_size=50_000,  # Sufficient for milestone training (reduced from 100k)
        observation_space=preprocessed_obs_space,
        action_space=env.action_space,
        device=device,
    )

    # Load tools and create expert oracle
    print(f"\nLoading expert policy...")
    tools = load_tools("tools")
    expert = ExpertPolicyOracle(args.expert_policy_path, tools, verbose=args.expert_verbose)

    # If eval_expert_only mode, run expert evaluation and exit
    if args.eval_expert_only:
        print("\n" + "=" * 70)
        print("EXPERT EVALUATION MODE")
        print("=" * 70)
        print("Running expert policy evaluation only (no training)\n")

        stats = evaluate_expert_policy(
            env=env,
            expert=expert,
            n_episodes=args.eval_expert_episodes,
            max_steps=args.max_episode_steps,
            milestone_id=args.milestone,
            record_video=args.record_video,
            video_dir=args.video_dir if args.record_video else None,
        )

        env.close()
        print("Expert evaluation complete. Exiting.")
        return stats  # Return stats for pipeline usage

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n{'=' * 70}")
        print(f"Resuming from checkpoint: {args.resume_from}")
        print(f"{'=' * 70}\n")

        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        # Check checkpoint format (full checkpoint vs state_dict only)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Full checkpoint format (includes optimizer, iteration, etc.)
            policy_net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load training state
            start_iteration = checkpoint.get("iteration", 0) + 1  # Start from next iteration
            global_step = checkpoint.get("global_step", 0)
            best_success_rate = checkpoint.get("best_success_rate", 0.0)

            print(f"✓ Loaded full checkpoint:")
            print(f"  Iteration: {checkpoint.get('iteration', 0)}")
            print(f"  Global step: {global_step}")
            print(f"  Best success rate: {best_success_rate:.1%}")

            # Load buffer if exists
            buffer_path = args.resume_from.replace(".pth", "_buffer.npz")
            if os.path.exists(buffer_path):
                buffer.load(buffer_path)
                print(f"✓ Loaded buffer: {buffer.size():,} transitions")
            else:
                print(f"⚠️  Warning: Buffer checkpoint not found at {buffer_path}")
                print(f"   Starting with empty buffer")

        else:
            # state_dict only format (e.g., final model for inference)
            policy_net.load_state_dict(checkpoint)
            print(f"✓ Loaded model weights (state_dict only)")
            print(f"⚠️  Warning: No training state found, starting from iteration 0")

        print()
    else:
        if args.resume_from:
            print(f"⚠️  Warning: Checkpoint not found at {args.resume_from}")
            print(f"   Starting training from scratch\n")

    # Validate skip_collection flag
    if args.skip_collection and not args.resume_from:
        raise ValueError("--skip-collection requires --resume-from to load existing buffer")
    if args.skip_collection and buffer.size() == 0:
        raise ValueError("--skip-collection requires non-empty buffer from checkpoint")

    # Phase 0: Collect expert demonstrations (validates expert policy)
    # Skip if resuming from checkpoint with existing buffer
    if args.pre_collect_episodes > 0 and (not args.resume_from or buffer.size() == 0):
        collect_expert_demonstrations(
            env=env,
            expert=expert,
            buffer=buffer,
            n_episodes=args.pre_collect_episodes,
            max_steps=args.max_episode_steps,
            milestone_id=args.milestone,
            video_dir=args.video_dir,
        )

    # DAgger iterations
    print("\n" + "=" * 70)
    print(f"Starting DAgger training on {args.milestone}")
    if start_iteration > 0:
        print(f"Resuming from iteration {start_iteration}")
    print("=" * 70 + "\n")

    # Initialize best_success_rate and global_step if not resuming
    if not args.resume_from:
        best_success_rate = 0.0
        global_step = 0

    # Track iteration-level metrics for monitoring
    iteration_metrics = []

    for iteration in range(start_iteration, args.n_iterations):
        iter_start_time = time.time()

        print(f"\n{'=' * 70}")
        print(f"DAgger Iteration {iteration + 1}/{args.n_iterations}")
        print(f"{'=' * 70}\n")

        # Phase 1: Rollout with current policy, query expert for labels
        # Skip rollout in iteration 0 (pure imitation learning on expert data)
        # Or skip if --skip-collection flag is set (train on existing buffer only)
        if iteration == 0 or args.skip_collection:
            if args.skip_collection:
                print(f"Phase 1: SKIPPED (--skip-collection: training on existing buffer only)")
            else:
                print(f"Phase 1: SKIPPED (Iteration 0 - Pure imitation learning on expert data)")
            print("-" * 70)
            print(f"  Buffer Size: {buffer.size():,}")
        else:
            print(f"Phase 1: Rollout ({args.episodes_per_iter} episodes)")
            print("-" * 70)

            rollout_stats = rollout_with_expert(
                env=env,
                policy_net=policy_net,
                expert=expert,
                buffer=buffer,
                device=device,
                n_episodes=args.episodes_per_iter,
                max_steps=args.max_episode_steps,
                milestone_id=args.milestone,
                record_video=args.record_rollout_videos and args.record_video,
                video_dir=args.video_dir if args.record_video else None,
                iteration=iteration,
                max_videos=args.max_videos_per_iter,
            )

            print(f"\n✓ Rollout complete:")
            print(f"  Success Rate: {rollout_stats['success_rate']:.1%}")
            print(f"  Mean Return: {rollout_stats['mean_return']:.2f}±{rollout_stats['std_return']:.2f}")
            print(f"  Mean Length: {rollout_stats['mean_length']:.1f}")
            print(f"  Buffer Size: {buffer.size():,}")

            # Log rollout stats
            if args.track:
                import wandb

                log_dict = {
                    "rollout/success_rate": rollout_stats["success_rate"],
                    "rollout/mean_return": rollout_stats["mean_return"],
                    "rollout/mean_length": rollout_stats["mean_length"],
                    "rollout/buffer_size": buffer.size(),
                    "iteration": iteration,
                }

                # Add videos to wandb
                if rollout_stats.get("videos"):
                    for i, video_path in enumerate(rollout_stats["videos"]):
                        log_dict[f"rollout/video_ep{i}"] = wandb.Video(video_path, fps=30, format="mp4")

                wandb.log(log_dict)

        # Phase 2: Train on aggregated dataset
        print(f"\nPhase 2: Training ({args.train_steps_per_iter} steps)")
        print("-" * 70)

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
            if train_step % 100 == 0 and args.track:
                import wandb

                wandb.log(
                    {
                        "train/bc_loss": loss.item(),
                        "train/accuracy": accuracy.item(),
                        "global_step": global_step,
                    }
                )

            if train_step % 500 == 0:
                print(
                    f"  Step {train_step:5d}/{args.train_steps_per_iter}: "
                    f"Loss={loss.item():.4f}, Acc={accuracy.item():.2%}"
                )

        print(f"\n✓ Training complete:")
        print(f"  Mean Loss: {np.mean(train_losses):.4f}")
        mean_train_acc = np.mean(train_accs)
        print(f"  Mean Accuracy: {mean_train_acc:.2%}")

        # Track iteration metrics
        iter_metric = {
            "iteration": iteration,
            "train_accuracy": mean_train_acc,
            "train_loss": np.mean(train_losses),
        }

        # Phase 3: Evaluate
        if (iteration + 1) % args.eval_frequency == 0:
            print(f"\nPhase 3: Evaluation ({args.eval_episodes} episodes)")
            print("-" * 70)

            eval_stats = evaluate_policy(
                env=env,
                policy_net=policy_net,
                device=device,
                n_episodes=args.eval_episodes,
                max_steps=args.max_episode_steps,
                milestone_id=args.milestone,
                record_video=args.record_video,
                video_dir=args.video_dir if args.record_video else None,
                iteration=iteration,
                max_videos=args.max_videos_per_iter,
            )

            print(f"\n✓ Evaluation results:")
            print(f"  Success Rate: {eval_stats['success_rate']:.1%}")
            print(f"  Mean Return: {eval_stats['mean_return']:.2f}±{eval_stats['std_return']:.2f}")
            print(f"  Mean Length: {eval_stats['mean_length']:.1f}")

            # Add eval metrics to iteration metric
            iter_metric.update(
                {
                    "eval_success_rate": eval_stats["success_rate"],
                    "eval_mean_return": eval_stats["mean_return"],
                    "eval_mean_length": eval_stats["mean_length"],
                }
            )

            if args.track:
                import wandb

                log_dict = {
                    "eval/success_rate": eval_stats["success_rate"],
                    "eval/mean_return": eval_stats["mean_return"],
                    "eval/mean_length": eval_stats["mean_length"],
                    "iteration": iteration,
                }

                # Add videos to wandb
                if eval_stats.get("videos"):
                    for i, video_path in enumerate(eval_stats["videos"]):
                        log_dict[f"eval/video_ep{i}"] = wandb.Video(video_path, fps=30, format="mp4")

                wandb.log(log_dict)

            # Save best model
            if args.save_best and eval_stats["success_rate"] > best_success_rate:
                best_success_rate = eval_stats["success_rate"]
                save_path = os.path.join(args.model_dir, f"{args.exp_name}_{args.milestone}_best.pth")
                torch.save(
                    {
                        "iteration": iteration,
                        "global_step": global_step,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_success_rate": best_success_rate,
                        "success_rate": eval_stats["success_rate"],
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"\n  💾 Saved best model: {save_path} (success={eval_stats['success_rate']:.1%})")

                # Save buffer for best model
                buffer_path = save_path.replace(".pth", "_buffer.npz")
                buffer.save(buffer_path)
                print(f"  💾 Saved buffer: {buffer_path}")

        # Save checkpoint (every iteration)
        save_path = os.path.join(args.model_dir, f"{args.exp_name}_{args.milestone}_iter{iteration + 1}.pth")
        torch.save(
            {
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_success_rate": best_success_rate,
            },
            save_path,
        )

        # Save buffer checkpoint
        buffer_path = save_path.replace(".pth", "_buffer.npz")
        buffer.save(buffer_path)
        print("  💾 Saved checkpoint and buffer")

        iter_time = time.time() - iter_start_time
        print(f"\n✓ Iteration {iteration + 1} completed in {iter_time:.1f}s")

        # Save iteration metrics
        iteration_metrics.append(iter_metric)

    # Final save
    final_path = os.path.join(args.model_dir, f"{args.exp_name}_{args.milestone}_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\n{'=' * 70}")
    print(f"✓ DAgger training complete!")
    print(f"  Final model saved to {final_path}")
    print(f"  Best success rate: {best_success_rate:.1%}")
    print(f"{'=' * 70}\n")

    # Use last iteration's eval results as final results
    wandb_url = None
    if args.track:
        import wandb
        wandb_url = wandb.run.url if wandb.run else None

    env.close()

    # Extract final stats from last iteration metrics (if available)
    final_success_rate = 0.0
    final_mean_return = 0.0
    final_mean_length = 0.0

    if iteration_metrics and "eval_success_rate" in iteration_metrics[-1]:
        last_eval = iteration_metrics[-1]
        final_success_rate = last_eval.get("eval_success_rate", 0.0)
        final_mean_return = last_eval.get("eval_mean_return", 0.0)
        final_mean_length = last_eval.get("eval_mean_length", 0.0)

        print("\nFinal Results (from last iteration eval):")
        print(f"  Success Rate: {final_success_rate:.1%}")
        print(f"  Mean Return: {final_mean_return:.2f}")
        print(f"  Mean Length: {final_mean_length:.1f}")

    return {
        "wandb_url": wandb_url,
        "final_success_rate": final_success_rate,
        "final_mean_return": final_mean_return,
        "final_mean_length": final_mean_length,
        "final_train_accuracy": iteration_metrics[-1].get("train_accuracy") if iteration_metrics else None,
        "iteration_metrics": iteration_metrics,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
