#!/usr/bin/env python3
"""
Auto DAgger Training Pipeline

Automatically trains DAgger policies for all milestones:
1. Evaluate expert policy to get average steps
2. Set max_episode_steps = mean_steps × multiplier
3. Train DAgger policy with computed max_steps

Usage:
    python auto_dagger_pipeline.py \
        --milestone-config ../milestone_config.json \
        --eval-episodes 5 \
        --step-multiplier 2.0 \
        --dagger-iterations 10
"""

import os
import sys
import json
import time
import csv
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

import tyro

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PipelineArgs:
    # Pipeline Config
    milestone_config: str = "milestone_config.json"
    """Path to milestone config JSON"""

    # Expert Evaluation
    eval_episodes: int = 1
    """Number of episodes to evaluate expert (to estimate steps)"""
    step_multiplier: float = 2.0
    """Multiply mean expert steps by this to get max_episode_steps for DAgger"""
    min_success_rate: float = 0.6
    """Minimum RIVAL_HOUSE success rate to proceed with training (0.0-1.0)"""

    # DAgger Training
    dagger_iterations: int = 6
    """Number of DAgger iterations per milestone"""
    episodes_per_iter: int = 1
    """Rollout episodes per DAgger iteration"""
    train_steps_per_iter: int = 5000
    """Training steps per DAgger iteration"""

    # Paths
    state_dir: str = ".milestone_trainer_cache/milestone_states"
    # state_dir: str = ".expert_snapshots_20251115_232912"
    """Directory containing milestone state files"""
    policy_dir: str = ".milestone_trainer_cache/successful_policies"
    """Directory containing expert policy files"""
    rom_path: str = "Emerald-GBAdvance/rom.gba"
    """Path to Pokemon ROM"""

    # Output
    output_dir: str = "dagger_pipeline_results"
    """Directory to save training results and logs"""

    # Training Settings
    batch_size: int = 256
    """Batch size for DAgger training"""
    learning_rate: float = 1e-4
    """Learning rate"""
    hidden_dim: int = 1024
    """Hidden dimension for policy network"""

    # Resume
    resume_from: Optional[str] = None
    """Resume from milestone ID (skip completed milestones)"""
    skip_milestones: List[str] = None
    """List of milestone IDs to skip"""
    milestone: Optional[str] = None
    """Train only this specific milestone (if specified, ignores resume_from and skip_milestones)"""
    no_auto_resume: bool = False
    """Disable automatic checkpoint loading (always start fresh training)"""

    # Other
    track: bool = False
    """Track with Weights & Biases"""
    wandb_project: str = "pokemon-dagger-pipeline"
    """W&B project name"""
    record_video: bool = True
    """Record videos during evaluation"""
    cuda: bool = True
    """Use CUDA if available"""


class MilestoneTrainingPipeline:
    """
    Pipeline to automatically train DAgger policies for all milestones
    """

    def __init__(self, args: PipelineArgs):
        self.args = args
        self.results = []

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Results file
        self.results_csv = os.path.join(args.output_dir, "pipeline_results.csv")
        self.results_json = os.path.join(args.output_dir, "pipeline_results.json")
        self.state_file = os.path.join(args.output_dir, "pipeline_state.json")

        # Milestone details directory
        self.milestone_details_dir = os.path.join(args.output_dir, "milestone_details")
        os.makedirs(self.milestone_details_dir, exist_ok=True)

        # Load milestone config
        print(f"Loading milestone config from {args.milestone_config}")
        with open(args.milestone_config, "r") as f:
            self.config = json.load(f)

        self.milestones = self.config["milestones"]
        print(f"✓ Loaded {len(self.milestones)} milestones")

    def get_state_path(self, starting_state: Optional[str]) -> Optional[str]:
        """
        Get full path to starting state file

        Args:
            starting_state: Either None, or path like "milestone_states/XXX.state" or "XXX_completed.state"

        Returns:
            Full path to state file, or None if starting_state is None
        """
        if not starting_state:
            return None

        # If already absolute path, return as-is
        if os.path.isabs(starting_state):
            return starting_state

        # If starts with milestone_states/, it's a relative path from project root
        if starting_state.startswith("milestone_states/"):
            # Extract filename
            filename = os.path.basename(starting_state)
            return os.path.join(self.args.state_dir, filename)

        # Otherwise assume it's just a filename in state_dir
        return os.path.join(self.args.state_dir, starting_state)

    def check_files_exist(self, milestone_id: str, starting_state: Optional[str]) -> tuple[bool, str]:
        """
        Check if required files exist for a milestone

        Returns:
            (exists, message)
        """
        # Check expert policy
        policy_path = os.path.join(self.args.policy_dir, f"{milestone_id}.py")
        if not os.path.exists(policy_path):
            return False, f"Expert policy not found: {policy_path}"

        # Check starting state (if not first milestone)
        if starting_state:
            state_path = self.get_state_path(starting_state)
            if not os.path.exists(state_path):
                return False, f"Starting state not found: {state_path}"

        return True, "All files exist"

    def update_pipeline_state(self, current_milestone: Optional[str] = None, current_phase: Optional[str] = None):
        """
        Update pipeline state JSON file

        Args:
            current_milestone: Current milestone ID
            current_phase: Current phase (expert_eval | dagger_train | completed)
        """
        completed = sum(1 for r in self.results if r.get("status") == "completed")
        failed = sum(1 for r in self.results if r.get("status") == "failed")
        skipped = sum(1 for r in self.results if r.get("status") == "skipped")
        in_progress = 1 if current_milestone and current_phase != "completed" else 0
        pending = len(self.milestones) - completed - failed - skipped - in_progress

        state = {
            "current_milestone": current_milestone,
            "current_phase": current_phase,
            "total": len(self.milestones),
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "pending": pending,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def save_milestone_detail(self, milestone_id: str, milestone_info: Dict):
        """
        Save detailed milestone information to JSON

        Args:
            milestone_id: Milestone ID
            milestone_info: Dict containing all milestone information
        """
        detail_file = os.path.join(self.milestone_details_dir, f"{milestone_id}.json")
        with open(detail_file, "w") as f:
            json.dump(milestone_info, f, indent=2)

    def get_best_video(self, video_dir: str, milestone_id: str, video_type: str = "expert") -> Optional[str]:
        """
        Find the best video in a directory

        Args:
            video_dir: Directory containing videos
            milestone_id: Milestone ID
            video_type: "expert" or "dagger"

        Returns:
            Video filename (not full path), or None if no videos found
        """
        if not os.path.exists(video_dir):
            return None

        if video_type == "expert":
            # Expert videos: prioritize "success" keyword and shortest steps
            success_videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4") and "success" in f]

            # Parse steps from filename (e.g., "expert_eval_ep1_success_steps42.mp4")
            def get_steps(filename):
                try:
                    parts = filename.split("steps")
                    if len(parts) > 1:
                        return int(parts[1].split(".")[0])
                except:
                    pass
                return 999999

            if success_videos:
                # Return successful video with minimum steps
                return min(success_videos, key=get_steps)

            # If no success videos, return any video (e.g., failed videos for debugging)
            all_videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
            if all_videos:
                # Return latest video (highest episode number)
                def get_episode(filename):
                    try:
                        parts = filename.split("_ep")
                        if len(parts) > 1:
                            return int(parts[1].split("_")[0])
                    except:
                        pass
                    return -1

                return max(all_videos, key=get_episode)

            return None

        elif video_type == "dagger":
            # DAgger videos: get latest eval video (highest iteration)
            eval_videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4") and "eval_iter" in f]
            if not eval_videos:
                return None

            # Parse iteration number from filename (e.g., "eval_iter2_ep0_RIVAL_HOUSE.mp4")
            def get_iteration(filename):
                try:
                    parts = filename.split("eval_iter")
                    if len(parts) > 1:
                        return int(parts[1].split("_")[0])
                except:
                    pass
                return -1

            # Return video from highest iteration
            return max(eval_videos, key=get_iteration)

        return None

    def evaluate_expert(self, milestone_id: str, starting_state: Optional[str]) -> Dict:
        """
        Evaluate expert policy to get average steps

        Returns:
            dict with keys: success_rate, mean_steps, std_steps, etc.
        """
        print(f"\n{'=' * 70}")
        print(f"Evaluating Expert Policy: {milestone_id}")
        print(f"{'=' * 70}\n")

        policy_path = os.path.join(self.args.policy_dir, f"{milestone_id}.py")
        state_path = self.get_state_path(starting_state)

        # Import here to avoid circular dependency
        from dagger_pokemon_milestone import Args as DAggerArgs, train

        # Clean up old expert evaluation videos for this milestone to avoid stale data
        expert_video_dir = os.path.join(self.args.output_dir, "expert_eval_videos", milestone_id)
        if os.path.exists(expert_video_dir):
            import shutil

            print(f"Cleaning old expert evaluation videos from {expert_video_dir}")
            shutil.rmtree(expert_video_dir)
        os.makedirs(expert_video_dir, exist_ok=True)

        # Create DAgger args for expert evaluation
        eval_args = DAggerArgs(
            exp_name=f"expert_eval_{milestone_id}",
            milestone=milestone_id,
            starting_state=state_path if state_path else "",
            expert_policy_path=policy_path,
            max_episode_steps=450,  # Generous limit for evaluation
            eval_expert_only=True,
            eval_expert_episodes=self.args.eval_episodes,
            record_video=self.args.record_video,
            video_dir=expert_video_dir,
            expert_verbose=False,
            cuda=self.args.cuda,
        )

        # Run evaluation (this will call evaluate_expert_policy and return stats)
        try:
            stats = train(eval_args)

            if stats is None:
                return {
                    "success": False,
                    "error": "No stats returned from evaluation",
                }

            # Extract stats from returned dict
            success_rate = stats.get("success_rate", 0.0)
            mean_length = stats.get("mean_length", 0.0)
            std_length = stats.get("std_length", 0.0)

            # Calculate mean steps from successful episodes only
            episode_details = stats.get("episode_details", [])
            successful_episodes = [d for d in episode_details if d["success"]]

            if successful_episodes:
                success_steps = [d["steps"] for d in successful_episodes]
                mean_success_steps = sum(success_steps) / len(success_steps)
                std_success_steps = (
                    sum((s - mean_success_steps) ** 2 for s in success_steps) / len(success_steps)
                ) ** 0.5
            else:
                mean_success_steps = mean_length
                std_success_steps = std_length

            return {
                "success": True,
                "success_rate": success_rate,
                "mean_steps": mean_success_steps,
                "std_steps": std_success_steps,
                "all_episodes_mean": mean_length,
                "num_episodes": len(episode_details),
                "num_success": len(successful_episodes),
            }

        except Exception as e:
            print(f"❌ Expert evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
            }

    def find_latest_checkpoint(self, model_dir: str, milestone_id: str) -> str | None:
        """
        Find the latest checkpoint for a milestone

        Args:
            model_dir: Directory containing model checkpoints
            milestone_id: Milestone ID

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        if not os.path.exists(model_dir):
            return None

        # Look for best model first (highest priority)
        best_model = os.path.join(model_dir, f"dagger_{milestone_id}_{milestone_id}_best.pth")
        if os.path.exists(best_model):
            return best_model

        # Otherwise, find highest iteration checkpoint
        import re

        iter_pattern = re.compile(rf"dagger_{milestone_id}_{milestone_id}_iter(\d+)\.pth")

        max_iter = -1
        latest_checkpoint = None

        for filename in os.listdir(model_dir):
            match = iter_pattern.match(filename)
            if match:
                iter_num = int(match.group(1))
                if iter_num > max_iter:
                    max_iter = iter_num
                    latest_checkpoint = os.path.join(model_dir, filename)

        return latest_checkpoint

    def train_dagger(self, milestone_id: str, starting_state: Optional[str], max_episode_steps: int) -> Dict:
        """
        Train DAgger policy for a milestone

        Returns:
            dict with training results
        """
        print(f"\n{'=' * 70}")
        print(f"Training DAgger Policy: {milestone_id}")
        print(f"Max Episode Steps: {max_episode_steps}")
        print(f"{'=' * 70}\n")

        policy_path = os.path.join(self.args.policy_dir, f"{milestone_id}.py")
        state_path = self.get_state_path(starting_state)

        # Import here
        from dagger_pokemon_milestone import Args as DAggerArgs, train

        # Auto-detect previous checkpoint for resume (unless disabled)
        model_dir = os.path.join(self.args.output_dir, "models", milestone_id)
        resume_from = None

        if self.args.no_auto_resume:
            print("🆕 Auto-resume disabled (--no-auto-resume), starting fresh\n")
        else:
            resume_from = self.find_latest_checkpoint(model_dir, milestone_id)

            if resume_from:
                print(f"🔄 Found existing checkpoint: {os.path.basename(resume_from)}")
                print("   Will resume training from this checkpoint\n")
            else:
                print("🆕 No existing checkpoint found, starting fresh\n")

        # Clean up old DAgger videos only when starting fresh (not resuming)
        dagger_video_dir = os.path.join(self.args.output_dir, "dagger_videos", milestone_id)
        if resume_from is None:
            if os.path.exists(dagger_video_dir):
                import shutil

                print(f"Cleaning old DAgger videos from {dagger_video_dir}")
                shutil.rmtree(dagger_video_dir)
            os.makedirs(dagger_video_dir, exist_ok=True)
        else:
            # Keep existing videos when resuming
            os.makedirs(dagger_video_dir, exist_ok=True)

        # Create DAgger training args
        train_args = DAggerArgs(
            exp_name=f"dagger_{milestone_id}",
            milestone=milestone_id,
            starting_state=state_path if state_path else "",
            expert_policy_path=policy_path,
            max_episode_steps=max_episode_steps,
            n_iterations=self.args.dagger_iterations,
            episodes_per_iter=self.args.episodes_per_iter,
            train_steps_per_iter=self.args.train_steps_per_iter,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            hidden_dim=self.args.hidden_dim,
            eval_expert_only=False,  # Actually train
            record_video=self.args.record_video,
            video_dir=dagger_video_dir,
            expert_verbose=False,
            track=self.args.track,
            wandb_project_name=self.args.wandb_project,
            model_dir=model_dir,
            cuda=self.args.cuda,
            resume_from=resume_from,  # Auto-resume from latest checkpoint
        )

        # Run training
        try:
            result = train(train_args)

            return {
                "success": True,
                "wandb_url": result.get("wandb_url") if result else None,
                "final_success_rate": result.get("final_success_rate", 0.0) if result else 0.0,
                "final_mean_return": result.get("final_mean_return", 0.0) if result else 0.0,
                "final_mean_length": result.get("final_mean_length", 0.0) if result else 0.0,
            }

        except Exception as e:
            print(f"❌ DAgger training failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
            }

    def run(self):
        """
        Run the full pipeline
        """
        print("\n" + "=" * 70)
        print("Auto DAgger Training Pipeline")
        print("=" * 70 + "\n")

        # Filter milestones if --milestone is specified
        milestones_to_run = self.milestones
        if self.args.milestone:
            milestones_to_run = [m for m in self.milestones if m["id"] == self.args.milestone]
            if not milestones_to_run:
                print(f"❌ Error: Milestone '{self.args.milestone}' not found in config")
                print(f"Available milestones: {', '.join(m['id'] for m in self.milestones)}")
                return
            print(f"🎯 Training single milestone: {self.args.milestone}\n")
        else:
            print(f"Total Milestones: {len(milestones_to_run)}")

        print(f"Eval Episodes: {self.args.eval_episodes}")
        print(f"Step Multiplier: {self.args.step_multiplier}")
        print(f"DAgger Iterations: {self.args.dagger_iterations}")
        print(f"Output Directory: {self.args.output_dir}\n")

        start_time = time.time()

        for i, milestone in enumerate(milestones_to_run):
            milestone_id = milestone["id"]
            starting_state = milestone.get("starting_state")

            print(f"\n{'#' * 70}")
            print(f"# Milestone {i + 1}/{len(milestones_to_run)}: {milestone_id}")
            print(f"# Description: {milestone.get('description', 'N/A')}")
            print(f"{'#' * 70}\n")

            result = {
                "milestone_id": milestone_id,
                "index": i,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Skip filtering if --milestone is specified (already filtered)
            if self.args.milestone:
                pass  # Already filtered to single milestone
            # Check if should skip
            elif self.args.skip_milestones and milestone_id in self.args.skip_milestones:
                print(f"⏭️  Skipping {milestone_id} (in skip list)")
                result["status"] = "skipped"
                result["reason"] = "in skip list"
                self.results.append(result)
                continue
            # Check if should resume
            elif self.args.resume_from and milestone_id != self.args.resume_from:
                print(f"⏭️  Skipping {milestone_id} (waiting for resume point)")
                result["status"] = "skipped"
                result["reason"] = "before resume point"
                self.results.append(result)
                continue
            elif self.args.resume_from and milestone_id == self.args.resume_from:
                print(f"▶️  Resuming from {milestone_id}")
                self.args.resume_from = None  # Clear resume flag

            # Check files exist
            files_exist, message = self.check_files_exist(milestone_id, starting_state)
            if not files_exist:
                print(f"❌ {message}")
                result["status"] = "failed"
                result["reason"] = message
                self.results.append(result)
                self.update_pipeline_state(milestone_id, "failed")
                continue

            # Initialize milestone detail
            milestone_detail = {
                "milestone_id": milestone_id,
                "description": milestone.get("description", ""),
                "status": "in_progress",
                "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_end": None,
            }

            # Step 1: Evaluate expert
            self.update_pipeline_state(milestone_id, "expert_eval")
            eval_result = self.evaluate_expert(milestone_id, starting_state)

            if not eval_result.get("success"):
                print(f"❌ Expert evaluation failed, skipping training")
                result["status"] = "failed"
                result["reason"] = "expert evaluation failed"
                result["error"] = eval_result.get("error", "unknown")
                milestone_detail["status"] = "failed"
                milestone_detail["error"] = eval_result.get("error", "unknown")
                milestone_detail["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.save_milestone_detail(milestone_id, milestone_detail)
                self.results.append(result)
                self.update_pipeline_state(milestone_id, "failed")
                continue

            success_rate = eval_result["success_rate"]
            mean_steps = eval_result["mean_steps"]

            result["expert_success_rate"] = success_rate
            result["expert_mean_steps"] = mean_steps
            result["expert_std_steps"] = eval_result["std_steps"]

            # Get best expert video
            expert_video_dir = os.path.join(self.args.output_dir, "expert_eval_videos", milestone_id)
            best_expert_video = self.get_best_video(expert_video_dir, milestone_id)

            # Find min/max steps from successful episodes
            episode_details = eval_result.get("episode_details", [])
            successful_episodes = [d for d in episode_details if d["success"]]
            if successful_episodes:
                min_steps = min(d["steps"] for d in successful_episodes)
                max_steps = max(d["steps"] for d in successful_episodes)
            else:
                min_steps = max_steps = 0

            # Save expert info to milestone detail
            milestone_detail["expert"] = {
                "success_rate": success_rate,
                "num_episodes": eval_result.get("num_episodes", 0),
                "num_success": eval_result.get("num_success", 0),
                "mean_steps": mean_steps,
                "std_steps": eval_result["std_steps"],
                "min_steps": min_steps,
                "max_steps": max_steps,
                "episodes": episode_details,
                "best_video": best_expert_video,
            }

            if success_rate < self.args.min_success_rate:
                print(f"❌ Expert success rate too low ({success_rate:.1%} < {self.args.min_success_rate:.1%})")
                result["status"] = "failed"
                result["reason"] = "expert success rate too low"
                milestone_detail["status"] = "failed"
                milestone_detail["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.save_milestone_detail(milestone_id, milestone_detail)
                self.results.append(result)
                self.update_pipeline_state(milestone_id, "failed")
                continue

            # Step 2: Calculate max_episode_steps
            max_episode_steps = int(mean_steps * self.args.step_multiplier)
            result["max_episode_steps"] = max_episode_steps

            print(f"\n✓ Expert Success Rate: {success_rate:.1%}")
            print(f"✓ Expert Mean Steps: {mean_steps:.1f} ± {eval_result['std_steps']:.1f}")
            print(f"✓ Computed max_episode_steps: {max_episode_steps}")

            # Step 3: Train DAgger
            self.update_pipeline_state(milestone_id, "dagger_train")
            train_result = self.train_dagger(milestone_id, starting_state, max_episode_steps)

            if not train_result.get("success"):
                print(f"❌ DAgger training failed")
                result["status"] = "failed"
                result["reason"] = "dagger training failed"
                result["error"] = train_result.get("error", "unknown")
                milestone_detail["status"] = "failed"
                milestone_detail["error"] = train_result.get("error", "unknown")
                milestone_detail["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.save_milestone_detail(milestone_id, milestone_detail)
                self.results.append(result)
                self.update_pipeline_state(milestone_id, "failed")
                continue

            # Get best dagger video
            dagger_video_dir = os.path.join(self.args.output_dir, "dagger_videos", milestone_id)
            best_dagger_video = self.get_best_video(dagger_video_dir, milestone_id, video_type="dagger")

            # Save dagger info to milestone detail
            milestone_detail["dagger"] = {
                "max_episode_steps": max_episode_steps,
                "step_multiplier": self.args.step_multiplier,
                "num_iterations": self.args.dagger_iterations,
                "final_success_rate": train_result.get("final_success_rate", 0.0),
                "final_mean_return": train_result.get("final_mean_return", 0.0),
                "final_mean_length": train_result.get("final_mean_length", 0.0),
                "final_train_accuracy": train_result.get("final_train_accuracy"),
                "iteration_metrics": train_result.get("iteration_metrics", []),
                "best_video": best_dagger_video,
                "wandb_url": train_result.get("wandb_url"),
            }

            result["status"] = "completed"
            result["dagger_final_success_rate"] = train_result.get("final_success_rate", 0.0)
            milestone_detail["status"] = "completed"
            milestone_detail["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save milestone detail
            self.save_milestone_detail(milestone_id, milestone_detail)
            self.results.append(result)

            print(f"\n✅ Milestone {milestone_id} completed successfully!\n")

            # Save results after each milestone
            self.save_results()
            self.update_pipeline_state(None, "completed")

        total_time = time.time() - start_time

        # Final summary
        self.print_summary(total_time)

    def save_results(self):
        """Save results to CSV and JSON"""
        # Save JSON
        with open(self.results_json, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save CSV
        if self.results:
            keys = self.results[0].keys()
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)

    def print_summary(self, total_time: float):
        """Print final summary"""
        print("\n" + "=" * 70)
        print("Pipeline Summary")
        print("=" * 70 + "\n")

        completed = sum(1 for r in self.results if r["status"] == "completed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")

        print(f"Total Milestones: {len(self.results)}")
        print(f"  ✅ Completed: {completed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"\nTotal Time: {total_time / 60:.1f} minutes")
        print(f"\nResults saved to:")
        print(f"  - {self.results_csv}")
        print(f"  - {self.results_json}")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    args = tyro.cli(PipelineArgs)
    pipeline = MilestoneTrainingPipeline(args)
    pipeline.run()
