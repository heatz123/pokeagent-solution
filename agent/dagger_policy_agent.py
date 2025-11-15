#!/usr/bin/env python3
"""
DAgger Policy Agent - Executes pre-trained neural network policies

This agent loads and executes DAgger-trained neural network weights (.pth files)
from the models/ directory. No LLM calls - pure neural network inference for
fast, cost-free execution.

Supports two modes:
1. Single model mode: One model for all milestones (model_path specified)
2. Multi-milestone mode: Auto-discover models per milestone (models_dir specified)
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

import torch
import torch.nn.functional as F

from rl_training.common.networks import ImpalaCNN
from rl_training.common.utils import preprocess_observation
from utils.milestone_manager import MilestoneManager
from utils.model_finder import find_best_model_for_milestone, discover_all_milestone_models


class DAggerPolicyAgent:
    """Agent that executes pre-trained DAgger neural network policies"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        action_dim: int = 10,
        hidden_dim: int = 1024,
        device: Optional[str] = None,
        start_from_milestone: str = None,
        prefer_final: bool = False,
        deterministic: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize DAggerPolicyAgent

        Args:
            model_path: Path to single .pth model file (single model mode)
            models_dir: Directory with milestone models (multi-milestone)
            action_dim: Number of actions (default: 10)
            hidden_dim: Hidden dimension (default: 1024, auto-detected)
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            start_from_milestone: Optional milestone ID to start from
            prefer_final: Prioritize _final.pth over _best.pth
            deterministic: Use argmax (True) or sample from distribution (False)
            temperature: Sampling temperature for stochastic mode (default: 1.0)

        Note: Either model_path OR models_dir must be specified (not both)
        """
        if model_path and models_dir:
            raise ValueError("Specify either model_path OR models_dir, not both")
        if not model_path and not models_dir:
            raise ValueError("Must specify either model_path or models_dir")

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.deterministic = deterministic
        self.temperature = temperature
        self.log_probs = False  # Will be set by external code (e.g., DAgger data collection)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🤖 DAggerPolicyAgent")
        print(f"   Device: {self.device}")
        print(f"   Mode: {'Deterministic (argmax)' if deterministic else f'Stochastic (temperature={temperature})'}")

        # Mode selection
        if model_path:
            # Single model mode
            self.mode = "single"
            self.model_path = model_path
            self.models_dir = None
            self.milestone_models = {}
            print(f"   Mode: Single model")
            print(f"   Model: {model_path}")
            self.model = self._load_single_model(model_path)
        else:
            # Multi-milestone mode
            self.mode = "multi"
            self.model_path = None
            self.models_dir = models_dir
            print(f"   Mode: Multi-milestone (auto-discovery)")
            print(f"   Models dir: {models_dir}")
            if prefer_final:
                print(f"   Priority: _final.pth > _best.pth")
            # Discover available models
            self.milestone_models = discover_all_milestone_models(
                models_dir, prefer_final
            )
            print(f"   📦 Discovered {len(self.milestone_models)} models:")
            for mid, mpath in sorted(self.milestone_models.items()):
                print(f"      {mid:<20} -> {os.path.basename(mpath)}")
            # Lazy load models on demand
            self.loaded_models = {}  # milestone_id -> model
            self.model = None  # Will be set based on current milestone

        # Initialize milestone manager
        self.milestone_manager = MilestoneManager()

        # Register custom milestones
        self._register_custom_milestones()

        # Track last action for prev_action in state
        self._last_action = None
        self._last_facing = None

        # Custom milestone completion tracking (client-side only)
        self.custom_milestone_completions = {}

        # Initialize starting milestone if provided
        if start_from_milestone:
            self._initialize_starting_milestone(start_from_milestone)

        # Action mapping (index -> action string)
        self.action_map = ["a", "b", "start", "select", "up", "down", "left", "right", "l", "r"]

        # Track last action probabilities for logging
        self._last_action_probs = None

    def _load_single_model(self, model_path: str):
        """Load single model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Auto-detect hidden_dim from model weights if not explicitly set
        # fc.2.weight shape is [hidden_dim, 3872]
        detected_hidden_dim = self.hidden_dim
        if 'fc.2.weight' in state_dict:
            detected_hidden_dim = state_dict['fc.2.weight'].shape[0]
            if detected_hidden_dim != self.hidden_dim:
                print(f"   ⚠️  Hidden dim mismatch: expected {self.hidden_dim}, detected {detected_hidden_dim}")
                print(f"   ✓ Using detected hidden_dim: {detected_hidden_dim}")
                self.hidden_dim = detected_hidden_dim

        # Create model with correct hidden_dim
        model = ImpalaCNN(action_dim=self.action_dim, hidden_dim=self.hidden_dim)

        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded checkpoint:")
            print(f"     Iteration: {checkpoint.get('iteration', 'N/A')}")
            print(f"     Global step: {checkpoint.get('global_step', 'N/A')}")
            if 'eval_success_rate' in checkpoint:
                print(f"     Eval success rate: {checkpoint['eval_success_rate']:.1%}")
        else:
            # Direct state dict (no wrapper)
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded model state dict")

        model.to(self.device)
        model.eval()  # Set to evaluation mode

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")

        return model

    def _load_model_for_milestone(self, milestone_id: str):
        """
        Load model for specific milestone (multi-milestone mode only)

        Args:
            milestone_id: Milestone ID

        Returns:
            Model for milestone, or None if not found
        """
        if self.mode != "multi":
            raise RuntimeError("_load_model_for_milestone only works in multi-milestone mode")

        # Check if already loaded
        if milestone_id in self.loaded_models:
            return self.loaded_models[milestone_id]

        # Check if model exists for this milestone
        if milestone_id not in self.milestone_models:
            print(f"⚠️ No model found for milestone: {milestone_id}")
            return None

        model_path = self.milestone_models[milestone_id]
        print(f"📦 Loading model for {milestone_id}: {os.path.basename(model_path)}")

        # Load model
        model = self._load_single_model(model_path)

        # Cache for future use
        self.loaded_models[milestone_id] = model

        return model

    def _register_custom_milestones(self):
        """Register custom milestones with completion conditions"""
        from utils.milestone_registration import register_all_milestones

        # Use centralized registration
        register_all_milestones(self.milestone_manager)

    def _initialize_starting_milestone(self, start_milestone_id: str):
        """
        Mark all milestones up to and including start_milestone_id as completed

        Args:
            start_milestone_id: Milestone ID to start from
        """
        ordered = self.milestone_manager.get_ordered_milestones()

        # Find the position of start_milestone_id
        start_index = -1
        for i, m in enumerate(ordered):
            if m['id'] == start_milestone_id:
                start_index = i
                break

        if start_index == -1:
            print(f"⚠️  Warning: Start milestone '{start_milestone_id}' "
                  f"not found in milestone list")
            return

        # Mark all milestones up to start_index as completed
        timestamp = time.time()

        for i in range(start_index + 1):  # Include start_milestone itself
            milestone_id = ordered[i]['id']

            # Add ALL milestones (server + custom) to custom_milestone_completions
            # Since we now use client-side tracking only
            self.custom_milestone_completions[milestone_id] = {
                'completed': True,
                'timestamp': timestamp
            }

        print(f"✅ Initialized {start_index + 1} milestones as completed "
              f"up to '{start_milestone_id}'")

    def step(self, game_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Execute neural network policy to select action

        Args:
            game_state: Game state dict with keys:
                - 'frame': PIL Image
                - 'player': player info dict
                - 'game': game info dict
                - 'map': map info dict
                - 'visual': visual info dict
                - 'milestones': milestone completion dict

        Returns:
            {'action': action_string} where action is one of:
            'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right', 'l', 'r'
        """
        # Add prev_action to game_state
        game_state["prev_action"] = (
            self._last_action if self._last_action else "no_op"
        )

        # Update facing based on prev_action
        if self._last_action in ['up', 'down', 'left', 'right']:
            facing_map = {
                'up': 'north',
                'down': 'south',
                'left': 'west',
                'right': 'east'
            }
            self._last_facing = facing_map[self._last_action]

        # Add facing to game_state
        game_state["facing"] = (
            self._last_facing if self._last_facing else "north"
        )

        # Check custom milestone completions BEFORE finding next milestone
        # This ensures newly completed milestones are recognized in the current step
        # Note: self._last_action matches game_state timing (both from previous step)
        self._check_custom_milestones(game_state, self._last_action)

        # Get milestones (use client-side tracking only, ignore server state)
        milestones = self.custom_milestone_completions

        # Find next incomplete milestone
        next_milestone_info = self.milestone_manager.get_next_milestone_info(milestones)

        if not next_milestone_info:
            print("🎉 All milestones completed!")
            return {'action': 'b'}

        milestone_id = next_milestone_info['id']

        # Track current milestone for timeout handling
        self._current_milestone_id = milestone_id

        # In multi-milestone mode, load appropriate model
        if self.mode == "multi":
            # Load model for this milestone if not already loaded
            if milestone_id not in self.loaded_models:
                self._load_model_for_milestone(milestone_id)

            # Get model for this milestone
            current_model = self.loaded_models.get(milestone_id)

            if current_model is None:
                print(f"⚠️ No model available for milestone {milestone_id}, using 'b'")
                return {'action': 'b'}
        else:
            # Single model mode
            current_model = self.model

        # Get screenshot and preprocess
        screenshot = game_state.get('frame')
        if screenshot is None:
            print("⚠️ No screenshot in game state, using 'b'")
            return {'action': 'b'}

        # Preprocess observation
        obs_preprocessed = preprocess_observation(screenshot)

        # Convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(obs_preprocessed).unsqueeze(0).to(self.device)

        # Get action from model
        with torch.no_grad():
            logits = current_model(obs_tensor)

            # Compute probabilities for logging
            probs = F.softmax(logits / self.temperature, dim=1)
            self._last_action_probs = probs[0].cpu().numpy()  # Store for logging

            if self.deterministic:
                # Deterministic: argmax (greedy)
                action_idx = torch.argmax(logits, dim=1).item()
            else:
                # Stochastic: sample from distribution
                action_idx = torch.multinomial(probs, num_samples=1).item()

        # Convert action index to action string
        action = self.action_map[action_idx]

        # Log action probabilities (only if log_probs=True, e.g., DAgger mode)
        if self.log_probs:
            prob_str = ", ".join([f"{self.action_map[i]}:{self._last_action_probs[i]:.3f}"
                                  for i in range(len(self.action_map))])
            print(f"   🎲 Action: {action} (prob={self._last_action_probs[action_idx]:.3f})")
            print(f"      All probs: {prob_str}")

        # Update last action for next step
        self._last_action = action

        return {'action': action}

    def get_current_milestone_id(self):
        """Get currently executing milestone ID"""
        return getattr(self, '_current_milestone_id', None)

    def get_last_action_probs(self):
        """Get action probabilities from last step"""
        return self._last_action_probs

    def _check_custom_milestones(self, game_state, action):
        """
        Check and track custom milestone completions (client-side only)

        Args:
            game_state: Current game state from server
            action: Action that was just executed
        """
        # Use client-side milestone tracking only
        milestones = self.custom_milestone_completions

        for custom in self.milestone_manager.custom_milestones:
            milestone_id = custom["id"]

            # Skip if already completed
            if self.custom_milestone_completions.get(milestone_id, {}).get('completed', False):
                continue

            # Check if previous milestone is completed
            insert_after_id = custom["insert_after"]
            if insert_after_id and not milestones.get(insert_after_id, {}).get('completed', False):
                continue

            # Check condition
            check_fn = custom["check_fn"]
            try:
                if check_fn(game_state, action):
                    print(f"🎯 Custom milestone completed: {milestone_id}")
                    self.custom_milestone_completions[milestone_id] = {
                        'completed': True,
                        'timestamp': time.time()
                    }
            except Exception as e:
                print(f"⚠️ Error checking custom milestone {milestone_id}: {e}")
