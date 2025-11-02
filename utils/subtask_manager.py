#!/usr/bin/env python3
"""
Subtask Manager for CodeAgent

Manages dynamic subtask decomposition for milestones.
Each milestone can have ONE active subtask at a time.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SubtaskManager:
    """
    Manages subtasks for CodeAgent milestone progression

    Key features:
    - Only ONE active subtask at a time (simple stack-free design)
    - Subtasks have precondition and success_condition
    - Auto-save/load subtask state per milestone
    - Integrate with MilestoneManager for custom milestone registration
    """

    def __init__(self, milestone_manager):
        """
        Initialize SubtaskManager

        Args:
            milestone_manager: MilestoneManager instance for custom milestone registration
        """
        self.milestone_manager = milestone_manager
        self.current_subtask = None  # Only one active subtask

        # Ensure cache directory exists
        self.cache_dir = ".pokeagent_cache/subtasks"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Path for current subtask (for web display)
        self.current_subtask_file = ".pokeagent_cache/current_subtask.json"

        logger.info("SubtaskManager initialized")

    def get_current_subtask(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active subtask

        Returns:
            Subtask dict or None if no active subtask
        """
        return self.current_subtask

    def set_current_subtask(self, subtask: Dict[str, Any]):
        """
        Set the current active subtask

        Args:
            subtask: Subtask dictionary with:
                - id: Unique subtask ID
                - parent_milestone_id: Parent milestone ID
                - description: Human-readable description
                - precondition: Python expression (optional)
                - success_condition: Python expression
                - code: Policy code (optional)
                - created_at: Creation timestamp
        """
        self.current_subtask = subtask

        # Register as custom milestone if success_condition exists
        if subtask and subtask.get('success_condition'):
            self._register_as_custom_milestone(subtask)

        # Save to file for web display
        self._save_current_subtask_to_file()

        logger.info(f"Set current subtask: {subtask['description'] if subtask else 'None'}")

    def clear_current_subtask(self):
        """
        Clear the current subtask (completed or skipped)
        """
        if self.current_subtask:
            logger.info(f"Clearing subtask: {self.current_subtask['description']}")
        self.current_subtask = None

        # Clear file for web display
        self._save_current_subtask_to_file()

    def evaluate_condition(self, condition_text: str, state: Dict[str, Any], prev_action: str = None) -> bool:
        """
        Evaluate a condition expression safely

        Args:
            condition_text: Python expression string (e.g., "state['player']['location'].find('2F') >= 0")
            state: Current game state dict
            action: Last action taken (optional)

        Returns:
            True if condition is met, False otherwise

        Note:
            Uses restricted eval with 'state' and 'action' variables available.
            Helper functions can be added later if needed.
        """
        if not condition_text or not condition_text.strip():
            return False

        try:
            # Restricted namespace - 'state' and 'action' variables available
            namespace = {
                "__builtins__": {},  # No built-in functions
                "state": state,
                "prev_action": prev_action
            }

            result = eval(condition_text, namespace)
            return bool(result)

        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            logger.debug(f"  Condition: {condition_text}")
            logger.debug(f"  State keys: {list(state.keys()) if state else 'None'}")
            logger.debug(f"  Prev action: {prev_action}")
            return False

    def save_state(self, milestone_id: str):
        """
        Save current subtask state to file

        Args:
            milestone_id: Milestone ID to save subtask for
        """
        filepath = os.path.join(self.cache_dir, f"{milestone_id}.json")

        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_subtask, f, indent=2, default=str)

            logger.debug(f"Saved subtask state to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save subtask state: {e}")

    def load_state(self, milestone_id: str) -> bool:
        """
        Load subtask state from file

        Args:
            milestone_id: Milestone ID to load subtask for

        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = os.path.join(self.cache_dir, f"{milestone_id}.json")

        if not os.path.exists(filepath):
            logger.debug(f"No saved subtask state found for {milestone_id}")
            return False

        try:
            with open(filepath, 'r') as f:
                self.current_subtask = json.load(f)

            logger.info(f"Loaded subtask state from {filepath}")
            logger.info(f"  Subtask: {self.current_subtask.get('description', 'Unknown')}")

            # Re-register as custom milestone
            if self.current_subtask and self.current_subtask.get('success_condition'):
                self._register_as_custom_milestone(self.current_subtask)

            return True

        except Exception as e:
            logger.error(f"Failed to load subtask state: {e}")
            return False

    def _register_as_custom_milestone(self, subtask: Dict[str, Any]):
        """
        Register subtask as a custom milestone for automatic completion tracking

        Args:
            subtask: Subtask dict with success_condition
        """
        try:
            # Create check function that evaluates success_condition
            def check_fn(state, action):
                return self.evaluate_condition(subtask['success_condition'], state)

            # Register with milestone manager
            self.milestone_manager.add_custom_milestone(
                milestone_id=subtask['id'],
                description=subtask['description'],
                insert_after=subtask['parent_milestone_id'],
                check_fn=check_fn,
                category="subtask"
            )

            logger.debug(f"Registered subtask as custom milestone: {subtask['id']}")

        except Exception as e:
            logger.warning(f"Failed to register subtask as custom milestone: {e}")

    def _save_current_subtask_to_file(self):
        """
        Save current subtask to file for web display

        Creates/updates .pokeagent_cache/current_subtask.json
        """
        try:
            if self.current_subtask:
                data = {
                    "subtask_id": self.current_subtask['id'],
                    "parent_milestone_id": self.current_subtask['parent_milestone_id'],
                    "description": self.current_subtask['description'],
                    "precondition": self.current_subtask.get('precondition', ''),
                    "success_condition": self.current_subtask.get('success_condition', ''),
                    "created_at": self.current_subtask.get('created_at', ''),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # No subtask - write null/empty
                data = None

            with open(self.current_subtask_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved current subtask to {self.current_subtask_file}")

        except Exception as e:
            logger.error(f"Failed to save current subtask to file: {e}")
