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
        self.completed_subtasks = []  # Track completed subtasks with timestamps

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

    def get_recent_completed_subtasks(self, count: int = 3) -> list:
        """
        Get recent completed subtasks

        Args:
            count: Number of recent subtasks to return

        Returns:
            List of completed subtask dicts (most recent first)
        """
        # Return last N subtasks (most recent first)
        return self.completed_subtasks[-count:] if self.completed_subtasks else []

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

        # NOTE: We don't register subtasks as custom milestones to prevent nesting issues.
        # Subtask completion is tracked via determine_situation() in CodeAgent.

        # Save to file for web display
        self._save_current_subtask_to_file()

        logger.info(f"Set current subtask: {subtask['description'] if subtask else 'None'}")

    def clear_current_subtask(self, completed=True):
        """
        Clear the current subtask (completed or skipped)

        Args:
            completed: If True, adds to completed_subtasks history
        """
        if self.current_subtask:
            logger.info(f"Clearing subtask: {self.current_subtask['description']}")

            # Record completed subtask with timestamp
            if completed:
                self.completed_subtasks.append({
                    'id': self.current_subtask['id'],
                    'description': self.current_subtask['description'],
                    'parent_milestone_id': self.current_subtask.get('parent_milestone_id'),
                    'success_condition': self.current_subtask.get('success_condition', ''),
                    'timestamp': datetime.now().timestamp(),
                    'type': 'subtask'  # For distinguishing from milestones
                })
                logger.debug(f"Added to completed subtasks: {self.current_subtask['description']}")

        self.current_subtask = None

        # Clear file for web display
        self._save_current_subtask_to_file()

    def evaluate_condition(self, condition_text: str, state: Dict[str, Any]):
        """
        Evaluate a condition expression safely

        Args:
            condition_text: Python expression string (e.g., "state['player']['location'].find('2F') >= 0")
            state: Current game state dict (includes prev_action field)

        Returns:
            tuple: (success: bool, result: bool, error_msg: str)
                - success: True if evaluation succeeded, False if exception occurred
                - result: True if condition is met, False otherwise
                - error_msg: Error message if evaluation failed, empty string otherwise

        Note:
            Uses eval with built-in functions available.
        """
        if not condition_text or not condition_text.strip():
            return (True, False, "")  # Empty condition = successfully evaluated as False

        # Clean up condition text - remove markdown code blocks and backticks
        # Gemini sometimes wraps conditions in ```python ... ``` or ``` ... ``` or ` ... `
        cleaned_condition = condition_text.strip()

        # Remove ```python and ``` markers
        if cleaned_condition.startswith("```python"):
            cleaned_condition = cleaned_condition[9:]  # Remove ```python
        if cleaned_condition.startswith("```"):
            cleaned_condition = cleaned_condition[3:]  # Remove ```
        if cleaned_condition.endswith("```"):
            cleaned_condition = cleaned_condition[:-3]  # Remove trailing ```

        # Remove single backticks at start/end
        if cleaned_condition.startswith("`") and cleaned_condition.endswith("`"):
            cleaned_condition = cleaned_condition[1:-1]

        cleaned_condition = cleaned_condition.strip()

        if not cleaned_condition:
            return (True, False, "")  # Empty after cleaning

        try:
            # Namespace with built-in functions enabled
            namespace = {
                "__builtins__": __builtins__,
                "state": state
            }

            result = eval(cleaned_condition, namespace)
            return (True, bool(result), "")  # Success, with result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"Condition evaluation failed: {error_msg}")
            logger.debug(f"  Condition: {condition_text}")
            logger.debug(f"  State keys: {list(state.keys()) if state else 'None'}")
            logger.debug(f"  Prev action: {state.get('prev_action', 'no_op') if state else 'no_op'}")
            return (False, False, error_msg)  # Evaluation failed

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

            # NOTE: We don't re-register as custom milestone to prevent nesting issues.

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
                # Add prev_action to state (same pattern as agent.step())
                state["prev_action"] = action if action else "no_op"

                # Convert raw game state to formatted state (same format as code execution)
                from utils.state_formatter import convert_state_to_dict
                formatted_state = convert_state_to_dict(state)
                return self.evaluate_condition(subtask['success_condition'], formatted_state)[1]

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
