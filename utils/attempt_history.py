#!/usr/bin/env python3
"""
Attempt History Manager for Curriculum Learning

Tracks previous attempts for each milestone to provide feedback for retry attempts.
"""

import json
import os
from typing import Dict, List, Any, Optional


class AttemptHistory:
    """
    Manages attempt history for curriculum learning

    Stores information about previous failed attempts:
    - What code was tried
    - Why it failed (stuck, timeout, error)
    - Where it failed (location, state)
    - How long it took
    """

    def __init__(self, milestone_id: str):
        """
        Args:
            milestone_id: Milestone being trained
        """
        self.milestone_id = milestone_id
        self.attempts: List[Dict[str, Any]] = []

    def add_attempt(
        self,
        attempt_num: int,
        code: str,
        result: Dict[str, Any],
        final_state: Optional[Dict[str, Any]] = None,
        screenshot_history: Optional[List[tuple]] = None,
        logs: Optional[List[str]] = None
    ):
        """
        Add a failed attempt to history

        Args:
            attempt_num: Attempt number (1-indexed)
            code: Policy code that was executed
            result: Result dict with 'success', 'reason', 'steps', 'duration'
            final_state: Final game state when failed (optional)
            screenshot_history: List of (step_count, PIL.Image) tuples (last ~10 frames)
            logs: Debug logs from policy execution (optional)
        """
        attempt_data = {
            'attempt_num': attempt_num,
            'code': code,
            'reason': result['reason'],
            'steps': result['steps'],
            'duration': result['duration'],
            'logs': logs or []
        }

        # Store full final state (cleaned)
        if final_state:
            attempt_data['final_state'] = self._clean_state_for_storage(final_state)

        # Store screenshot history (PIL Images for next attempt)
        if screenshot_history:
            attempt_data['screenshot_history'] = screenshot_history

        self.attempts.append(attempt_data)

    def _clean_state_for_storage(self, state: dict) -> dict:
        """
        Remove non-serializable objects from state

        Args:
            state: State dict potentially containing PIL Images

        Returns:
            Cleaned state dict
        """
        cleaned = {}
        for key, value in state.items():
            # Skip PIL Images and other non-serializable objects
            if key in ['frame', 'visual']:
                continue
            # Recursively clean nested dicts
            if isinstance(value, dict):
                cleaned[key] = self._clean_state_for_storage(value)
            elif isinstance(value, list):
                # Clean list items if they are dicts
                cleaned[key] = [
                    self._clean_state_for_storage(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    def format_for_prompt(self) -> str:
        """
        Format attempt history for inclusion in LLM prompt

        Returns:
            Formatted string describing previous attempts
        """
        if not self.attempts:
            return ""

        lines = ["⚠️ PREVIOUS ATTEMPTS FAILED:"]
        lines.append("")

        for attempt in self.attempts:
            attempt_num = attempt['attempt_num']
            reason = attempt['reason']
            steps = attempt['steps']
            duration = attempt['duration']

            lines.append(f"Attempt {attempt_num}: FAILED ({reason})")
            lines.append(f"  Steps taken: {steps}")
            lines.append(f"  Duration: {duration:.1f}s")

            if 'final_location' in attempt:
                lines.append(f"  Final location: {attempt['final_location']}")

            if 'final_position' in attempt:
                pos = attempt['final_position']
                lines.append(f"  Final position: ({pos.get('x', 0)}, {pos.get('y', 0)})")

            # Show code snippet
            code = attempt['code']
            code_preview = code[:200] + "..." if len(code) > 200 else code
            lines.append(f"  Code tried:")
            lines.append(f"```python")
            lines.append(f"{code_preview}")
            lines.append(f"```")
            lines.append("")

        lines.append("🔧 INSTRUCTION: Analyze the previous failures and generate DIFFERENT code that avoids these issues.")
        lines.append("")

        return "\n".join(lines)

    def to_execution_error_format(self) -> Optional[Dict[str, Any]]:
        """
        Convert attempt history to CodeAgent's execution_error format

        This allows reusing CodeAgent's prompt builder infrastructure.

        Returns:
            Dict with 'error', 'code', 'traceback' keys, or None if no attempts
        """
        if not self.attempts:
            return None

        # Use most recent attempt
        latest = self.attempts[-1]

        # Build error message
        reason = latest['reason']
        steps = latest['steps']

        error_messages = {
            'stuck': f"Agent got stuck after {steps} steps (repeating same state)",
            'timeout': f"Execution timed out after {steps} steps",
            'max_steps': f"Reached maximum steps limit ({steps} steps)",
            'terminated': f"Episode terminated unexpectedly after {steps} steps",
            'truncated': f"Episode truncated after {steps} steps"
        }

        error_msg = error_messages.get(reason, f"Failed: {reason} after {steps} steps")

        # Add location info if available
        if 'final_location' in latest:
            error_msg += f" at location: {latest['final_location']}"

        # Format like CodeAgent's execution_error
        return {
            'error': error_msg,
            'code': latest['code'],
            'traceback': self.format_for_prompt()  # Full history in traceback
        }

    def save_to_file(self, directory: str = ".pokeagent_cache/attempt_histories"):
        """
        Save attempt history to file

        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory, f"{self.milestone_id}_history.json")

        data = {
            'milestone_id': self.milestone_id,
            'attempts': self.attempts
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, directory: str = ".pokeagent_cache/attempt_histories") -> bool:
        """
        Load attempt history from file

        Args:
            directory: Directory to load from

        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = os.path.join(directory, f"{self.milestone_id}_history.json")

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.attempts = data.get('attempts', [])
            return True
        except Exception as e:
            print(f"⚠️ Failed to load attempt history: {e}")
            return False

    def clear(self):
        """Clear all attempts"""
        self.attempts = []

    def get_attempt_count(self) -> int:
        """Get number of attempts"""
        return len(self.attempts)

    def get_latest_screenshot_history(self) -> Optional[List[tuple]]:
        """
        Get screenshot history from latest attempt

        Returns:
            List of (step_count, PIL.Image) tuples, or None if no history
        """
        if not self.attempts:
            return None

        latest = self.attempts[-1]
        return latest.get('screenshot_history', None)

    def get_latest_attempt(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent attempt

        Returns:
            Latest attempt dict, or None if no attempts
        """
        if not self.attempts:
            return None
        return self.attempts[-1]
