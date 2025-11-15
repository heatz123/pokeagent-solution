#!/usr/bin/env python3
"""
Curriculum Learning Prompt Builder

Specialized prompt builder for milestone-based curriculum learning.
Different from CodeAgent's prompt - optimized for retry attempts with visual feedback.

Screenshot structure:
- Screenshot #1: Current state (starting position)
- Screenshots #2-10: Latest attempt's final 9 frames (visual failure evidence)

Prompt structure:
1. Overview & system instruction
2. Milestone information
3. Current state (with Screenshot #1)
4. Knowledge base (filtered)
5. Previous attempt (latest only, with Screenshots #2-10)
6. Code generation instruction
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CurriculumPromptBuilder:
    """
    Prompt builder for curriculum learning mode

    Optimized for:
    - Single milestone focus
    - Visual failure analysis (screenshots)
    - Learning from most recent attempt
    - Full state context (inventory, badges, flags)
    """

    def build_prompt(
        self,
        current_state: Dict[str, Any],
        milestone_info: Dict[str, Any],
        knowledge_base: Optional[List[Dict]] = None,
        attempt_history: Optional[Any] = None,
        tools_description: str = "",
        enable_policy_logging: bool = True
    ) -> str:
        """
        Build curriculum learning prompt

        Args:
            current_state: Current game state dict (formatted)
            milestone_info: Milestone information
                - 'id': Milestone ID
                - 'description': Milestone description
                - 'check_description': Completion condition (optional)
            knowledge_base: Filtered knowledge entries
            attempt_history: AttemptHistory instance (optional)
            tools_description: Available tools documentation
            enable_policy_logging: If True, document log() function in prompt

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. Overview
        sections.append(self._build_overview(attempt_history))
        sections.append("")

        # 2. Milestone information
        sections.append(self._build_milestone_section(milestone_info))
        sections.append("")

        # 3. Current state
        sections.append(self._build_current_state_section(current_state))
        sections.append("")

        # 4. Knowledge base
        if knowledge_base:
            sections.append(self._build_knowledge_section(knowledge_base))
            sections.append("")

        # 5. Previous attempt (latest only)
        if attempt_history and attempt_history.get_attempt_count() > 0:
            num_screenshots = min(9, len(attempt_history.get_latest_screenshot_history() or []))
            sections.append(self._build_previous_attempt_section(attempt_history, num_screenshots))
            sections.append("")

        # 6. Code generation instruction
        sections.append(self._build_code_instruction(tools_description, attempt_history, enable_policy_logging))

        return "\n".join(sections)

    def _build_overview(self, attempt_history) -> str:
        """Build overview section"""
        if attempt_history and attempt_history.get_attempt_count() > 0:
            attempt_num = attempt_history.get_attempt_count() + 1
            return f"""You are an AI agent learning to play Pokemon Emerald through curriculum learning.
Your goal is to complete the current milestone by generating a Python policy function.

This is ATTEMPT #{attempt_num} - previous attempt(s) failed. Learn from the failure below and generate DIFFERENT code."""
        else:
            return """You are an AI agent learning to play Pokemon Emerald through curriculum learning.
Your goal is to complete the current milestone by generating a Python policy function.

This is your FIRST attempt at this milestone."""

    def _build_milestone_section(self, milestone_info: Dict[str, Any]) -> str:
        """Build milestone information section"""
        lines = ["## CURRENT MILESTONE"]
        lines.append(f"- ID: {milestone_info.get('id', 'unknown')}")
        lines.append(f"- Description: {milestone_info.get('description', 'No description')}")

        if 'check_description' in milestone_info:
            lines.append(f"- Completion condition: {milestone_info['check_description']}")

        return "\n".join(lines)

    def _build_current_state_section(self, current_state: Dict[str, Any]) -> str:
        """Build current state section (with Screenshot #1 reference)"""
        import pprint

        lines = ["## CURRENT GAME STATE (Screenshot #1)"]
        lines.append("The game has been reset to the starting state for this milestone.")
        lines.append("")
        lines.append("```python")
        lines.append(pprint.pformat(current_state, indent=2, width=100))
        lines.append("```")

        return "\n".join(lines)

    def _build_knowledge_section(self, knowledge_base: List[Dict]) -> str:
        """Build knowledge base section"""
        lines = ["## KNOWLEDGE BASE"]
        lines.append("Relevant information about the current area:")
        lines.append("")

        for entry in knowledge_base[:50]:  # Show up to 50 entries
            content = entry.get('content', '')
            if content:
                lines.append(f"- {content}")

        return "\n".join(lines)

    def _build_previous_attempt_section(self, attempt_history, num_screenshots: int) -> str:
        """Build previous attempt section (latest only)"""
        latest = attempt_history.get_latest_attempt()
        if not latest:
            return ""

        lines = ["## PREVIOUS ATTEMPT (MOST RECENT)"]
        lines.append(f"Attempt #{latest['attempt_num']} failed. Analyze what went wrong and generate DIFFERENT code.")
        lines.append("")

        # Failure summary
        lines.append(f"**Result:** FAILED ({latest['reason']})")
        lines.append(f"**Steps taken:** {latest['steps']}")
        lines.append(f"**Duration:** {latest['duration']:.1f}s")
        lines.append("")

        # Final state details (same format as current state for consistency)
        final_state = latest.get('final_state', {})
        if final_state:
            import pprint
            lines.append("### Final State (where it failed)")
            lines.append("```python")
            lines.append(pprint.pformat(final_state, indent=2, width=100))
            lines.append("```")
            lines.append("")

        # Code that failed
        lines.append("### Code That Failed")
        lines.append("```python")
        lines.append(latest.get('code', '# No code'))
        lines.append("```")
        lines.append("")

        # Debug logs (if any)
        logs = latest.get('logs', [])
        if logs:
            lines.append("### Debug Logs")
            lines.append(f"The code printed {len(logs)} debug messages:")
            for i, msg in enumerate(logs, 1):  # Show all logs
                lines.append(f"  {i}. {msg}")
            lines.append("")

        # Screenshot reference
        if num_screenshots > 0:
            lines.append(f"### Visual Evidence (Screenshots #2-{1 + num_screenshots})")
            lines.append(f"Screenshots #2-{1 + num_screenshots} show the FINAL {num_screenshots} frames where this code failed.")
            lines.append("Analyze the visual progression to understand what went wrong:")
            lines.append("- Did the agent get stuck in a loop?")
            lines.append("- Did it walk into a wall or NPC?")
            lines.append("- Did it fail to interact with an object?")
            lines.append("- Did it go in the wrong direction?")

        return "\n".join(lines)

    def _build_code_instruction(self, tools_description: str, attempt_history, enable_policy_logging: bool = True) -> str:
        """Build code generation instruction"""
        lines = ["## YOUR TASK"]
        lines.append("Generate a Python policy function that will complete this milestone.")
        lines.append("")

        if tools_description:
            lines.append("### Available Tools")
            lines.append(tools_description)
            lines.append("")

        # Add analysis instructions
        lines.append("## YOUR ANALYSIS")
        lines.append("")
        lines.append("Before coding, analyze step by step:")
        lines.append("")
        lines.append("### 1. Visual Analysis (Screenshot #1)")
        lines.append("- Scene: What environment do you see?")
        lines.append("- Player: Position and facing direction?")
        lines.append("- Dialog/UI: Any dialog or menu visible? What text?")
        lines.append("- Objects/NPCs: What objects (doors, items, NPCs) are visible and where?")
        lines.append("")

        if attempt_history and attempt_history.get_attempt_count() > 0:
            lines.append("### 2. Temporal Analysis (Screenshots #2-10)")
            lines.append("- Position changes: How did player move across frames?")
            lines.append("- Dialog progression: Did dialog advance or stay stuck?")
            lines.append("- Patterns: Are frames identical (stuck) or showing progress?")
            lines.append("")

        lines.append("### 3. Knowledge Review")
        lines.append("- Relevant info: What knowledge entries help with this milestone?")
        lines.append("- Missing info: What additional context would be useful?")
        lines.append("")
        lines.append("### 4. Plan & Reasoning")
        lines.append("- Goal: What's your immediate objective?")
        lines.append("- Why: Why is this action the best choice?")
        lines.append("- How: How does it move toward milestone completion?")
        lines.append("")

        lines.append("### Requirements")
        lines.append("- Function signature: `def run(state):`")
        lines.append("- Return: single action string ('up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'no_op')")
        lines.append("- Use available tools for navigation and interaction")
        lines.append("- DO NOT import from tools - all tool functions are already available in global scope")
        lines.append("- Access state using dict syntax: state['player']['position']")
        # lines.append("- VLM queries: Use add_to_state_schema() for visual info not in state (e.g., clock UI, battle UI)")
        # lines.append("""- Example: add_to_state_schema("is_in_battle", "Is the battle screen visible?", bool), then access via state["is_in_battle"]""")
        # lines.append("- Note: VLM is expensive - prefer state dict info when available")

        # Add log() documentation if enabled
        if enable_policy_logging:
            lines.append("- Debugging: Use log('message') to print debug info")
            lines.append("- Example: log(f\"Position: {state['player']['position']}\")")

        lines.append("")

        if attempt_history and attempt_history.get_attempt_count() > 0:
            lines.append("### Important")
            lines.append("- Screenshot #1 shows your STARTING state")
            lines.append("- Screenshots #2-10 show where the PREVIOUS attempt FAILED")
            lines.append("- Learn from the failure and generate DIFFERENT code")
            lines.append("- Do NOT repeat the same approach that failed")
        else:
            lines.append("### Important")
            lines.append("- Screenshot #1 shows your starting state")
            lines.append("- Analyze the visual frame and current state carefully")

        lines.append("")
        lines.append("## RESPONSE FORMAT")
        lines.append("")
        lines.append("Structure your response as:")
        lines.append("")
        lines.append("**VISUAL:**")
        lines.append("[Screenshot analysis]")
        lines.append("")

        if attempt_history and attempt_history.get_attempt_count() > 0:
            lines.append("**TEMPORAL:**")
            lines.append("[Frame progression analysis - Note: Both attempts started from the same initial state (Screenshot #1)]")
            lines.append("- Code analysis: Which part of the previous code logic caused this behavior?")
            lines.append("- Root cause: What condition or assumption in the code was incorrect?")
            lines.append("- Fix strategy: What specific logic needs to change in the new attempt?")
            lines.append("")

        lines.append("**KNOWLEDGE:**")
        lines.append("[Relevant knowledge entries used]")
        lines.append("")
        lines.append("**PLAN:**")
        lines.append("[Immediate goal]")
        lines.append("")
        lines.append("**REASONING:**")
        lines.append("[Why this action]")
        lines.append("")
        lines.append("**CODE:**")
        lines.append("```python")
        lines.append("def run(state):")
        lines.append("    # Your implementation")
        lines.append("    return action")
        lines.append("```")

        return "\n".join(lines)

    def get_screenshots(
        self,
        current_screenshot,
        attempt_history: Optional[Any] = None
    ) -> List:
        """
        Get screenshots in order for LLM

        Returns:
            [current_screenshot, prev_frame_1, prev_frame_2, ..., prev_frame_9]
        """
        screenshots = [current_screenshot]

        if attempt_history:
            latest_screenshots = attempt_history.get_latest_screenshot_history()
            if latest_screenshots:
                # Get last 9 frames
                for step_count, frame in latest_screenshots[-9:]:
                    screenshots.append(frame)

        return screenshots