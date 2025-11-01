"""
Prompt Builder for SimpleAgent

Provides a clean separation of prompt generation logic from agent logic.
Allows for easy customization and testing of prompts.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    # Display counts
    actions_display_count: int = 40
    history_display_count: int = 30
    max_active_objectives: int = 5
    max_completed_objectives: int = 3

    # Section toggles
    include_pathfinding_rules: bool = True
    include_movement_memory: bool = True
    include_stuck_warning: bool = True

    # Formatting options
    use_emojis: bool = True
    verbose_objectives: bool = True

    # Template strings (can be customized)
    system_instruction_template: str = field(default=None)
    response_structure_template: str = field(default=None)
    pathfinding_rules_template: str = field(default=None)

    def __post_init__(self):
        """Set default templates if not provided"""
        if self.system_instruction_template is None:
            self.system_instruction_template = self._default_system_instruction()
        if self.response_structure_template is None:
            self.response_structure_template = self._default_response_structure()
        if self.pathfinding_rules_template is None:
            self.pathfinding_rules_template = self._default_pathfinding_rules()

    @staticmethod
    def _default_system_instruction() -> str:
        return """You are playing as the Protagonist in Pokemon Emerald. Progress quickly to the milestones by balancing exploration and exploitation of things you know, but have fun for the Twitch stream while you do it.
            Based on the current game frame and state information, think through your next move and choose the best button action.
            If you notice that you are repeating the same action sequences over and over again, you definitely need to try something different since what you are doing is wrong! Try exploring different new areas or interacting with different NPCs if you are stuck."""

    @staticmethod
    def _default_response_structure() -> str:
        return """IMPORTANT: Please think step by step before choosing your action. Structure your response like this:

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing?
IMPORTANT: Look carefully at the game image for objects (clocks, pokeballs, bags) and NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue. When you see them try determine their location (X,Y) on the map relative to the player and any objects.]

OBJECTIVES:
[Review your current objectives. You have main storyline objectives (story_*) that track overall Emerald progression - these are automatically verified and you CANNOT manually complete them.  There may be sub-objectives that you need to complete before the main milestone. You can create your own sub-objectives to help achieve the main goals. Do any need to be updated, added, or marked as complete?
- Add sub-objectives: ADD_OBJECTIVE: type:description:target_value (e.g., "ADD_OBJECTIVE: location:Find Pokemon Center in town:(15,20)" or "ADD_OBJECTIVE: item:Buy Pokeballs:5")
- Complete sub-objectives only: COMPLETE_OBJECTIVE: objective_id:notes (e.g., "COMPLETE_OBJECTIVE: my_sub_obj_123:Successfully bought Pokeballs")
- NOTE: Do NOT try to complete storyline objectives (story_*) - they auto-complete when milestones are reached]

PLAN:
[Think about your immediate goal - what do you want to accomplish in the next few actions? Consider your current objectives and recent history.
Check MOVEMENT MEMORY for areas you've had trouble with before and plan your route accordingly.]

REASONING:
[Explain why you're choosing this specific action. Reference the MOVEMENT PREVIEW and MOVEMENT MEMORY sections. Check the visual frame for NPCs before moving. If you see NPCs in the image, avoid walking into them. Consider any failed movements or known obstacles from your memory.]

ACTION:
[Your final action choice - PREFER SINGLE ACTIONS like 'RIGHT' or 'A'. Only use multiple actions like 'UP, UP, RIGHT' if you've verified each step is WALKABLE in the movement preview and map.]"""

    @staticmethod
    def _default_pathfinding_rules() -> str:
        return """
🚨 PATHFINDING RULES:
1. **SINGLE STEP FIRST**: Always prefer single actions (UP, DOWN, LEFT, RIGHT, A, B) unless you're 100% certain about multi-step paths
2. **CHECK EVERY STEP**: Before chaining movements, verify EACH step in your sequence using the MOVEMENT PREVIEW and map
3. **BLOCKED = STOP**: If ANY step shows BLOCKED in the movement preview, the entire sequence will fail
4. **NO BLIND CHAINS**: Never chain movements through areas you can't see or verify as walkable
5. **PERFORM PATHFINDING**: Find a path to a target location (X',Y') from the player position (X,Y) on the map. DO NOT TRAVERSE THROUGH OBSTACLES (#) -- it will not work.

💡 SMART MOVEMENT STRATEGY:
- Use MOVEMENT PREVIEW to see exactly what happens with each direction
- If your target requires multiple steps, plan ONE step at a time
- Only chain 2-3 moves if ALL intermediate tiles are confirmed WALKABLE
- When stuck, try a different direction rather than repeating the same blocked move
- After moving in a direction, you will be facing that direction for interactions with NPCs, etc.

EXAMPLE - DON'T DO THIS:
❌ "I want to go right 5 tiles" → "RIGHT, RIGHT, RIGHT, RIGHT, RIGHT" (may hit wall on step 2!)

EXAMPLE - DO THIS INSTEAD:
✅ Check movement preview → "RIGHT shows (X+1,Y) WALKABLE" → "RIGHT" (single safe step)
✅ Next turn, check again → "RIGHT shows (X+2,Y) WALKABLE" → "RIGHT" (another safe step)

💡 SMART NAVIGATION:
- The Player's sprite in the visual frame is located at the coordinates (X,Y) in the game state. Objects in the visual frame should be represented in relation to the Player's sprite.
- Check the VISUAL FRAME for NPCs (people/trainers) and other objects like clocks before moving - they're not always on the map! NPCs may block movement even when the movement preview shows them as walkable.
- Review MOVEMENT MEMORY for locations where you've failed to move before
- Only explore areas marked with ? (these are confirmed explorable edges)
- Avoid areas surrounded by # (walls) - they're fully blocked
- Use doors (D), stairs (S), or walk around obstacles when pathfinding suggests it

💡 NPC & OBSTACLE HANDLING:
- If you see NPCs in the image, avoid walking into them or interact with A/B if needed
- If a movement fails (coordinates don't change), that location likely has an NPC or obstacle
- Use your MOVEMENT MEMORY to remember problem areas and plan around them
- NPCs can trigger battles or dialogue, which may be useful for objectives
"""


class SimpleAgentPromptBuilder:
    """
    Builds prompts for SimpleAgent

    Separates prompt generation logic from agent logic for better
    maintainability, testing, and customization.
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize the prompt builder

        Args:
            config: Prompt configuration. If None, uses default config.
        """
        self.config = config or PromptConfig()
        logger.debug(f"Initialized SimpleAgentPromptBuilder with config: {self.config}")

    def build_prompt(
        self,
        formatted_state: str,
        recent_actions: List[str],
        history_entries: List[Any],
        active_objectives: List[Any],
        completed_objectives: List[Any],
        movement_memory: str,
        stuck_warning: str,
        context: str,
        coords: Optional[Tuple[int, int]],
        **kwargs
    ) -> str:
        """
        Build the complete prompt for SimpleAgent

        Args:
            formatted_state: Pre-formatted game state from format_state_for_llm()
            recent_actions: List of recent button presses
            history_entries: List of HistoryEntry objects
            active_objectives: List of active Objective objects
            completed_objectives: List of completed Objective objects
            movement_memory: Pre-formatted movement memory string
            stuck_warning: Pre-formatted stuck warning string
            context: Current game context ("overworld", "dialog", "battle", "menu", "title")
            coords: Current player coordinates (x, y)
            **kwargs: Additional arguments (for future extensibility)

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. System instruction
        sections.append(self.build_system_instruction())
        sections.append("")  # Empty line

        # 2. Recent action history
        sections.append(self.build_action_history_section(
            recent_actions,
            self.config.actions_display_count
        ))
        sections.append("")

        # 3. Location/context history
        sections.append(self.build_location_history_section(
            history_entries,
            self.config.history_display_count
        ))
        sections.append("")

        # 4. Current objectives
        sections.append(self.build_objectives_section(
            active_objectives,
            completed_objectives
        ))
        sections.append("")

        # 5. Current game state
        sections.append(self.build_game_state_section(formatted_state))
        sections.append("")

        # 6. Movement memory (if available and enabled)
        if self.config.include_movement_memory and movement_memory:
            sections.append(self.build_movement_memory_section(movement_memory))
            sections.append("")

        # 7. Stuck warning (if available and enabled)
        if self.config.include_stuck_warning and stuck_warning:
            sections.append(self.build_stuck_warning_section(stuck_warning))
            sections.append("")

        # 8. Available actions
        sections.append(self.build_available_actions_section())
        sections.append("")

        # 9. Response structure
        sections.append(self.build_response_structure_section())
        sections.append("")

        # 10. Pathfinding rules (only if not in title sequence)
        if self.should_include_pathfinding_rules(context):
            sections.append(self.build_pathfinding_rules_section())
            sections.append("")

        # 11. Context info
        sections.append(self.build_context_info_section(context, coords))

        # Join all sections
        prompt = "\n".join(sections)

        logger.debug(f"Built prompt with {len(sections)} sections, total length: {len(prompt)} chars")
        return prompt

    def build_system_instruction(self) -> str:
        """Build the main system instruction section"""
        return self.config.system_instruction_template

    def build_action_history_section(
        self,
        recent_actions: List[str],
        display_count: int
    ) -> str:
        """
        Build the recent action history section

        Args:
            recent_actions: List of recent button presses
            display_count: Number of recent actions to display

        Returns:
            Formatted action history section
        """
        # Get last N actions
        actions_to_show = recent_actions[-display_count:] if recent_actions else []
        actions_str = ', '.join(actions_to_show) if actions_to_show else 'None'

        return f"""RECENT ACTION HISTORY (last {display_count} actions):
{actions_str}"""

    def build_location_history_section(
        self,
        history_entries: List[Any],
        display_count: int
    ) -> str:
        """
        Build the location/context history section

        Args:
            history_entries: List of HistoryEntry objects
            display_count: Number of history entries to display

        Returns:
            Formatted location history section
        """
        if not history_entries:
            history_summary = "No previous history."
        else:
            # Get last N entries
            recent_entries = history_entries[-display_count:]

            # Format each entry
            summary_lines = []
            for i, entry in enumerate(recent_entries, 1):
                formatted_entry = self.format_history_entry(entry, i)
                summary_lines.append(formatted_entry)

            history_summary = "\n".join(summary_lines)

        return f"""LOCATION/CONTEXT HISTORY (last {display_count} steps):
{history_summary}"""

    def build_objectives_section(
        self,
        active_objectives: List[Any],
        completed_objectives: List[Any]
    ) -> str:
        """
        Build the current objectives section

        Args:
            active_objectives: List of active Objective objects
            completed_objectives: List of completed Objective objects

        Returns:
            Formatted objectives section
        """
        objectives_summary = self._format_objectives_for_llm(
            active_objectives,
            completed_objectives
        )

        return f"""CURRENT OBJECTIVES:
{objectives_summary}"""

    def build_game_state_section(self, formatted_state: str) -> str:
        """
        Build the current game state section

        Args:
            formatted_state: Pre-formatted game state string

        Returns:
            Formatted game state section
        """
        return f"""CURRENT GAME STATE:
{formatted_state}"""

    def build_movement_memory_section(self, movement_memory: str) -> str:
        """
        Build the movement memory section

        Args:
            movement_memory: Pre-formatted movement memory string

        Returns:
            Formatted movement memory section
        """
        return movement_memory

    def build_stuck_warning_section(self, stuck_warning: str) -> str:
        """
        Build the stuck warning section

        Args:
            stuck_warning: Pre-formatted stuck warning string

        Returns:
            Formatted stuck warning section
        """
        return stuck_warning

    def build_available_actions_section(self) -> str:
        """Build the available actions section"""
        return "Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT"

    def build_response_structure_section(self) -> str:
        """Build the response structure instructions section"""
        return self.config.response_structure_template

    def build_pathfinding_rules_section(self) -> str:
        """Build the pathfinding rules section"""
        return self.config.pathfinding_rules_template

    def build_context_info_section(
        self,
        context: str,
        coords: Optional[Tuple[int, int]]
    ) -> str:
        """
        Build the closing context info section

        Args:
            context: Current game context
            coords: Current player coordinates

        Returns:
            Formatted context info
        """
        return f"Context: {context} | Coords: {coords}"

    # ========================================
    # Helper Methods
    # ========================================

    def format_history_entry(self, entry: Any, index: int) -> str:
        """
        Format a single history entry

        Args:
            entry: HistoryEntry object
            index: Entry number (1-indexed)

        Returns:
            Formatted history entry string
        """
        coord_str = f"({entry.player_coords[0]},{entry.player_coords[1]})" if entry.player_coords else "(?)"
        return f"{index}. {entry.context} at {coord_str}: {entry.action_taken}"

    def format_objectives_list(
        self,
        objectives: List[Any],
        max_count: int = 5
    ) -> str:
        """
        Format a list of objectives

        Args:
            objectives: List of Objective objects
            max_count: Maximum number of objectives to show

        Returns:
            Formatted objectives list
        """
        if not objectives:
            return ""

        lines = []
        for i, obj in enumerate(objectives[:max_count], 1):
            target_str = f" (Target: {obj.target_value})" if obj.target_value else ""
            lines.append(f"  {i}. [{obj.objective_type}] {obj.description}{target_str} [ID: {obj.id}]")

        return "\n".join(lines)

    def _format_objectives_for_llm(
        self,
        active_objectives: List[Any],
        completed_objectives: List[Any]
    ) -> str:
        """
        Format objectives for LLM consumption

        Args:
            active_objectives: List of active Objective objects
            completed_objectives: List of completed Objective objects

        Returns:
            Formatted objectives summary
        """
        lines = []

        # Active objectives
        if active_objectives:
            emoji = "🎯" if self.config.use_emojis else ""
            lines.append(f"{emoji} ACTIVE OBJECTIVES:".strip())

            objectives_list = self.format_objectives_list(
                active_objectives,
                self.config.max_active_objectives
            )
            lines.append(objectives_list)
        else:
            emoji = "🎯" if self.config.use_emojis else ""
            lines.append(f"{emoji} ACTIVE OBJECTIVES: None - Consider setting some goals!".strip())

        # Recently completed objectives
        if completed_objectives:
            recent_completed = completed_objectives[-self.config.max_completed_objectives:]
            emoji = "✅" if self.config.use_emojis else ""
            lines.append(f"{emoji} RECENTLY COMPLETED:".strip())

            for obj in recent_completed:
                checkmark = "✓" if self.config.use_emojis else "*"
                lines.append(f"  {checkmark} [{obj.objective_type}] {obj.description}")

        return "\n".join(lines)

    def should_include_pathfinding_rules(self, context: str) -> bool:
        """
        Determine if pathfinding rules should be included

        Args:
            context: Current game context

        Returns:
            True if pathfinding rules should be included, False otherwise
        """
        if not self.config.include_pathfinding_rules:
            return False

        # Don't include pathfinding rules in title sequence
        return context != "title"


# ============================================
# CodeAgent Prompt Builder
# ============================================


@dataclass
class CodePromptConfig:
    """Configuration for CodeAgent prompt generation"""

    # Section toggles
    include_visual_note: bool = True
    include_milestones: bool = True
    include_example_code: bool = True
    include_state_schema: bool = True

    # Template strings (can be customized)
    system_instruction_template: str = field(default=None)
    code_requirements_template: str = field(default=None)
    example_code_template: str = field(default=None)
    state_schema_template: str = field(default=None)

    def __post_init__(self):
        """Set default templates if not provided"""
        if self.system_instruction_template is None:
            self.system_instruction_template = self._default_system_instruction()
        if self.code_requirements_template is None:
            self.code_requirements_template = self._default_code_requirements()
        if self.example_code_template is None:
            self.example_code_template = self._default_example_code()
        if self.state_schema_template is None:
            self.state_schema_template = self._default_state_schema()

    @staticmethod
    def _default_system_instruction() -> str:
        return """You are playing as the AI agent in Pokemon Emerald. Your goal is to progress through the game by analyzing the current state and making strategic decisions through code.

Based on the current game frame and state information, think through your next move step by step, then write Python policy code that helps progress toward the next milestone."""

    @staticmethod
    def _default_code_requirements() -> str:
        return """IMPORTANT: Please think step by step before writing your code. Structure your response like this:

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing?
IMPORTANT: Look carefully at the game image for objects and NPCs that might not be shown on the map. Consider both visual and text information.]

OBJECTIVES:
[Review the current milestone and your progress. What is the immediate goal? What steps are needed to reach the next milestone?]

KNOWLEDGE UPDATE (Optional):
[If you learned something important that should be remembered for future runs, add it here:
ADD_KNOWLEDGE: <one sentence describing the fact>
Example: ADD_KNOWLEDGE: The clock has been set.
Example: ADD_KNOWLEDGE: Talking to Mom triggers the Pokedex event.
Only add NEW facts that aren't already in the knowledge base above.]

PLAN:
[Think about your immediate goal - what do you want to accomplish in the next action? Consider the current milestone and game state.]

REASONING:
[Explain why you're choosing this specific action. Reference the game state, visual information, and your plan. Why is this the best move right now?]

CODE:
[Your final Python code - define a function called 'run' that takes 'state' as parameter and returns ONE action string OR a list of actions.
Add brief comments explaining your logic. It should work regardless of the specific state within the current milestone. Keep it simple and focused.]
- Valid actions: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'
- Single action: return 'up'
- Multiple actions (executed in sequence): return ['up', 'up', 'a']

REQUIREMENTS:
- Return action as a lowercase string OR list of lowercase strings
- Include helpful comments in your code"""

    @staticmethod
    def _default_example_code() -> str:
        return '''EXAMPLE RESPONSE:

ANALYSIS:
The frame is the moving van interior at game start. There are no mandatory dialogues here before exiting. The exit trigger is on the right edge of the van interior. Holding right will deterministically walk the player to the exit and transition to the next scene.

OBJECTIVES:
Milestone: “Exit the moving van.”
Success when the location changes from the van interior to the house (e.g., Mom’s house 1F) or an outdoor/entrance transition occurs.

PLAN:
Use a one-action policy: always return 'right' each step until the scene changes. No branching, no checks—this keeps the example minimal and robust for this specific milestone.

REASONING:
In this room, moving right is both sufficient and optimal: it requires no interaction, has no hazards, and deterministically triggers the exit. Simplicity here reduces surface area for mistakes and keeps the example focused.

CODE:
```python
def run(state):
    """
    Minimal policy for the milestone: Exit the moving van.
    Always returns 'right' to walk toward the exit trigger on the right edge.
    This intentionally avoids conditional logic to serve as a simple, optimal example.
    """
    # No state inspection needed; the van exit is reached by moving right.
    return 'right'
```'''

    @staticmethod
    def _default_state_schema() -> str:
        return """STATE DATA STRUCTURE:
When you write code, you receive a 'state' parameter with the following structure:

```python
state = {
    # Player information
    'player': {
        'name': str,                  # Player's name
        'position': {'x': int, 'y': int},  # Current coordinates
        'location': str,              # Current map name (e.g., "LITTLEROOT TOWN")
        'facing': str,                # "North", "South", "East", "West"
        'money': int,                 # Current money
        'party': [                    # List of Pokemon in party
            {
                'species_name': str,  # Pokemon name
                'level': int,
                'current_hp': int,
                'max_hp': int,
                'status': str,        # "Normal", "Poisoned", "Paralyzed", etc.
                'moves': [str, str, str, str],  # Up to 4 moves
                'types': [str] or [str, str]    # Pokemon types
            }
        ]
    },

    # Game state
    'game': {
        'game_state': str,            # "overworld", "dialog", "battle", "menu", "title"
        'is_in_battle': bool,
        'dialog_text': str,           # Current dialogue text (if any)
        'money': int,
        'badges': int,                # Number of badges earned
        'time': (int, int, int),      # (hours, minutes, seconds)
        'battle_info': {              # Only present during battles
            'battle_type': str,
            'player_pokemon': {...},
            'opponent_pokemon': {...}
        }
    },

    # Map information
    'map': {
        'id': int,
        'current_map': str,           # Map name
        'player_coords': {'x': int, 'y': int},
        'tiles': [[tile_data, ...], ...]  # 15x15 grid centered on player
    },

    # Other
    'step_number': int,
    'status': str
}
```
"""


class CodeAgentPromptBuilder:
    """
    CodeAgent를 위한 프롬프트 생성 클래스

    SimpleAgent와 유사한 패턴을 따르되, 코드 생성에 특화된 섹션 구성
    """

    def __init__(self, config: Optional[CodePromptConfig] = None):
        """
        Initialize the CodeAgent prompt builder

        Args:
            config: 프롬프트 설정. None이면 기본 설정 사용.
        """
        self.config = config or CodePromptConfig()
        logger.debug(f"Initialized CodeAgentPromptBuilder with config: {self.config}")

    def build_prompt(
        self,
        formatted_state: str,
        next_milestone_info: Optional[Dict[str, Any]] = None,
        stuck_warning: str = "",
        previous_code: str = "",
        execution_error: Optional[Dict[str, Any]] = None,
        knowledge_base: str = "",
        previous_actions: Optional[List[Any]] = None,
        **kwargs
    ) -> str:
        """
        전체 프롬프트를 생성하는 메인 메서드

        Args:
            formatted_state: format_state_for_llm()의 결과
            next_milestone_info: 다음 마일스톤 정보 (선택)
                - 'id': 마일스톤 ID
                - 'description': 마일스톤 설명
            stuck_warning: Stuck 감지 경고 메시지 (선택)
            previous_code: 이전 코드 정보 (stuck일 때만)
            execution_error: 실행 에러 정보 (에러 발생 시)
                - 'error': 에러 메시지
                - 'code': 에러가 발생한 코드
                - 'traceback': 트레이스백 (선택)
            knowledge_base: Knowledge base 텍스트 (선택)
            previous_actions: 이전 코드가 생성한 action 리스트 (선택)
            **kwargs: 미래 확장성을 위한 추가 인자

        Returns:
            완성된 프롬프트 문자열
        """
        # kwargs는 미래 확장성을 위해 유지 (현재 미사용)
        _ = kwargs

        sections = []

        # 1. System instruction
        sections.append(self.build_system_instruction())
        sections.append("")

        # 2. Visual note
        if self.config.include_visual_note:
            sections.append(self.build_visual_note_section())
            sections.append("")

        # 3. Current game state (SimpleAgent와 동일한 포맷)
        sections.append(self.build_game_state_section(formatted_state))
        sections.append("")

        # 4. Next milestone (optional)
        if self.config.include_milestones and next_milestone_info:
            sections.append(self.build_milestone_section(next_milestone_info))
            sections.append("")

        # 5. Knowledge Base (if available)
        if knowledge_base:
            sections.append(knowledge_base)
            sections.append("")

        # 6. Execution error (우선순위 높음 - stuck보다 먼저)
        if execution_error:
            sections.append(self.build_execution_error_section(execution_error))
            sections.append("")

        # 7. Stuck warning (if available)
        if stuck_warning:
            sections.append(self.build_stuck_warning_section(stuck_warning))
            sections.append("")

        # 8. Previous code (if stuck)
        if previous_code:
            sections.append(self.build_previous_code_section(previous_code))
            sections.append("")

        # 8.5. Actions from previous code (if available)
        if previous_actions and len(previous_actions) > 0:
            sections.append(self.build_previous_actions_section(previous_actions))
            sections.append("")

        # 9. State schema (optional)
        if self.config.include_state_schema:
            sections.append(self.build_state_schema_section())
            sections.append("")

        # 10. Code requirements
        sections.append(self.build_code_requirements_section())
        sections.append("")

        # 11. Example code (optional)
        if self.config.include_example_code:
            sections.append(self.build_example_code_section())

        # Join all sections
        prompt = "\n".join(sections)

        logger.debug(f"Built CodeAgent prompt with {len(sections)} sections, total length: {len(prompt)} chars")
        return prompt

    # ========================================
    # Section Builders
    # ========================================

    def build_system_instruction(self) -> str:
        """시스템 지시사항 섹션"""
        return self.config.system_instruction_template

    def build_visual_note_section(self) -> str:
        """비주얼 정보 안내 섹션 (CodeAgent 전용)"""
        return "VISUAL: You can see the current game screen in the attached image."

    def build_game_state_section(self, formatted_state: str) -> str:
        """
        현재 게임 상태 섹션

        SimpleAgent의 build_game_state_section()과 동일한 포맷 사용
        """
        return f"""CURRENT GAME STATE (text data):
{formatted_state}"""

    def build_stuck_warning_section(self, stuck_warning: str) -> str:
        """
        Stuck 경고 섹션

        Args:
            stuck_warning: Stuck 경고 메시지

        Returns:
            포맷팅된 stuck warning 섹션
        """
        return stuck_warning

    def build_previous_code_section(self, previous_code: str) -> str:
        """
        이전 코드 섹션 (stuck일 때 표시)

        Args:
            previous_code: 이전에 생성한 raw Python 코드

        Returns:
            포맷팅된 이전 코드 섹션
        """
        return f"""PREVIOUS CODE (NOT WORKING):
The following code was generated but resulted in a stuck state (same game state repeated 3+ times).
This code needs to be modified or rewritten:

```python
{previous_code}
```

IMPORTANT: The above code did NOT work. You can try a different strategy:
- If it was moving in one direction, try a different direction
- If it was pressing A, try exploring or moving instead
- If it was exploring, try interacting with NPCs or objects
- Consider the game state and visual carefully - what might have been blocking progress?
"""

    def build_previous_actions_section(self, previous_actions: List[Any]) -> str:
        """
        이전 코드로 생성된 action들 표시 섹션

        Args:
            previous_actions: 이전 코드가 생성한 action 리스트

        Returns:
            포맷팅된 action 히스토리 섹션
        """
        # Limit to last 20 actions if too many
        display_actions = previous_actions[-20:] if len(previous_actions) > 20 else previous_actions

        # Format actions for display
        actions_str = ', '.join(str(a) for a in display_actions)

        total_count = len(previous_actions)
        display_count = len(display_actions)

        if total_count > display_count:
            header = f"ACTIONS FROM PREVIOUS CODE (last {display_count} of {total_count}):"
        else:
            header = f"ACTIONS FROM PREVIOUS CODE (total {total_count}):"

        return f"""{header}
{actions_str}

⚠️ These are the actions that the previous code generated. If stuck, analyze why these actions didn't work."""

    def build_execution_error_section(self, execution_error: Dict[str, Any]) -> str:
        """
        실행 에러 피드백 섹션

        Args:
            execution_error: 에러 정보
                - 'error': 에러 메시지
                - 'code': 에러가 발생한 코드
                - 'traceback': 트레이스백 (선택)

        Returns:
            포맷팅된 에러 피드백 섹션
        """
        error_msg = execution_error.get('error', 'Unknown error')
        error_code = execution_error.get('code', '')
        traceback_str = execution_error.get('traceback', '')

        section = "⚠️ PREVIOUS CODE HAD AN EXECUTION ERROR:\n"
        section += f"\nError: {error_msg}\n"
        section += f"\nFailed code:\n```python\n{error_code}\n```\n"

        if traceback_str:
            section += f"\nDetailed traceback:\n{traceback_str}\n"

        section += "\n🔧 Please analyze the error carefully and generate CORRECTED code that fixes this issue."
        section += "\nMake sure the new code is syntactically correct and handles edge cases properly.\n"

        return section

    def build_milestone_section(self, milestone_info: Dict[str, Any]) -> str:
        """
        다음 마일스톤 정보 섹션 (CodeAgent 전용)

        Args:
            milestone_info: 마일스톤 정보
                - 'id': 마일스톤 ID
                - 'description': 마일스톤 설명

        Returns:
            포맷팅된 마일스톤 섹션
        """
        milestone_id = milestone_info.get('id', 'Unknown')
        milestone_desc = milestone_info.get('description', 'No description')

        return f"""NEXT MILESTONE:
ID: {milestone_id}
Goal: {milestone_desc}"""

    def build_state_schema_section(self) -> str:
        """State 자료구조 설명 섹션 (CodeAgent 전용)"""
        return self.config.state_schema_template

    def build_code_requirements_section(self) -> str:
        """코드 생성 요구사항 섹션 (CodeAgent 전용)"""
        return self.config.code_requirements_template

    def build_example_code_section(self) -> str:
        """예시 코드 섹션 (CodeAgent 전용)"""
        return self.config.example_code_template