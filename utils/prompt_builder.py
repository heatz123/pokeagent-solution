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
[Explain why you're choosing this specific action. Reference the MOVEMENT MEMORY sections. Check the visual frame for NPCs before moving. If you see NPCs in the image, avoid walking into them. Consider any failed movements or known obstacles from your memory.]

ACTION:
[Your final action choice - PREFER SINGLE ACTIONS like 'RIGHT' or 'A'. Only use multiple actions like 'UP, UP, RIGHT' if you've verified each step is WALKABLE in the movement preview and map.]"""

    @staticmethod
    def _default_pathfinding_rules() -> str:
        return """

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
    include_knowledge_update: bool = True
    include_execution_logs: bool = True
    include_visual_observation_examples: bool = False  # Show add_to_state_schema examples in subtask mode

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
            self.code_requirements_template = self._build_code_requirements_template()
        if self.example_code_template is None:
            self.example_code_template = self._default_example_code()
        if self.state_schema_template is None:
            self.state_schema_template = self._default_state_schema()

    @staticmethod
    def _default_system_instruction() -> str:
        return """You are playing as the AI agent in Pokemon Emerald. Your goal is to progress through the game by analyzing the current state and making strategic decisions through code.

Based on the current game frame and state information, think through your next move step by step, then write Python policy code that helps progress toward the next milestone."""

    def _build_code_requirements_template(self) -> str:
        """Build code requirements template with optional KNOWLEDGE_UPDATE section"""

        knowledge_section = """
KNOWLEDGE_UPDATE:
[If you learned something important that should be remembered for future runs, add it here:
ADD_KNOWLEDGE: <one sentence describing the fact>
Example: ADD_KNOWLEDGE: The clock has been set.
Example: ADD_KNOWLEDGE: Talking to Mom triggers the Pokedex event.
Only add NEW facts that aren't already in the knowledge base above.]
""" if self.include_knowledge_update else ""

        return f"""IMPORTANT: Please think step by step before writing your code. Structure your response like this:

CRITICAL_EVALUATION:
[Before proceeding, critically evaluate your approach:
1. Are we meeting the main milestone requirements?
   - Review the current milestone description and your progress toward it
   - Is our approach aligned with milestone goals?
2. Is the current approach working?
   - Review recent ANALYSIS sections - are we stuck in repeated failed attempts?
   - If you see patterns of failure, what fundamentally needs to change?]

ANALYSIS:
- Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing?
IMPORTANT: Look carefully at the game image for objects (clocks, pokeballs, bags) and NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue. When you see them try determine their location (X,Y) on the map relative to the player and any objects.
- If the previous code resulted in a stuck situation, analyze why it got stuck. We may have to try a different approach.

OBJECTIVES:
[Review the current milestone and your progress. What is the immediate goal? What steps are needed to reach the next milestone?]
{knowledge_section}
PLAN:
[Think about your immediate goal - what do you want to accomplish in the next action? Consider the current milestone and game state.]

REASONING:
[Explain why you're choosing this specific action. Reference the game state, visual information, and your plan. Why is this the best move right now?]

CODE:
[Your final Python code - define a function called 'run' that takes 'state' as parameter and returns ONE action string OR a list of actions.
Add brief comments explaining your logic. Keep it simple and focused.]

REQUIREMENTS:
- Return action as a lowercase string OR list of lowercase strings
- Valid actions: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right', 'no_op'
- Action sequences are limited to a maximum of 3 actions
- Include helpful comments in your code"""

    @staticmethod
    def _default_example_code() -> str:
        return '''EXAMPLE RESPONSE:

ANALYSIS:
The frame is the moving van interior at game start. There are no mandatory dialogues here before exiting. The exit trigger is on the right edge of the van interior. Holding right will deterministically walk the player to the exit and transition to the next scene.

OBJECTIVES:
Milestone: "Exit the moving van."
Success when the location changes from the van interior to the house (e.g., Mom's house 1F) or an outdoor/entrance transition occurs.

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
```

---

EXAMPLE WITH VISUAL OBSERVATION:

ANALYSIS:
Looking at the frame, I can see we're in the bedroom with a clock on the wall. The player is currently at position (5, 2). The clock appears to be one tile north of the player. I need to check if the clock UI is already open to decide whether to move or interact.

OBJECTIVES:
Milestone: "Set the clock"
Need to interact with the clock to open its UI and set the time.

PLAN:
Use visual observation to check if clock UI is open. If not open, move up to the clock and interact. If already open, press A to confirm.

REASONING:
The visual frame shows important UI state that may not be reflected in the text state. By querying the VLM, I can make more informed decisions about whether to navigate or interact.

CODE:
```python
# Register visual observation for clock UI state
add_to_state_schema(
    key="is_clock_ui_open",
    vlm_prompt="Is the clock setting UI currently visible on screen? Look for any menus, dialogs or interfaces related to setting time.",
    return_type=bool
)

def run(state):
    """
    Policy for clock interaction with visual observation.
    Uses VLM to check if clock UI is already open.
    """
    # Access visual observation (VLM call happens here on first access)
    if state["is_clock_ui_open"]:
        # Clock UI is open, confirm the time
        return 'a'
    else:
        # Clock UI not open, navigate and interact
        player_x = state['player']['position']['x']
        player_y = state['player']['position']['y']

        # Check if we're at the right position (5, 2)
        if player_x == 5 and player_y == 2:
            # We're right below the clock, move up and interact
            return ['up', 'a']
        else:
            # Navigate to clock position first
            if player_x < 5:
                return 'right'
            elif player_x > 5:
                return 'left'
            elif player_y < 2:
                return 'down'
            else:
                return 'up'
```

IMPORTANT NOTES ON VISUAL OBSERVATIONS:
- Use add_to_state_schema() BEFORE the run() function to register visual queries
- Visual queries are cached per step (multiple accesses to state["key"] = one VLM call)
- Use sparingly as VLM calls add 1-2 seconds latency per unique query
- Prefer using state dict data when available (coordinates, location, etc.)
- Use visual observations for UI state, menu detection, NPC positions, or other visual-only info
- Be specific in your vlm_prompt for better accuracy

---

EXAMPLE WITH LOGGING:

ANALYSIS:
Looking at the frame, I can see we're in the bedroom with a clock on the wall. The player is at position (5, 2). I need to interact with the clock to set it, which requires moving up and pressing A.

OBJECTIVES:
Milestone: "Set the clock"
Need to interact with the clock to open its UI and set the time.

PLAN:
Move up to position (5, 1) where the clock is, then press A to interact. Use logging to track progress and help with debugging.

REASONING:
Logging helps debug and track what the code is doing. This is especially useful when diagnosing stuck situations or understanding the decision flow. The logs will appear in the next prompt under EXECUTION LOGS section.

CODE:
```python
def run(state):
    """
    Policy for clock interaction with logging.
    Uses log() to record debug information for future reference.
    """
    player_x = state['player']['position']['x']
    player_y = state['player']['position']['y']
    location = state['player']['location']

    log(f"Current position: ({player_x}, {player_y})")
    log(f"Location: {location}")

    # Check if we're at the right position (5, 2)
    if player_x == 5 and player_y == 2:
        log("At clock position! Moving up and interacting")
        return ['up', 'a']
    elif player_x == 5 and player_y == 1:
        log("Already at clock, just pressing A")
        return 'a'
    else:
        # Navigate to clock position first
        log(f"Navigating to clock from ({player_x}, {player_y})")
        if player_x < 5:
            log("Moving right")
            return 'right'
        elif player_x > 5:
            log("Moving left")
            return 'left'
        elif player_y < 2:
            log("Moving down")
            return 'down'
        else:
            log("Moving up")
            return 'up'
```

IMPORTANT NOTES ON log() FUNCTION:
- Use log('message') to record debug information during code execution
- Logs are shown in the next prompt under EXECUTION LOGS section
- Useful for debugging stuck situations, tracking decisions, or noting game dialog
- Prefer f-strings for formatted messages (more readable and efficient)
- Examples:
  - log(f"Current position: ({x}, {y})")  # Use f-strings for values
  - log("Checking if dialog is open...")   # Static messages
  - log(f"Dialog text: {dialog_text}")     # Dynamic content
  - log(f"NPC count: {npc_count}")         # Variable values
- Logs persist across multiple steps, so you can see what happened in previous runs
- Use logs to understand patterns, track state changes, and diagnose issues'''

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
        'time': (int, int, int)       # (hours, minutes, seconds)
    },

    # Map information
    'map': {
        'location': str,              # Map location name
        'current_map': str,           # Current map name
        'player_coords': {'x': int, 'y': int},  # Player coordinates
        'ascii_map': [str, str, ...], # List of strings - each row is a string (stitched/global map)
        'legend': [str, str, ...],    # List of strings - map legend explaining symbols
        'player_position': {          # Player position with facing direction
            'x': int,                 #   Absolute X coordinate
            'y': int,                 #   Absolute Y coordinate
        },
        'warps': [                    # Warp/door information (optional)
            {
                'position': {'x': int, 'y': int},  # Warp location (absolute coords)
                'leads_to': str,      # Destination location name
                'type': str,          # 'door', 'stairs', 'route_transition'
                'direction': str      # 'north', 'south', 'up', 'down'
            }
        ]
    },

    # Battle information (only present during battles)
    'battle_info': {
        'battle_type': str,           # 'wild', 'trainer', etc.
        'player_pokemon': {...},      # Your active Pokemon details
        'opponent_pokemon': {...}     # Opponent's active Pokemon details
    }
}
```

Notes on state structure:
- The 'ascii_map' field is a list of strings (each row is a string) showing the current map with GAME coordinates
- Map shows from game coordinate (0,0) to explored maximum (not just 15x15 view)
- Coordinates are GAME coordinates: 0-based coordinates starting from the origin of the current map
- Unexplored areas within the map range are shown as '?' symbols
- Use the 'legend' field (list of strings) to understand what each symbol means (P=Player, .=walkable, #=wall, D=door, S=stairs, etc.)
- To display the map, join the list: '\n'.join(state['map']['ascii_map'])
- Player position is marked with 'P' and coordinates are in 'player_position' (world coords)
- 'warps' list shows doors/stairs with their destinations and world coordinates
- All coordinates (player_position, warps, map headers) use the same world coordinate system
- NPCs and some obstacles may NOT be shown on the map - MUST check the visual screenshot
- Navigation strategy: Use visual image for NPCs/obstacles, ascii_map for spatial planning
- 'battle_info' is only available during battles
- game_state and dialog_text can be unreliable - trust the visual image when detecting dialogue
- You can use action sequences (maximum 3 actions) like ['up', 'a'] to handle interactions and movements together
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
        current_subtask: Optional[Dict[str, Any]] = None,
        stuck_warning: str = "",
        previous_code: str = "",
        execution_error: Optional[Dict[str, Any]] = None,
        knowledge_base: Any = None,
        previous_actions: Optional[List[Any]] = None,
        previous_analyses: Optional[List[Tuple[int, str]]] = None,
        execution_logs: Optional[List[Tuple[int, str]]] = None,
        **kwargs
    ) -> str:
        """
        전체 프롬프트를 생성하는 메인 메서드

        Args:
            formatted_state: format_state_for_llm()의 결과
            next_milestone_info: 다음 마일스톤 정보 (선택)
                - 'id': 마일스톤 ID
                - 'description': 마일스톤 설명
            current_subtask: 현재 활성 subtask (선택)
            stuck_warning: Stuck 감지 경고 메시지 (선택)
            previous_code: 이전 코드 정보 (stuck일 때만)
            execution_error: 실행 에러 정보 (에러 발생 시)
                - 'error': 에러 메시지
                - 'code': 에러가 발생한 코드
                - 'traceback': 트레이스백 (선택)
            knowledge_base: KnowledgeBase instance (선택)
            previous_actions: 이전 코드가 생성한 action 리스트 (선택)
            previous_analyses: 이전 ANALYSIS 섹션 리스트 (선택)
            execution_logs: 코드 실행 중 log() 함수로 기록된 로그들 (선택)
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

        # 1.5. Environment introduction
        sections.append(self.build_environment_section())
        sections.append("")

        # 2. Visual note
        if self.config.include_visual_note:
            num_screenshots = kwargs.get('num_screenshots', 1)
            sections.append(self.build_visual_note_section(num_screenshots))
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
            sections.append(self.build_knowledge_base_section(knowledge_base))
            sections.append("")

        # 5.5. Previous analyses (if available)
        if previous_analyses and len(previous_analyses) > 0:
            sections.append(self.build_previous_analyses_section(previous_analyses))
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

        # 8.6. Execution logs (if available and enabled)
        if self.config.include_execution_logs and execution_logs and len(execution_logs) > 0:
            sections.append(self.build_execution_logs_section(execution_logs))
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

    def build_visual_note_section(self, num_screenshots: int = 1) -> str:
        """
        비주얼 정보 안내 섹션 (CodeAgent 전용)

        Args:
            num_screenshots: Number of screenshots attached (1 or more)

        Returns:
            Visual note text describing the screenshots
        """
        if num_screenshots <= 1:
            return "VISUAL: You can see the current game screen in the attached image."
        else:
            return f"""VISUAL OBSERVATIONS - READ IN ORDER:
You will see {num_screenshots} screenshots in REVERSE chronological order (newest → oldest).

⭐ Screenshot 1/{num_screenshots} is the CURRENT STATE ⭐
  - This shows the present moment
  - Use this for ALL current decisions (location, dialog, position)
  - This is the MOST IMPORTANT screenshot

📜 Screenshots 2-{num_screenshots} are HISTORY (recent → old)
  - Screenshot 2 = 1 step ago
  - Screenshot 3 = 2 steps ago
  - Screenshot {num_screenshots} = {num_screenshots-1} steps ago (OLDEST)
  - Use these to understand how you got to the current state
  - Look for stuck patterns (same visuals repeating)

FORMAT: Each screenshot is labeled as:
- [CURRENT STATE ⭐] Screenshot 1/{num_screenshots} (Step X):
- [History -N steps ago] Screenshot M/{num_screenshots} (Step Y):

IMPORTANT:
- ALWAYS analyze Screenshot 1 (CURRENT STATE) first!
- Use the step numbers to correlate with the EXECUTION LOGS section
- The screenshots are in REVERSE order - newest first!"""

    def build_game_state_section(self, formatted_state: str) -> str:
        """
        현재 게임 상태 섹션

        ASCII map을 시각적으로 표시하고 나머지는 JSON으로 표시
        """
        import json

        # JSON 파싱
        try:
            state_dict = json.loads(formatted_state)
        except (json.JSONDecodeError, TypeError):
            # Fallback: 파싱 실패시 그대로 반환
            return f"""CURRENT GAME STATE (text data):
{formatted_state}"""

        sections = []
        sections.append("CURRENT GAME STATE:")
        sections.append("")

        # Map visualization (if available)
        if 'map' in state_dict and 'ascii_map' in state_dict['map']:
            map_data = state_dict['map']

            sections.append("=" * 60)
            sections.append("MAP VISUALIZATION")
            sections.append("=" * 60)
            sections.append(f"Location: {map_data.get('location', 'Unknown')}")

            # Player position
            if 'player_position' in map_data:
                pos = map_data['player_position']
                sections.append(f"Player Position: ({pos['x']}, {pos['y']})")

            sections.append("")

            # ASCII map (list of strings)
            if isinstance(map_data['ascii_map'], list):
                for row in map_data['ascii_map']:
                    sections.append(row)
            else:
                # Fallback for string format
                sections.append(str(map_data['ascii_map']))

            sections.append("")

            # Legend (list of strings)
            if 'legend' in map_data:
                if isinstance(map_data['legend'], list):
                    for line in map_data['legend']:
                        sections.append(line)
                else:
                    # Fallback for string format
                    sections.append(str(map_data['legend']))
                sections.append("")

            # Warps
            # COMMENTED OUT: Warp positions are confusing (show spawn pos, not tile pos)
            # if 'warps' in map_data and map_data['warps']:
            #     sections.append("Available Warps:")
            #     for warp in map_data['warps']:
            #         pos = warp['position']
            #         sections.append(
            #             f"  • ({pos['x']}, {pos['y']}): {warp['type']} → "
            #             f"{warp['leads_to']} ({warp['direction']})"
            #         )
            #     sections.append("")

            sections.append("=" * 60)
            sections.append("")

        # Rest of state as JSON (filtering already done by filter_state_for_llm)
        sections.append("GAME STATE DATA (JSON):")
        sections.append(json.dumps(state_dict, indent=2))

        return "\n".join(sections)

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
        이전 코드로 생성된 action들과 각 action 실행 전 position 표시 섹션

        Args:
            previous_actions: 이전 코드가 생성한 (position, action) 튜플 리스트

        Returns:
            포맷팅된 action 히스토리 섹션
        """
        # Limit to last 20 actions if too many
        display_actions = previous_actions[-20:] if len(previous_actions) > 20 else previous_actions

        # Format actions with positions for display
        action_lines = []
        for item in display_actions:
            # Handle both new format (tuple) and old format (just action)
            if isinstance(item, tuple) and len(item) == 2:
                pos, action = item
                # Format position
                if pos and len(pos) == 2 and pos[0] is not None and pos[1] is not None:
                    pos_str = f"({pos[0]},{pos[1]})"
                else:
                    pos_str = "(?,?)"
            else:
                # Old format (just action)
                pos_str = "(?,?)"
                action = item

            # Format action (handle both single action and list)
            if isinstance(action, list):
                action_str = str(action)  # e.g., "['up', 'a']"
            else:
                action_str = str(action)

            action_lines.append(f"{pos_str} → {action_str}")

        actions_str = '\n'.join(action_lines)

        total_count = len(previous_actions)
        display_count = len(display_actions)

        if total_count > display_count:
            header = f"ACTIONS FROM PREVIOUS CODE (last {display_count} of {total_count}):"
        else:
            header = f"ACTIONS FROM PREVIOUS CODE (total {total_count}):"

        return f"""{header}
{actions_str}

⚠️ These are the actions that the previous code generated with the player position BEFORE each action was executed. If stuck, analyze why these actions didn't work."""

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

    def build_environment_section(self) -> str:
        """Build environment introduction section"""
        return """ENVIRONMENT:
You are controlling a Pokemon Emerald (GBA) game agent.

OBSERVATION:
- Visual: 240x160 screenshot of current game state (ground truth)
- State data: Player info, location, game state, dialog, map (see below for details)

IMPORTANT: State data may be inaccurate or incomplete due to memory reading limitations.
When there is conflict between visual observation and state data, TRUST THE VISUAL OBSERVATION.
Always verify critical information (dialog text, NPC positions, obstacles) from the screenshot.

ACTION SPACE:
Return ONE action string or a list of up to THREE actions:
- Movement: 'up', 'down', 'left', 'right' (tile-based)
- Buttons: 'a' (confirm/interact), 'b' (cancel/back), 'start' (menu), 'select'
- Special: 'no_op' (do nothing, useful when waiting)

Examples: return 'a'  or  return ['up', 'a']  or  return ['up', 'a', 'a']

KEY MECHANICS:
- Each action takes ~1 second to execute in game
- Movement is tile-based; walls/NPCs block movement
- Press 'a' near objects/NPCs to interact
- Dialogs require button presses to advance
- Game has natural delays (animations, transitions)"""

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

    def build_knowledge_base_section(self, knowledge_base: Any, limit: int = 20) -> str:
        """
        Format knowledge base entries for LLM prompt

        Args:
            knowledge_base: KnowledgeBase instance
            limit: Maximum number of recent entries to include

        Returns:
            Formatted string for prompt
        """
        if not knowledge_base or len(knowledge_base) == 0:
            return "KNOWLEDGE BASE: Empty. You can add learnings with ADD_KNOWLEDGE: <sentence>"

        # Get recent entries (returns list of KnowledgeEntry objects or dicts)
        recent_entries = knowledge_base.get_recent(limit)

        if not recent_entries:
            return "KNOWLEDGE BASE: Empty. You can add learnings with ADD_KNOWLEDGE: <sentence>"

        lines = ["KNOWLEDGE BASE (learned facts):"]
        for i, entry in enumerate(recent_entries, 1):
            # Handle both KnowledgeEntry objects and dicts
            if isinstance(entry, dict):
                content = entry['content']
                step = entry['created_step']
                milestone = entry['created_milestone']
            else:
                content = entry.content
                step = entry.created_step
                milestone = entry.created_milestone

            # Shorten milestone name for readability
            milestone_short = milestone.replace('story_', '')
            lines.append(f"{i}. {content} [Step {step}]")

        return "\n".join(lines)

    def build_previous_analyses_section(self, analyses: List[Tuple[int, str]]) -> str:
        """
        Format previous ANALYSIS sections for context

        Supports both:
        - Regular: (step_number, analysis_text)
        - Summary: ("X-Y", summary_text) for step ranges

        Args:
            analyses: List of (step_info, analysis_text) tuples
                     step_info can be int or str (for ranges like "1-50")

        Returns:
            Formatted string for prompt
        """
        if not analyses:
            return "PREVIOUS ANALYSES: None yet."

        # Display all analyses (after summarization, should be 1-20 entries)
        lines = [f"PREVIOUS ANALYSES (showing {len(analyses)} entries):"]

        for step_info, text in analyses:
            # Check if it's a summary (step_info is a range string like "1-50")
            if isinstance(step_info, str) and '-' in step_info:
                # It's a summary
                lines.append(f"\n[Steps {step_info} Summary]")
            else:
                # Regular analysis
                lines.append(f"\n[Step {step_info}]")

            lines.append(text.strip())
            lines.append("")  # Empty line between analyses for clarity

        return "\n".join(lines)

    def build_state_schema_section(self) -> str:
        """State 자료구조 설명 섹션 (CodeAgent 전용)"""
        return self.config.state_schema_template

    def build_execution_logs_section(self, execution_logs: List[Tuple[int, str]], limit: int = 30) -> str:
        """
        Format execution logs for LLM prompt

        Args:
            execution_logs: List of (step_count, message) tuples
            limit: Maximum number of recent logs to include (default: 30)

        Returns:
            Formatted string for prompt
        """
        if not execution_logs or len(execution_logs) == 0:
            return "EXECUTION LOGS: No logs yet. Use log('message') in your code to record debug info."

        # Get recent logs (last N entries)
        recent_logs = execution_logs[-limit:] if len(execution_logs) > limit else execution_logs

        # Format output - one line per log entry
        lines = [f"EXECUTION LOGS (from your previous code, showing last {len(recent_logs)} entries):"]
        lines.append("")

        for step, msg in recent_logs:
            lines.append(f"[Step {step}] {msg}")

        lines.append("")
        lines.append("💡 These are debug messages you logged using log(). Use them to understand what happened in previous executions.")

        return "\n".join(lines)

    def build_code_requirements_section(self) -> str:
        """코드 생성 요구사항 섹션 (CodeAgent 전용)"""
        return self.config.code_requirements_template

    def build_example_code_section(self) -> str:
        """예시 코드 섹션 (CodeAgent 전용)"""
        return self.config.example_code_template

    # ========================================
    # Subtask-specific Methods
    # ========================================

    def build_subtask_info_section(
        self,
        current_subtask: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build current subtask information section

        Args:
            current_subtask: Current subtask dict (or None)

        Returns:
            Formatted subtask info (description, precondition, success_condition)
        """
        if not current_subtask:
            return """CURRENT SUBTASK:
None (just starting)
- Precondition: N/A
- Success Condition: N/A"""

        return f"""CURRENT SUBTASK:
{current_subtask['description']}
- Precondition: {current_subtask.get('precondition', 'N/A')}
- Success Condition: {current_subtask.get('success_condition', 'N/A')}"""

    def build_situation_section(
        self,
        situation: str
    ) -> str:
        """
        Build situation section with situation-specific instructions

        Args:
            situation: NORMAL, SUCCESS_ACHIEVED, PRECONDITION_FAILED, STUCK

        Returns:
            Formatted situation section
        """
        situation_instructions = {
            "NORMAL": """You are starting work on this milestone. Decide if you need to break it into subtasks.""",

            "SUCCESS_ACHIEVED": """✅ PREVIOUS SUBTASK COMPLETED!
The success condition was met. Now decide what to do next:
- Create the next subtask if there's more work
- Or mark milestone as complete if done""",

            "PRECONDITION_FAILED": """⚠️ PRECONDITION FAILED
The current subtask's precondition is no longer valid.
This means:
- The precondition might be wrong → refine it
- Or you need a different subtask to get back on track""",

            "STUCK": """🔴 AGENT IS STUCK
Same action/location repeated. Current approach is NOT working.
You can:
- Modify the run code
- Or break down into smaller subtasks
- Or modify the current subtask if it's unnecessary"""
        }

        instruction = situation_instructions.get(situation, "")
        return f"""SITUATION: {situation}
{instruction}"""

    def build_clock_example_section(self) -> str:
        """
        Build CLOCK_SET example section (always shown for subtask mode)

        Returns:
            Formatted clock example
        """
        return """
EXAMPLE: Setting clock in bedroom (CLOCK_SET milestone)
This involves several steps:
1. Enter house (if outside) → Location changes to "HOUSE 1F"
2. Go upstairs → Location changes to "HOUSE 2F"
3. Interact with clock → Navigate to specific coordinates, face up and press A

Each of these could be a subtask:

First subtask example:
Decision: CREATE_OR_MODIFY
Description: Enter player's house from town
Precondition: state['player']['location'] == 'LITTLEROOT TOWN'
Success Condition: state['player']['location'].find('HOUSE') >= 0

Second subtask example:
Decision: CREATE_OR_MODIFY
Description: Navigate to 2nd floor bedroom
Precondition: state['player']['location'].find('1F') >= 0
Success Condition: state['player']['location'].find('2F') >= 0

Third subtask example:
Decision: CREATE_OR_MODIFY
Description: Move to clock and interact (at position 5,1, then press up+A)
Precondition: state['player']['location'].find('2F') >= 0
Success Condition: (state['player']['position']['x'], state['player']['position']['y']) == (5, 1) and prev_action == ['up', 'a']

NOTE: Both 'state' and 'prev_action' variables are available in conditions.
- state: Current game state dict
- prev_action: Last action taken (can be single string like 'up' or list like ['up', 'a'])
- For multi-action sequences, code should return a list (e.g., return ['up', 'a'])

"""

    def build_subtask_response_structure(
        self,
        main_milestone: Dict[str, Any],
        current_subtask: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build response structure section for subtask mode

        Args:
            main_milestone: Main milestone dict
            current_subtask: Current subtask dict (or None)

        Returns:
            Formatted response structure instructions
        """
        task_desc = current_subtask['description'] if current_subtask else main_milestone['description']

        knowledge_section = """
KNOWLEDGE_UPDATE:
[If you learned something important that should be remembered for future runs, add it here:
ADD_KNOWLEDGE: <one sentence describing the fact>
Only add NEW facts that aren't already in the knowledge base above.]
""" if self.config.include_knowledge_update else ""

        milestone_desc = main_milestone.get('description', 'Unknown milestone')
        subtask_desc = current_subtask.get('description', 'No active subtask') if current_subtask else 'No active subtask'

        return f"""---

IMPORTANT: Structure your response EXACTLY like this (use 'SECTION_NAME:' format with colon, NOT markdown headers):

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing?
IMPORTANT: Look carefully at the game image for objects (clocks, pokeballs, bags) and NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue. When you see them try determine their location (X,Y) on the map relative to the player and any objects. If there is a dialog box, read the dialog text carefully.

APPROACH EVALUATION:
1. Current approach: [What strategy is the current/previous code using?]
2. Alternative idea: [Propose ONE completely different approach to achieve the milestone. Maybe our previous assumption was wrong. For example, if navigation fails repeatedly, try BFS pathfinding or wall-following instead of simple movement.]
3. Decision: [KEEP current approach | TRY alternative] because [explain your choice in 1 sentence]

[Continue analysis below...]

If EXECUTION LOGS section is shown above, review.
- What did the previous code log? What was it trying to do?
- What insights can you gain from the logged messages?
- Did the logs reveal any issues (e.g., repeated attempts, stuck situations, unexpected values)?
- Use the logs to understand the previous code's decision-making process

Check recent PREVIOUS ANALYSES. If similar failed attempt exists, QUOTE it:
  "[Step X] ..."
We may use DIFFERENT subtask idea or code approach.
If the last 3 steps did not change (position, location, or dialog), discard previous hypotheses and re-evaluate from the CURRENT screenshot and ascii_map only.]

{knowledge_section}
TASK_DECOMPOSITION:
[What should you do with the current subtask?

Options:
1. KEEP_CURRENT - Continue with current subtask as-is
2. CREATE_OR_MODIFY - Current task/precondition/success condition is done/invalid, create next subtask
3. DECOMPOSE - Current task is too complex, break it into smaller subtask

Before deciding KEEP/CREATE/DECOMPOSE, check:
If the same subtask's actions haven't changed location or state for several steps, treat it as infeasible.
When infeasible, create a new subtask that resolves whatever is blocking progress before retrying.

Reasoning: [Why are you making this decision? What evidence from state/observations/progress supports this choice?]

Decision: [KEEP_CURRENT | CREATE_OR_MODIFY | DECOMPOSE]

If CREATE_OR_MODIFY or DECOMPOSE, define the subtask:
Description: [What to do - keep same if only modifying conditions]
Precondition: [When to start - Python expression using 'state' (and 'prev_action' if needed). Must use only basic state fields from STATE DATA STRUCTURE.]
Success Condition: [When done - Python expression using 'state' (and 'prev_action' if needed). Must use only basic state fields from STATE DATA STRUCTURE.]

Note: For CREATE_OR_MODIFY, you can either:
- Create a new subtask (provide new Description, Precondition, Success Condition)
- Modify current subtask (keep Description, update Precondition/Success Condition)

Examples:

Example 1 (Creating new subtask):
Reasoning: The player is on 1F and needs to reach 2F bedroom to interact with clock. Current location shows 1F, so navigation to stairs and upstairs is the next logical step.
Decision: CREATE_OR_MODIFY
Description: Navigate to 2nd floor bedroom
Precondition: state['player']['location'].find('1F') >= 0
Success Condition: state['player']['location'].find('2F') >= 0

Example 2 (Modifying conditions):
Reasoning: The success condition was too loose - just reaching 2F isn't enough. Need to be at specific coordinates (5,1) and interact with clock by pressing up+a.
Decision: CREATE_OR_MODIFY
Description: Move to clock and interact
Precondition: state['player']['location'].find('2F') >= 0
Success Condition: state['player']['position']['x'] == 5 and state['player']['position']['y'] == 1 and prev_action in == ['up', 'a']

CODE:
[Implement policy for current subtask]

{f'''Note (Processing Visual Observations):
You can query the screenshot for visual information using the built-in add_to_state_schema() function.

IMPORTANT: add_to_state_schema is a BUILT-IN function already available in your code environment.
- DO NOT import it
- DO NOT define it yourself
- Just call it directly before your run() function

Usage:
- Call it BEFORE the run() function to register visual queries
- Syntax: add_to_state_schema(key="name", vlm_prompt="question", return_type=type)
- Supported types: bool, int, str, float, list, dict
- Results are cached per step (accessing state["key"] multiple times = 1 VLM call)
- Adds ~300-1500ms latency per unique query
- Use sparingly - prefer state dict data when available
- Good for: UI state, menu detection, NPC positions, dialog boxes, visual-only info
- Be specific in prompts for better accuracy
- Visual keys work ONLY in run() function, NOT in Precondition/Success Condition

Example with visual observation:
```python
add_to_state_schema(
    key="num_npcs",
    vlm_prompt="How many NPC characters (people) are visible? Count only NPCs, not the player. Answer with just a number.",
    return_type=int
)

def run(state):
    # {task_desc}
    # Accessing again uses cached result (no additional VLM call)
    if state["num_npcs"] > 0:
        return 'up'  # Avoid NPCs

    return 'right'  # Navigate
```

Standard policy (no visual observations):''' if self.config.include_visual_observation_examples else 'Standard policy:'}
```python
def run(state):
    # {task_desc}
    # Implementation
    return action  # Single action string like 'up', 'down', 'left', 'right', 'a', 'b', or list of up to three actions like ['up', 'a'] or ['up', 'a', 'a']
```

Note (Using log() for debugging):
You can use the built-in log() function to record debug messages during code execution.

IMPORTANT: log is a BUILT-IN function already available in your code environment.
- DO NOT import it
- DO NOT define it yourself
- Just call it directly within your run() function
- Logs are shown in the next prompt under EXECUTION LOGS section
- Use for debugging stuck situations, tracking decisions, or noting game dialog
- Prefer f-strings for formatted messages

"""

    def build_subtask_prompt(
        self,
        main_milestone: Dict[str, Any],
        current_subtask: Optional[Dict[str, Any]],
        situation: str,
        state: Dict[str, Any],
        milestone_manager: Any = None,
        subtask_manager: Any = None,
        # 추가 파라미터들 (재사용 섹션용)
        execution_error: Optional[Dict[str, Any]] = None,
        previous_code: str = "",
        knowledge_base: Any = None,
        previous_analyses: Optional[List[Tuple[int, str]]] = None,
        execution_logs: Optional[List[Tuple[int, str]]] = None,
        include_state_schema: bool = True,
        num_screenshots: int = 1
    ) -> str:
        """
        Build unified prompt for subtask-based code generation

        Now reuses section builders from build_prompt for consistency!

        Args:
            main_milestone: Main milestone dict
            current_subtask: Current subtask dict (or None)
            situation: Situation string (SUCCESS_ACHIEVED, PRECONDITION_FAILED, STUCK, NORMAL)
            state: Current game state
            milestone_manager: MilestoneManager instance for formatting milestones
            subtask_manager: SubtaskManager instance (optional)
            execution_error: Execution error info (optional)
            previous_code: Previous code (optional, for stuck situations)
            knowledge_base: Knowledge base text (optional)
            previous_analyses: Previous ANALYSIS sections (optional)
            execution_logs: Execution logs from code runs (optional)
            include_state_schema: Whether to include state schema (default: True)

        Returns:
            str: Complete prompt for VLM
        """
        sections = []

        # 1. System instruction (재사용)
        sections.append(self.build_system_instruction())
        sections.append("")

        # 1.5. Environment introduction
        sections.append(self.build_environment_section())
        sections.append("")

        # 2. Visual note (재사용)
        sections.append(self.build_visual_note_section(num_screenshots))
        sections.append("")

        # 3. Main milestone (강화 - milestone 중요성 강조)
        sections.append(f"""🎯 YOUR PRIMARY GOAL: [{main_milestone['description']}]

CRITICAL: This is your main objective. ALL subtasks must directly contribute to achieving this milestone.
- If a subtask doesn't lead toward this milestone, STOP and create a different subtask
- If you're stuck on the same subtask for multiple steps without milestone progress, change your approach
- Subtasks are tools to achieve the milestone - don't get lost in subtasks that don't help
- Always ask: "Is my current subtask moving me closer to completing this milestone?"
""")
        sections.append("")

        # 4. Current subtask info (새 메서드)
        sections.append(self.build_subtask_info_section(current_subtask))
        sections.append("")

        # 5. Situation (새 메서드)
        sections.append(self.build_situation_section(situation))
        sections.append("")

        # 6. Current game state (재사용)
        from utils.state_formatter import convert_state_to_dict, filter_state_for_llm
        import json

        formatted = convert_state_to_dict(state)
        filtered = filter_state_for_llm(formatted)
        formatted_state = json.dumps(filtered, indent=2, default=str)
        sections.append(self.build_game_state_section(formatted_state))
        sections.append("")

        # 7. Recent milestones and subtasks (기존 + subtask_manager 추가)
        recent_milestones_text = self._format_recent_milestones(state, milestone_manager, subtask_manager)
        sections.append(recent_milestones_text)
        sections.append("")

        # 8. Knowledge Base (재사용 - if available)
        if knowledge_base:
            sections.append(self.build_knowledge_base_section(knowledge_base))
            sections.append("")

        # 8.5. Previous analyses (재사용 - if available)
        if previous_analyses and len(previous_analyses) > 0:
            sections.append(self.build_previous_analyses_section(previous_analyses))
            sections.append("")

        # 9. Execution error (재사용 - if available)
        if execution_error:
            sections.append(self.build_execution_error_section(execution_error))
            sections.append("")

        # 10. Previous code (재사용 - if stuck or available)
        if previous_code:
            sections.append(self.build_previous_code_section(previous_code))
            sections.append("")

        # 11. Execution logs (재사용 - if available)
        if self.config.include_execution_logs and execution_logs and len(execution_logs) > 0:
            sections.append(self.build_execution_logs_section(execution_logs))
            sections.append("")

        # 12. State schema (재사용 - 선택적)
        if include_state_schema:
            sections.append(self.build_state_schema_section())
            sections.append("")

        # 13. Clock example (항상 표시)
        sections.append(self.build_clock_example_section())
        sections.append("")

        # 14. Response structure (subtask용)
        sections.append(self.build_subtask_response_structure(main_milestone, current_subtask))

        return "\n".join(sections)

    def _format_recent_milestones(
        self,
        state: Dict[str, Any],
        milestone_manager: Any = None,
        subtask_manager: Any = None
    ) -> str:
        """
        Format recent successful subtasks AND milestones for context (time-ordered)

        Args:
            state: Game state with milestones
            milestone_manager: MilestoneManager instance (optional)
            subtask_manager: SubtaskManager instance (optional)

        Returns:
            Formatted string of recent subtasks and milestones
        """
        all_items = []

        # 1. Get completed milestones from state
        milestones = state.get('milestones', {})
        if milestones:
            for mid, mdata in milestones.items():
                if isinstance(mdata, dict) and mdata.get('completed', False):
                    timestamp = mdata.get('timestamp', 0)
                    description = mid
                    condition = ''
                    if milestone_manager:
                        milestone_info = milestone_manager.get_milestone_info(mid)
                        description = milestone_info.get('description', mid)
                        condition = milestone_info.get('condition', '')

                    all_items.append({
                        'type': 'milestone',
                        'description': description,
                        'condition': condition,
                        'timestamp': timestamp
                    })

        # 2. Get completed subtasks from subtask_manager
        if subtask_manager:
            completed_subtasks = subtask_manager.get_recent_completed_subtasks(count=10)
            for subtask in completed_subtasks:
                all_items.append({
                    'type': 'subtask',
                    'description': subtask['description'],
                    'condition': subtask.get('success_condition', ''),
                    'timestamp': subtask['timestamp']
                })

        if not all_items:
            return "RECENT SUCCESSFUL SUBTASKS: None yet"

        # 3. Sort by timestamp (most recent first) and take last 5
        all_items.sort(key=lambda x: x['timestamp'], reverse=True)
        recent = all_items[:5]

        # 4. Format output
        lines = ["RECENT SUCCESSFUL SUBTASKS (time-ordered):"]
        lines.append("⚠️ Do not completely trust these subtasks - success conditions are often incorrect or incomplete.")
        for item in recent:
            lines.append(f"  {item['description']}")
            if item.get('condition'):
                lines.append(f"    → {item['condition']}")

        return "\n".join(lines)