#!/usr/bin/env python3
"""
CodeAgent wrapper for PokemonEnv integration

Bridges CodeAgent and PokemonEnv by handling:
- Action format conversion
- State formatting
- Initialization
"""

import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class CodeAgentEnvWrapper:
    """
    Wrapper that connects CodeAgent to PokemonEnv

    Supports two modes:
    - Normal mode: Multiple code generations per episode (CodeAgent decides when)
    - Curriculum mode: One code generation per episode with previous attempt feedback

    Features:
    - VLM support: Generated policies can use add_to_state_schema() for visual queries
    - Tool injection: All tools from tools/ directory are available to policies
    - State object: Policies receive State object with lazy VLM evaluation

    Usage:
        # Normal mode
        wrapper = CodeAgentEnvWrapper(model="gpt-5")
        obs, info = env.reset()  # obs=PIL Image, info=state dict
        action_dict = wrapper.get_action(info)  # Pass state dict from info

        # Curriculum mode with VLM support
        wrapper = CodeAgentEnvWrapper(model="gpt-5", curriculum_mode=True)
        obs, info = env.reset()  # obs=PIL Image, info=state dict
        code = wrapper.start_episode(info, attempt_history)  # Pass state dict
        action_dict = wrapper.get_action(info)  # Pass state dict (creates State object)

        # Generated policy can use VLM:
        # def run(state):
        #     add_to_state_schema("is_in_menu", "Is the menu open?", bool)
        #     if state["is_in_menu"]:  # VLM query happens here!
        #         return "b"
        #     return "a"
    """

    def __init__(self, model: str = "gpt-5", enable_logging: bool = False, curriculum_mode: bool = False, active_knowledge_search: bool = False, enable_policy_logging: bool = True):
        """
        Initialize CodeAgent wrapper

        Args:
            model: LLM model to use
            enable_logging: Enable detailed logging
            curriculum_mode: If True, generate policy once per episode with attempt history
            active_knowledge_search: If True, use tool calling for LLM-driven search (Claude only); if False, use location-based automatic search
            enable_policy_logging: If True, enable log() function in generated policies (default: True)
        """
        self.model = model
        self.enable_logging = enable_logging
        self.curriculum_mode = curriculum_mode
        self.active_knowledge_search = active_knowledge_search
        self.enable_policy_logging = enable_policy_logging

        if not enable_logging:
            # Suppress verbose logging
            logging.getLogger('agent.code_agent').setLevel(logging.WARNING)

        # Lazy import to avoid circular dependencies
        from agent.code_agent import CodeAgent

        # Initialize CodeAgent
        try:
            self.agent = CodeAgent(model=model)
            logger.info(f"✅ CodeAgent initialized with model: {model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CodeAgent: {e}")
            raise

        # Track last action
        self.last_action_str = None

        # Curriculum mode: Episode-level policy
        self.episode_policy_code = None
        self.episode_policy_fn = None
        self.episode_started = False
        self.policy_logs = []  # Store logs from policy execution

        # Load tools from tools/ directory (same as CodeAgent)
        from utils.tool_loader import load_tools
        self.tools = load_tools('tools')
        logger.info(f"📦 Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

        # VLM support (same as CodeAgent)
        from utils.vlm_caller import VLMCaller
        from utils.vlm_state import get_global_schema_registry

        vlm_model = os.getenv("VLM_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
        self.vlm_caller = VLMCaller(model=vlm_model)
        self.schema_registry = get_global_schema_registry()
        logger.info(f"🔍 VLM enabled with model: {vlm_model}")

    def start_episode(self, state: Dict[str, Any], attempt_history=None, milestone_id: Optional[str] = None) -> str:
        """
        Start new episode in curriculum mode with two-phase approach:
        Phase 1: Knowledge gathering (LLM uses tools to search)
        Phase 2: Code generation (with gathered knowledge)

        Args:
            state: Initial state dict from environment info
            attempt_history: Optional AttemptHistory with previous failed attempts
            milestone_id: Optional specific milestone ID to train on (if None, uses next uncompleted)

        Returns:
            Generated policy code as string

        Raises:
            ValueError: If not in curriculum mode
        """
        if not self.curriculum_mode:
            raise ValueError("start_episode() is only for curriculum mode. Set curriculum_mode=True in __init__()")

        # Clear schema registry before new episode (same as CodeAgent)
        self.schema_registry.clear()

        # Reload tools (same as CodeAgent does for code generation)
        from utils.tool_loader import load_tools
        self.tools = load_tools('tools', force_reload=True)
        logger.info(f"🔄 Reloaded {len(self.tools)} tools for policy generation")

        # Format state
        from utils.state_formatter import convert_state_to_dict, filter_state_for_llm
        formatted_state = convert_state_to_dict(state)
        filtered_state = filter_state_for_llm(formatted_state)

        # Get milestone info
        if milestone_id:
            # Use specified milestone
            next_milestone_info = self._get_milestone_info_by_id(milestone_id)
            if not next_milestone_info:
                logger.warning(f"⚠️ Milestone '{milestone_id}' not found, using auto-selection")
                augmented_milestones = self.agent._get_augmented_milestones(state)
                next_milestone_info = self.agent.milestone_manager.get_next_milestone_info(augmented_milestones)
        else:
            # Auto-select next uncompleted milestone
            augmented_milestones = self.agent._get_augmented_milestones(state)
            next_milestone_info = self.agent.milestone_manager.get_next_milestone_info(augmented_milestones)

        # ============================================================
        # PHASE 1: Knowledge Gathering
        # ============================================================
        gathered_knowledge = []
        if self.agent.use_knowledge_base:
            if self.active_knowledge_search:
                # Active: LLM-driven tool calling (Claude only)
                gathered_knowledge = self._gather_knowledge_with_tools(
                    filtered_state=filtered_state,
                    milestone_info=next_milestone_info,
                    attempt_history=attempt_history
                )
            else:
                # Passive: Location-based automatic search (legacy)
                gathered_knowledge = self._gather_knowledge_by_location(filtered_state)

        # ============================================================
        # PHASE 2: Code Generation (existing approach)
        # ============================================================
        from utils.curriculum_prompt_builder import CurriculumPromptBuilder
        curriculum_builder = CurriculumPromptBuilder()

        prompt = curriculum_builder.build_prompt(
            current_state=filtered_state,
            milestone_info=next_milestone_info or {},
            knowledge_base=gathered_knowledge,  # ← Use gathered knowledge!
            attempt_history=attempt_history,
            tools_description=self._get_tools_description(),
            enable_policy_logging=self.enable_policy_logging
        )

        # Get screenshots
        current_screenshot = state.get('frame')
        screenshots = curriculum_builder.get_screenshots(
            current_screenshot=current_screenshot,
            attempt_history=attempt_history
        )

        # Convert screenshots to format for LLM (list of (step_count, PIL.Image))
        frames_to_send = [(i, frame) for i, frame in enumerate(screenshots)]

        # Call LLM (reuse CodeAgent's method) with timing and logging
        logger.info(f"   🤖 Calling LLM with {len(frames_to_send)} screenshots...")
        import time
        start_time = time.time()
        response = self.agent._call_llm(prompt, frames_to_send)
        duration = time.time() - start_time

        # Log interaction (same as CodeAgent.generate_policy_code does)
        self.agent.llm_logger.log_interaction(
            interaction_type="curriculum_code_generation",
            prompt=prompt,
            response=response,
            duration=duration,
            model_info={"model": self.agent.model, "tokens": {"prompt": 0, "completion": 0}}
        )

        # Extract code (reuse CodeAgent's method)
        self.episode_policy_code = self.agent._extract_code(response)

        # Compile policy function for execution
        self._compile_episode_policy()

        self.episode_started = True

        logger.info(f"   📝 Generated policy ({len(self.episode_policy_code)} chars)")
        if attempt_history and attempt_history.get_attempt_count() > 0:
            logger.info(f"   🔄 Learning from {attempt_history.get_attempt_count()} previous attempt(s)")

        return self.episode_policy_code

    def _get_milestone_info_by_id(self, milestone_id: str) -> Optional[Dict[str, Any]]:
        """
        Get milestone info by ID

        Args:
            milestone_id: Milestone ID to look up

        Returns:
            Milestone info dict or None if not found
        """
        # All milestones are stored as dicts in custom_milestones
        custom_milestones = self.agent.milestone_manager.custom_milestones
        for milestone in custom_milestones:
            if milestone.get('id') == milestone_id:
                return {
                    'id': milestone.get('id'),
                    'description': milestone.get('description', 'No description'),
                    'check_description': milestone.get('check_description', None)
                }

        # Fallback: Check ALL_MILESTONES (static list)
        all_milestones = self.agent.milestone_manager.ALL_MILESTONES
        for milestone in all_milestones:
            if milestone.get('id') == milestone_id:
                return {
                    'id': milestone.get('id'),
                    'description': milestone.get('description', 'No description'),
                    'check_description': milestone.get('check_description', None)
                }

        return None

    def _get_knowledge_tools_schema(self) -> List[Dict]:
        """Get Claude SDK tool schema for knowledge base search"""
        return [
            {
                "name": "search_knowledge",
                "description": """Search the game knowledge base for information.

Use this to find:
- Navigation paths between locations
- Item locations (Pokeballs, HMs, etc.)
- NPC locations and information
- Game mechanics and tips
- Previously learned information

The knowledge base contains information learned during gameplay.

Examples:
- search_knowledge("Route 103 navigation")
- search_knowledge("Pokeball location Oldale Town")
- search_knowledge("Rustboro City path from Petalburg")
- search_knowledge("gym leader locations")""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for (be specific)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default 20)",
                            "default": 20
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def _build_knowledge_gathering_prompt(
        self,
        filtered_state: Dict[str, Any],
        milestone_info: Dict[str, Any],
        attempt_history
    ) -> str:
        """Build prompt for Phase 1: Knowledge gathering"""
        lines = []

        lines.append("You are preparing to complete a milestone in Pokemon Emerald.")
        lines.append("Before generating policy code, you need to gather relevant knowledge.")
        lines.append("")

        lines.append("## MILESTONE")
        lines.append(f"**Goal:** {milestone_info.get('description', 'No description')}")
        if 'check_description' in milestone_info:
            lines.append(f"**Completion condition:** {milestone_info['check_description']}")
        lines.append("")

        lines.append("## CURRENT STATE")
        player = filtered_state.get('player', {})
        if player:
            lines.append(f"- Location: {player.get('location', 'unknown')}")
            position = player.get('position', {})
            lines.append(f"- Position: ({position.get('x', 0)}, {position.get('y', 0)})")

            # Show inventory if available
            inventory = player.get('inventory', {})
            if inventory:
                items = [f"{k}: {v}" for k, v in inventory.items() if v > 0]
                if items:
                    lines.append(f"- Inventory: {', '.join(items[:5])}")

            # Show party if available
            party = player.get('party', [])
            if party:
                pokemon_list = []
                for p in party[:3]:
                    species = p.get('species', 'Unknown')
                    level = p.get('level', 0)
                    pokemon_list.append(f"{species} (Lv{level})")
                lines.append(f"- Party: {', '.join(pokemon_list)}")

        lines.append("")

        if attempt_history and attempt_history.get_attempt_count() > 0:
            latest = attempt_history.get_latest_attempt()
            lines.append("## PREVIOUS ATTEMPT FAILED")
            lines.append(f"- Reason: {latest['reason']}")
            lines.append(f"- Steps taken: {latest['steps']}")
            final_state = latest.get('final_state', {})
            final_player = final_state.get('player', {})
            if final_player:
                lines.append(f"- Final location: {final_player.get('location', 'unknown')}")
            lines.append("")

        lines.append("## YOUR TASK")
        lines.append("Search for relevant knowledge that will help complete this milestone.")
        lines.append("")
        lines.append("Use the `search_knowledge` tool to find:")
        lines.append("- Navigation paths to reach your goal")
        lines.append("- Item/NPC locations mentioned in the milestone")
        lines.append("- Game mechanics relevant to this task")
        lines.append("- Any previously learned information")
        lines.append("")
        lines.append("Make 2-4 targeted searches. Be specific in your queries.")
        lines.append("When you have enough information, respond with: 'I have gathered sufficient knowledge.'")

        return "\n".join(lines)

    def _gather_knowledge_with_tools(
        self,
        filtered_state: Dict[str, Any],
        milestone_info: Dict[str, Any],
        attempt_history
    ) -> List[Dict]:
        """
        Phase 1: Let LLM gather relevant knowledge using tool calling

        Returns:
            List of knowledge entries gathered by LLM
        """
        logger.info("   📚 Phase 1: Knowledge Gathering")

        # Build knowledge gathering prompt
        prompt = self._build_knowledge_gathering_prompt(
            filtered_state=filtered_state,
            milestone_info=milestone_info,
            attempt_history=attempt_history
        )

        # Initialize conversation
        messages = [{"role": "user", "content": prompt}]

        gathered_knowledge = []
        max_searches = 5  # Limit to prevent infinite loops
        search_count = 0

        import time
        phase1_start = time.time()

        while search_count < max_searches:
            try:
                # Call Claude with tool
                response = self.agent.client.messages.create(
                    model=self.agent.model,
                    max_tokens=2048,
                    messages=messages,
                    tools=self._get_knowledge_tools_schema()
                )

                # Add assistant response to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Check for tool use
                has_tool_call = False
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        has_tool_call = True

                        if content_block.name == "search_knowledge":
                            query = content_block.input["query"]
                            limit = content_block.input.get("limit", 20)

                            logger.info(f"      🔍 Searching: '{query}' (limit={limit})")

                            # Search knowledge base (directly with query text)
                            results = self.agent.knowledge_base.get_by_keywords(
                                query_text=query,
                                limit=limit,
                                always_include_recent=0  # Prioritize LCS relevance over recency
                            )

                            # Store gathered knowledge (deduplicate by content)
                            existing_contents = {k.get('content') for k in gathered_knowledge}
                            for result in results:
                                if result.get('content') not in existing_contents:
                                    gathered_knowledge.append(result)

                            # Format results for LLM
                            if results:
                                result_text = f"Found {len(results)} knowledge entries:\n\n"
                                for i, entry in enumerate(results[:10], 1):  # Show first 10
                                    result_text += f"{i}. {entry.get('content', '')}\n"
                                logger.info(f"         ✅ Found {len(results)} entries")
                            else:
                                result_text = "No knowledge entries found for this query."
                                logger.info(f"         ❌ No entries found")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": result_text
                            })

                            search_count += 1

                if not has_tool_call:
                    # LLM is done gathering knowledge
                    logger.info(f"      ✅ LLM finished gathering knowledge")
                    break

                # Add tool results to conversation
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

            except Exception as e:
                logger.error(f"      ❌ Error during knowledge gathering: {e}")
                break

        phase1_duration = time.time() - phase1_start

        logger.info(f"   📚 Phase 1 Complete: {len(gathered_knowledge)} unique entries gathered from {search_count} searches ({phase1_duration:.1f}s)")

        return gathered_knowledge

    def _gather_knowledge_by_location(self, filtered_state: Dict[str, Any]) -> List[Dict]:
        """
        Legacy approach: Automatic location-based knowledge retrieval

        Returns:
            List of knowledge entries based on player location
        """
        logger.info("   📚 Phase 1: Knowledge Gathering (location-based)")

        # Extract location from state
        location = filtered_state.get('player', {}).get('location', '')

        if not location:
            logger.info("      ⚠️ No location found in state")
            return []

        logger.info(f"      🔍 Auto-searching location: '{location}'")

        # Search knowledge base (limit to 50 for better prompt efficiency)
        results = self.agent.knowledge_base.get_by_keywords(
            query_text=location,
            limit=50,
            always_include_recent=0  # Prioritize LCS relevance over recency
        )

        logger.info(f"   📚 Phase 1 Complete: {len(results)} entries found for '{location}'")

        return results

    def _get_tools_description(self) -> str:
        """Get tools documentation for prompt"""
        lines = []
        for tool_name, tool_fn in self.tools.items():
            if hasattr(tool_fn, '__doc__') and tool_fn.__doc__:
                lines.append(f"**{tool_name}**: {tool_fn.__doc__.strip()}")
            else:
                lines.append(f"**{tool_name}**: (no documentation)")
        return "\n".join(lines)

    def _compile_episode_policy(self):
        """Compile episode policy code to executable function"""
        try:
            # Create execution environment with tools and VLM support (same as CodeAgent._execute_code)
            from utils.vlm_state import add_to_state_schema

            exec_globals = {
                **self.tools,  # Inject tool functions (e.g., find_path_action)
                'add_to_state_schema': add_to_state_schema  # Inject VLM schema function
            }

            # Add log() function if enabled (same pattern as CodeAgent)
            if self.enable_policy_logging:
                def log(message: str):
                    """Record a message during code execution"""
                    self.policy_logs.append(str(message))
                exec_globals['log'] = log

            exec(self.episode_policy_code, exec_globals)

            # Get run function
            if 'run' in exec_globals:
                self.episode_policy_fn = exec_globals['run']
            else:
                logger.error("⚠️ No 'run' function in generated code!")
                self.episode_policy_fn = None

        except Exception as e:
            logger.error(f"❌ Failed to compile policy: {e}")
            self.episode_policy_fn = None

    def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action from CodeAgent

        Args:
            state: State dict from environment info

        Returns:
            Action dict: {'action': 'up'} or {'action': ['up', 'a']}
        """
        try:
            if self.curriculum_mode:
                # Curriculum mode: Use pre-generated policy
                if not self.episode_started or self.episode_policy_fn is None:
                    raise ValueError("Call start_episode() before get_action() in curriculum mode")

                # Clear logs before execution (if logging enabled)
                if self.enable_policy_logging:
                    self.policy_logs = []

                # Create State object with VLM support (same as CodeAgent)
                from utils.vlm_state import State
                from utils.state_formatter import convert_state_to_dict

                screenshot = state.get('frame')
                # Convert state to same format as shown in prompt (IMPORTANT: LLM saw converted format!)
                formatted_state = convert_state_to_dict(state)
                state_obj = State(
                    base_data=formatted_state,
                    schema_registry=self.schema_registry,
                    vlm_caller=lambda screenshot, prompt, return_type: self.vlm_caller.call(
                        screenshot, prompt, return_type
                    ),
                    screenshot=screenshot
                )

                # Execute pre-generated policy with State object (VLM support!)
                action = self.episode_policy_fn(state_obj)
                action_dict = {'action': action}

                # Log VLM accesses for debugging (same as CodeAgent)
                vlm_log = state_obj.get_vlm_access_log()
                if vlm_log and self.enable_logging:
                    logger.info(f"🔍 VLM queries made: {len(vlm_log)}")
                    for entry in vlm_log:
                        logger.info(f"   {entry['key']}: {entry['result']} ({entry['return_type']})")

                # Display policy logs (same as CodeAgent)
                if self.enable_policy_logging and self.policy_logs and self.enable_logging:
                    for msg in self.policy_logs:
                        logger.info(f"📝 {msg}")

            else:
                # Normal mode: CodeAgent decides when to regenerate
                action_dict = self.agent.step(state)

            # Track last action
            action = action_dict.get('action', 'b')
            if isinstance(action, list):
                self.last_action_str = action[-1] if action else 'b'
            else:
                self.last_action_str = action

            return action_dict

        except Exception as e:
            import traceback
            logger.error(f"❌ Error getting action: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            # Re-raise to terminate episode immediately
            raise

    def action_dict_to_int(self, action_dict: Dict[str, Any], env) -> int:
        """
        Convert CodeAgent action dict to PokemonEnv action int

        Args:
            action_dict: {'action': 'up'} or {'action': ['up', 'a']}
            env: PokemonEnv instance

        Returns:
            Action int (0-9)
        """
        action = action_dict.get('action', 'b')

        # If list, take first action (PokemonEnv supports single action per step)
        if isinstance(action, list):
            action = action[0] if action else 'b'

        # Convert to int
        action_str = str(action).upper()
        action_map = {
            "A": 0, "B": 1, "START": 2, "SELECT": 3,
            "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7,
            "L": 8, "R": 9
        }

        return action_map.get(action_str, 1)  # Default: B

    def get_last_action_str(self) -> str:
        """Get last action as string for prev_action"""
        return self.last_action_str or "no_op"

    def reset(self):
        """Reset wrapper state (for new episode)"""
        self.last_action_str = None

        # Curriculum mode: Reset episode state
        if self.curriculum_mode:
            self.episode_policy_code = None
            self.episode_policy_fn = None
            self.episode_started = False

        # CodeAgent reset if needed
        if hasattr(self.agent, 'reset'):
            self.agent.reset()

    def get_episode_policy_code(self) -> Optional[str]:
        """Get current episode policy code (curriculum mode only)"""
        return self.episode_policy_code

    def get_policy_logs(self) -> List[str]:
        """Get logs from last policy execution"""
        return self.policy_logs.copy()


# Utility function for easy creation
def create_code_agent_wrapper(
    model: str = "gpt-5",
    enable_logging: bool = False,
    curriculum_mode: bool = False,
    active_knowledge_search: bool = False,
    enable_policy_logging: bool = True
) -> CodeAgentEnvWrapper:
    """
    Factory function to create CodeAgentEnvWrapper

    Args:
        model: LLM model
        enable_logging: Enable logging
        curriculum_mode: Enable curriculum learning mode
        active_knowledge_search: Use tool calling for LLM-driven knowledge search (Claude only)
        enable_policy_logging: Enable log() function in generated policies (default: True)

    Returns:
        CodeAgentEnvWrapper instance
    """
    return CodeAgentEnvWrapper(
        model=model,
        enable_logging=enable_logging,
        curriculum_mode=curriculum_mode,
        active_knowledge_search=active_knowledge_search,
        enable_policy_logging=enable_policy_logging
    )
