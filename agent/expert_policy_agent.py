#!/usr/bin/env python3
"""
Expert Policy Agent - Executes pre-trained policies from milestone trainer

This agent loads and executes successful policies stored in
.milestone_trainer_cache/successful_policies/ directory.
No LLM calls - pure policy execution for fast, cost-free inference.
"""

import os
import glob
import signal
import time
from typing import Dict, Any

from utils.tool_loader import load_tools
from utils.milestone_manager import MilestoneManager
from utils.state_formatter import convert_state_to_dict
from utils.vlm_state import State, get_global_schema_registry, add_to_state_schema
from utils.vlm_caller import VLMCaller


class ExpertPolicyAgent:
    """Agent that executes pre-trained expert policies"""

    def __init__(self, policies_dir: str = ".milestone_trainer_cache/successful_policies",
                 start_from_milestone: str = None):
        """
        Initialize ExpertPolicyAgent

        Args:
            policies_dir: Directory containing policy .py files (default: .milestone_trainer_cache/successful_policies)
            start_from_milestone: Optional milestone ID to start from. All milestones up to (and including) this will be marked as completed.
        """
        self.policies_dir = policies_dir
        self.policies = {}  # {milestone_id: code_string}

        # Initialize milestone manager
        self.milestone_manager = MilestoneManager()

        # Register custom milestones (same as CodeAgent)
        self._register_custom_milestones()

        # Load tools (same as CodeAgent)
        self.tools = load_tools('tools')
        print(f"📦 Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

        # Initialize VLM caller (optional, for policies that use vision)
        vlm_model = os.getenv("VLM_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
        self.vlm_caller = VLMCaller(model=vlm_model)

        # Track last action for prev_action in state (same as CodeAgent)
        self._last_action = None
        self._last_facing = None

        # Custom milestone completion tracking (client-side only, same as CodeAgent)
        # NOTE: ExpertPolicyAgent doesn't save/load from file (memory only)
        self.custom_milestone_completions = {}

        # Initialize starting milestone if provided
        if start_from_milestone:
            self._initialize_starting_milestone(start_from_milestone)

        # Load all policies
        self._load_policies()

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

    def _load_policies(self):
        """
        Load all policy files from policies directory

        Each .py file should be named {MILESTONE_ID}.py and contain a run(state) function.
        """
        if not os.path.exists(self.policies_dir):
            print(f"⚠️ Policies directory not found: {self.policies_dir}")
            print(f"   No policies loaded. Agent will return 'b' for all milestones.")
            return

        policy_files = glob.glob(os.path.join(self.policies_dir, "*.py"))

        for file_path in policy_files:
            filename = os.path.basename(file_path)
            milestone_id = filename[:-3]  # Remove .py extension

            try:
                with open(file_path, 'r') as f:
                    code = f.read()

                self.policies[milestone_id] = code

            except Exception as e:
                print(f"⚠️ Failed to load policy {filename}: {e}")

        # Print summary
        policy_list = list(self.policies.keys())
        if len(policy_list) > 5:
            display_list = policy_list[:5] + ['...']
        else:
            display_list = policy_list

        print(f"📦 Loaded {len(self.policies)} policies: {display_list}")

    def step(self, game_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Execute policy for current milestone

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
            'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'no_op'
        """
        # Add prev_action to game_state (same as CodeAgent)
        game_state["prev_action"] = (
            self._last_action if self._last_action else "no_op"
        )

        # Update facing based on prev_action (direction keys update facing)
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

        # Check custom milestone completions BEFORE selecting next milestone
        # This ensures completions from previous step are reflected
        if self._last_action:
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

        # Find corresponding policy
        code = self.policies.get(milestone_id)

        if not code:
            print(f"⚠️ No policy found for milestone: {milestone_id}")
            print(f"   Description: {next_milestone_info.get('description', 'N/A')}")
            return {'action': 'b'}

        # Track current policy code for timeout handling
        self._current_policy_code = code

        # Execute policy
        print(f"▶️ Executing policy: {milestone_id}")
        action = self._execute_code(code, game_state)

        # Update last action for next step (same as CodeAgent)
        self._last_action = action

        return {'action': action}

    def get_current_milestone_id(self):
        """Get currently executing milestone ID"""
        return getattr(self, '_current_milestone_id', None)

    def get_current_policy_code(self):
        """Get currently executing policy code"""
        return getattr(self, '_current_policy_code', None)

    def _check_custom_milestones(self, game_state, action):
        """
        Check and track custom milestone completions (client-side only)

        NOTE: ExpertPolicyAgent doesn't save to file (CodeAgent difference)

        Args:
            game_state: Current game state from server
            action: Action that was just executed
        """
        # Use client-side milestone tracking only
        milestones = self.custom_milestone_completions

        for custom in self.milestone_manager.custom_milestones:
            milestone_id = custom["id"]

            # Skip if already completed (check custom completions only, ignore server)
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
                    # NOTE: ExpertPolicyAgent doesn't log policy or save to file
                    # (CodeAgent's _log_milestone_policy() and _save_custom_milestone_completions() removed)
            except Exception as e:
                print(f"⚠️ Error checking custom milestone {milestone_id}: {e}")

    def _execute_code(self, code: str, state: Dict[str, Any]) -> str:
        """
        Execute policy code

        NOTE: This is copied from CodeAgent._execute_code with simplifications.
        If CodeAgent._execute_code has critical bug fixes, sync manually.

        TODO: Future refactoring - extract to utils/code_executor.py

        Args:
            code: Python code string with run(state) function
            state: Game state dict (raw format)

        Returns:
            action string (e.g., 'up', 'a')
        """
        try:
            # Clear schema registry before each execution
            schema_registry = get_global_schema_registry()
            schema_registry.clear()

            # Convert state to same format LLM saw in prompt
            formatted_state = convert_state_to_dict(state)

            # Log function (simplified - just print)
            def log(message: str):
                """Log message during code execution"""
                print(f"📝 {message}")

            # Create execution environment with add_to_state_schema, log, and tools
            exec_globals = {
                'add_to_state_schema': add_to_state_schema,
                'log': log,
                **self.tools  # Inject tool functions (handle_battle, navigate_ui, etc.)
            }

            # Execute code with 15-second timeout
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution exceeded 15 seconds")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)  # 15 second timeout

            # Execute code (timeout will cover exec + State creation + run() call)
            exec(code, exec_globals)

            # Create State object with VLM support
            screenshot = state.get('frame')
            state_obj = State(
                base_data=formatted_state,
                schema_registry=schema_registry,
                vlm_caller=lambda screenshot, prompt, return_type: self.vlm_caller.call(
                    screenshot, prompt, return_type
                ),
                screenshot=screenshot
            )

            # Call run function with State object
            if 'run' not in exec_globals:
                # Cancel alarm if no run function found
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

                print(f"⚠️ No 'run' function found in code, using 'b'")
                return 'b'

            action = exec_globals['run'](state_obj)

            # Cancel alarm after run() completes successfully
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            # Log VLM accesses for debugging
            vlm_log = state_obj.get_vlm_access_log()
            if vlm_log:
                print(f"🔍 VLM queries made: {len(vlm_log)}")
                for entry in vlm_log:
                    print(f"   {entry['key']}: {entry['result']} ({entry['return_type']})")

            # Validate action (support single action or list of actions)
            valid_actions = ['a', 'b', 'start', 'select', 'up', 'down', 'left', 'right', 'no_op']

            # Support single action (str)
            if isinstance(action, str):
                if action.lower() in valid_actions:
                    return action.lower()
                else:
                    print(f"⚠️ Invalid action returned: {action}, using 'b'")
                    return 'b'

            # Support multiple actions (list)
            elif isinstance(action, list):
                if len(action) == 0:
                    print(f"⚠️ Empty action list returned, using 'b'")
                    return 'b'

                # Validate each action in the list
                validated_actions = []
                for act in action:
                    if isinstance(act, str) and act.lower() in valid_actions:
                        validated_actions.append(act.lower())
                    else:
                        print(f"⚠️ Invalid action in list: {act}, using 'b'")
                        return 'b'

                return validated_actions  # Return list of validated actions

            # Invalid type
            else:
                print(f"⚠️ Invalid action type returned: {type(action).__name__} (expected str or list), using 'b'")
                return 'b'

        except TimeoutError as e:
            # Cancel alarm on timeout
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            print(f"⏰ Code execution timeout: {str(e)}")
            print(f"Code:\n{code}")

            raise  # Re-raise to propagate to client

        except Exception as e:
            # Cancel alarm on any exception
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except:
                pass  # If signal cleanup fails, continue with error handling

            import traceback
            print(f"❌ Code execution error: {str(e)}")
            print(f"Code:\n{code}")
            traceback.print_exc()

            return 'no_op'  # No-op action on error (don't make things worse)
