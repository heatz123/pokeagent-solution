#!/usr/bin/env python3
"""
Battle Handler tool for CodeAgent

Provides simple battle automation for Pokemon battles.
This file is auto-loaded - no import needed in LLM code!
"""

import logging

logger = logging.getLogger(__name__)


def handle_battle(state, strategy: str = "fight") -> str:
    """
    Automate Pokemon battle with intelligent move selection.

    Args:
        state: State object with VLM support
        strategy: Battle strategy to use
            - "fight": Simply press 'a' repeatedly (default)
            - "water_gun": Always use Water Gun move (hardcoded)
            - "run": Navigate to Run option and select it (wild battles only)

    Returns:
        str: Next action ('a', 'b', 'up', 'down', 'left', 'right', 'no_op')

    Example:
        # Fight strategy: Use strongest attack
        action = handle_battle(state, strategy="fight")
        return action

        # Water Gun strategy: Always use Water Gun
        action = handle_battle(state, strategy="water_gun")
        return action

        # Run strategy: Navigate to Run menu (uses VLM)
        action = handle_battle(state, strategy="run")
        return action

    Battle flow:
    - fight: Simply press 'a' (no VLM for move selection)
    - water_gun: in_battle → always use Water Gun, not in_battle → 'a'
    - run: Navigate to Run option using VLM, then confirm
    """
    try:
        from utils.vlm_state import add_to_state_schema

        # Check if in battle (only VLM call for fight strategy)
        add_to_state_schema(
            key="in_battle",
            vlm_prompt="Is the character currently in a Pokemon battle? Look for battle UI, HP bars, Pokemon sprites. Answer true only if battle screen is visible, false otherwise.",
            return_type=bool,
        )

        if not state["in_battle"]:
            logger.info("Not in battle")
            return "a"

        logger.info(f"Battle detected, strategy: {strategy}")

        if strategy == "fight":
            # Simple fight strategy: just press 'a'
            logger.info("Fight strategy: pressing 'a'")
            return "a"

        elif strategy == "water_gun":
            # Always use Water Gun
            logger.info("Water Gun strategy: using Water Gun")
            return _use_water_gun(state)

        elif strategy == "run":
            # Check if we can run (wild vs trainer battle)
            add_to_state_schema(
                key="can_run_from_battle",
                vlm_prompt="Can the player run from this battle? Wild Pokemon battles allow running, but Trainer battles do not. Look for text like 'Wild POKEMON appeared!' (can run) vs trainer name/sprite (cannot run). Answer true if wild battle, false if trainer battle.",
                return_type=bool,
            )

            can_run = state["can_run_from_battle"]
            logger.info(f"Can run: {can_run}")

            if not can_run:
                # Trainer battle: can't run, switch to fight
                logger.info("Trainer battle detected, switching to fight strategy")
                return "a"

            # Wild battle: navigate to Run option
            logger.info("Wild battle: navigating to Run")
            return _navigate_to_run(state)

        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'a'")
            return "a"

    except Exception as e:
        logger.error(f"Battle handler error: {e}")
        import traceback

        traceback.print_exc()
        return "no_op"


def _navigate_to_run(state) -> str:
    """
    Navigate to Run option in battle menu.

    Args:
        state: State object with VLM support

    Returns:
        str: Action to take
    """
    from utils.vlm_state import add_to_state_schema

    # Check if main battle menu is visible
    add_to_state_schema(
        key="battle_main_menu_visible",
        vlm_prompt="Is the main battle menu visible? (Fight/Pokemon/Bag/Run options). Answer true if you can see these 4 options, false otherwise.",
        return_type=bool,
    )

    if not state["battle_main_menu_visible"]:
        # Menu not visible, press 'a' to advance/open menu
        logger.info("Main menu not visible, pressing 'a'")
        return "a"

    # Check current selection
    add_to_state_schema(
        key="run_selected",
        vlm_prompt="Is the 'Run' option currently highlighted/selected in the battle menu? Look for highlighting or cursor on Run option.",
        return_type=bool,
    )

    if state["run_selected"]:
        # Already on Run, confirm
        logger.info("Run selected, confirming with 'a'")
        return "a"

    # Need to navigate to Run
    # Ask VLM for direction
    add_to_state_schema(
        key="direction_to_run",
        vlm_prompt="To navigate to the 'Run' option in the battle menu, which direction should I press? Answer with one of: 'up', 'down', 'left', 'right'. Run is usually in the bottom-right corner.",
        return_type=str,
    )

    direction = state["direction_to_run"].lower()
    logger.info(f"Navigating {direction} to Run")

    return direction if direction in ["up", "down", "left", "right"] else "down"


# Global cache for water gun navigation
_water_gun_prev_action = None


def _use_water_gun(state) -> str:
    """
    Always use Water Gun move using simple state machine.
    Grid layout: 1 2 / 3 4
    Water Gun is at position 4.
    Sequence: a (enter) -> right (1->2) -> down (2->4) -> a (use)

    Args:
        state: State object with VLM support

    Returns:
        str: Action to take
    """
    global _water_gun_prev_action
    from utils.vlm_state import add_to_state_schema

    # Check if move selection screen is visible
    add_to_state_schema(
        key="move_selection_visible",
        vlm_prompt=(
            "Is the move selection screen visible? "
            "(shows 4 moves with PP counts). "
            "Answer true if you can see the move list, false otherwise."
        ),
        return_type=bool,
    )

    is_move_screen = state["move_selection_visible"]
    logger.info(f"Move selection visible: {is_move_screen}, prev action: {_water_gun_prev_action}")

    if not is_move_screen:
        # Not on move screen, press 'a' to enter
        logger.info("Not on move screen, pressing 'a'")
        _water_gun_prev_action = "a"
        return "a"

    # On move screen - follow state machine
    if _water_gun_prev_action == "a":
        logger.info("Prev was 'a', now pressing 'right'")
        _water_gun_prev_action = "right"
        return "right"
    elif _water_gun_prev_action == "right":
        logger.info("Prev was 'right', now pressing 'down'")
        _water_gun_prev_action = "down"
        return "down"
    elif _water_gun_prev_action == "down":
        logger.info("Prev was 'down', now pressing 'a' to use Water Gun")
        _water_gun_prev_action = "a"
        return "a"
    else:
        # Default/initial state
        logger.info("Initial state, pressing 'right'")
        _water_gun_prev_action = "right"
        return "right"
