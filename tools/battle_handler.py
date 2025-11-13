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
    Automate Pokemon battle with minimal VLM calls.

    Args:
        state: State object with VLM support
        strategy: Battle strategy to use
            - "fight": Keep pressing 'a' to progress battle (default)
                       Advances text, selects Fight, uses first move
            - "run": Navigate to Run option and select it (wild battles only)

    Returns:
        str: Next action ('a', 'b', 'up', 'down', 'left', 'right', 'no_op')

    Example:
        # Fight strategy: Just press 'a' (minimal VLM)
        action = handle_battle(state, strategy="fight")
        return action

        # Run strategy: Navigate to Run menu (uses VLM)
        action = handle_battle(state, strategy="run")
        return action

    Battle flow:
    - fight: in_battle → 'a', not in_battle → 'no_op'
    - run: Navigate to Run option using VLM, then confirm
    """
    try:
        from utils.vlm_state import add_to_state_schema

        # Check if in battle (only VLM call for fight strategy)
        add_to_state_schema(
            key="in_battle",
            vlm_prompt="Is the character currently in a Pokemon battle? Look for battle UI, HP bars, Pokemon sprites. Answer true only if battle screen is visible, false otherwise.",
            return_type=bool
        )

        if not state["in_battle"]:
            logger.info("Not in battle")
            return 'no_op'

        logger.info(f"Battle detected, strategy: {strategy}")

        if strategy == "fight":
            # Simple: just press 'a' to progress
            # This will advance text, select Fight, select first move
            logger.info("Fight strategy: pressing 'a'")
            return 'a'

        elif strategy == "run":
            # Check if we can run (wild vs trainer battle)
            add_to_state_schema(
                key="can_run_from_battle",
                vlm_prompt="Can the player run from this battle? Wild Pokemon battles allow running, but Trainer battles do not. Look for text like 'Wild POKEMON appeared!' (can run) vs trainer name/sprite (cannot run). Answer true if wild battle, false if trainer battle.",
                return_type=bool
            )

            can_run = state["can_run_from_battle"]
            logger.info(f"Can run: {can_run}")

            if not can_run:
                # Trainer battle: can't run, switch to fight
                logger.info("Trainer battle detected, switching to fight strategy")
                return 'a'

            # Wild battle: navigate to Run option
            logger.info("Wild battle: navigating to Run")
            return _navigate_to_run(state)

        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'a'")
            return 'a'

    except Exception as e:
        logger.error(f"Battle handler error: {e}")
        import traceback
        traceback.print_exc()
        return 'no_op'


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
        return_type=bool
    )

    if not state["battle_main_menu_visible"]:
        # Menu not visible, press 'a' to advance/open menu
        logger.info("Main menu not visible, pressing 'a'")
        return 'a'

    # Check current selection
    add_to_state_schema(
        key="run_selected",
        vlm_prompt="Is the 'Run' option currently highlighted/selected in the battle menu? Look for highlighting or cursor on Run option.",
        return_type=bool
    )

    if state["run_selected"]:
        # Already on Run, confirm
        logger.info("Run selected, confirming with 'a'")
        return 'a'

    # Need to navigate to Run
    # Ask VLM for direction
    add_to_state_schema(
        key="direction_to_run",
        vlm_prompt="To navigate to the 'Run' option in the battle menu, which direction should I press? Answer with one of: 'up', 'down', 'left', 'right'. Run is usually in the bottom-right corner.",
        return_type=str
    )

    direction = state["direction_to_run"].lower()
    logger.info(f"Navigating {direction} to Run")

    return direction if direction in ['up', 'down', 'left', 'right'] else 'down'