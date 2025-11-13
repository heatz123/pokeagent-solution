#!/usr/bin/env python3
"""
UI Navigation tool for CodeAgent

Provides VLM-based UI navigation to handle menus, dialogs, and confirmations.
This file is auto-loaded - no import needed in LLM code!
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def navigate_ui(state, intent: str = "exit") -> str:
    """
    Navigate UI elements (menus, dialogs, confirmations) using VLM.

    This tool analyzes the current screenshot to detect UI elements and
    determines the appropriate action to take based on your intent.

    Args:
        state: State object with VLM support
        intent: What you want to do with the UI
            - "exit": Exit/close the current UI, menu, or dialog (default)
            - "confirm": Confirm the current dialog/prompt
            - "select_yes": Select "Yes" option and confirm
            - "select_no": Select "No" option and confirm
            - "cancel": Cancel/decline the current dialog

    Returns:
        str: Next action ('a', 'b', 'up', 'down', 'left', 'right', 'no_op')

    Example:
        # Exit from any menu/dialog
        action = navigate_ui(state, intent="exit")
        return action

        # Confirm a dialog (e.g., "Save?")
        action = navigate_ui(state, intent="confirm")
        return action

        # Select "Yes" in a Yes/No dialog
        action = navigate_ui(state, intent="select_yes")
        return action

    Common UI patterns handled:
    - Text dialogs: Press 'a' to advance, 'b' to close
    - Confirmation prompts: Navigate to Yes/No and press 'a'
    - Menus: Navigate with arrows, select with 'a', exit with 'b'
    - Item/Pokemon selection: Navigate and confirm
    """
    try:
        from utils.vlm_state import add_to_state_schema

        # Register VLM queries for UI state detection
        add_to_state_schema(
            key="ui_is_open",
            vlm_prompt="Is there a menu, dialog box, or text prompt currently displayed on screen? Answer true if any UI overlay is visible, false if it's just the game world.",
            return_type=bool
        )

        add_to_state_schema(
            key="ui_type",
            vlm_prompt="What type of UI is displayed? Options: 'none' (no UI), 'text_dialog' (NPC/system text), 'confirmation' (Yes/No prompt), 'menu' (selection menu), 'battle' (battle screen)",
            return_type=str
        )

        # Check if UI is open
        if not state["ui_is_open"]:
            logger.info("No UI detected, no action needed")
            return 'no_op'

        ui_type = state["ui_type"].lower()
        logger.info(f"UI detected: {ui_type}, intent: {intent}")

        # Handle based on UI type and intent
        if intent == "exit":
            # Exit strategy: usually 'b' button closes menus
            if ui_type == "text_dialog":
                # For text dialogs, need to check if we can exit
                add_to_state_schema(
                    key="can_exit_dialog",
                    vlm_prompt="Can this text dialog be closed with B button, or does it need A to advance? Return true if it can be closed with B, false if it needs A to advance.",
                    return_type=bool
                )

                if state["can_exit_dialog"]:
                    logger.info("Text dialog: Exiting with 'b'")
                    return 'b'
                else:
                    logger.info("Text dialog: Advancing with 'a'")
                    return 'a'

            elif ui_type == "menu":
                logger.info("Menu: Exiting with 'b'")
                return 'b'

            elif ui_type == "confirmation":
                # For confirmation, select "No" to cancel
                return _handle_yes_no_dialog(state, select_yes=False)

            else:
                # Generic exit: try 'b'
                logger.info(f"Unknown UI type '{ui_type}': Trying 'b'")
                return 'b'

        elif intent == "confirm":
            # Confirm current dialog
            if ui_type == "confirmation":
                # Select "Yes"
                return _handle_yes_no_dialog(state, select_yes=True)
            else:
                # Generic confirm: press 'a'
                logger.info(f"Confirming with 'a'")
                return 'a'

        elif intent == "select_yes":
            return _handle_yes_no_dialog(state, select_yes=True)

        elif intent == "select_no":
            return _handle_yes_no_dialog(state, select_yes=False)

        elif intent == "cancel":
            # Cancel is usually 'b' or selecting "No"
            if ui_type == "confirmation":
                return _handle_yes_no_dialog(state, select_yes=False)
            else:
                logger.info("Canceling with 'b'")
                return 'b'

        else:
            logger.warning(f"Unknown intent '{intent}', defaulting to exit")
            return 'b'

    except Exception as e:
        logger.error(f"UI navigation error: {e}")
        import traceback
        traceback.print_exc()
        return 'no_op'


def _handle_yes_no_dialog(state, select_yes: bool) -> str:
    """
    Helper function to handle Yes/No confirmation dialogs.

    Args:
        state: State object with VLM support
        select_yes: True to select Yes, False to select No

    Returns:
        str: Action to take
    """
    from utils.vlm_state import add_to_state_schema

    # Check current selection
    add_to_state_schema(
        key="yes_no_selection",
        vlm_prompt="In the Yes/No dialog, which option is currently highlighted/selected? Answer 'yes', 'no', or 'unknown'.",
        return_type=str
    )

    current_selection = state["yes_no_selection"].lower()
    target = "yes" if select_yes else "no"

    logger.info(f"Yes/No dialog: current='{current_selection}', target='{target}'")

    if current_selection == target:
        # Already on target, confirm
        logger.info(f"Already on {target}, confirming with 'a'")
        return 'a'

    elif current_selection == "unknown":
        # Can't determine selection, ask VLM for best action
        add_to_state_schema(
            key="yes_no_action",
            vlm_prompt=f"To select '{target.upper()}' in this Yes/No dialog, what action should I take? Answer with one of: 'a' (if already selected), 'up', 'down', 'left', 'right'.",
            return_type=str
        )

        action = state["yes_no_action"].lower()
        logger.info(f"VLM suggests action: {action}")
        return action if action in ['a', 'up', 'down', 'left', 'right'] else 'a'

    else:
        # Need to move selection
        # Most Pokemon games: Yes is top/left, No is bottom/right
        if select_yes:
            # Move to Yes: try 'up' first, then 'left'
            add_to_state_schema(
                key="yes_position",
                vlm_prompt="Is the 'Yes' option above the 'No' option (vertical layout) or to the left (horizontal layout)? Answer 'vertical' or 'horizontal'.",
                return_type=str
            )

            layout = state["yes_position"].lower()
            if "vertical" in layout:
                logger.info("Moving to Yes with 'up'")
                return 'up'
            else:
                logger.info("Moving to Yes with 'left'")
                return 'left'
        else:
            # Move to No: try 'down' first, then 'right'
            add_to_state_schema(
                key="no_position",
                vlm_prompt="Is the 'No' option below the 'Yes' option (vertical layout) or to the right (horizontal layout)? Answer 'vertical' or 'horizontal'.",
                return_type=str
            )

            layout = state["no_position"].lower()
            if "vertical" in layout:
                logger.info("Moving to No with 'down'")
                return 'down'
            else:
                logger.info("Moving to No with 'right'")
                return 'right'