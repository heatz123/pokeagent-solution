#!/usr/bin/env python3
"""
Mart Purchase tool for CodeAgent

Provides VLM-based item purchasing at Poké Marts.
This file is auto-loaded - no import needed in LLM code!
"""

import logging

logger = logging.getLogger(__name__)


def buy_item_at_mart(state, item_name: str, quantity: int = 1) -> str:
    """
    Buy items at a Poké Mart using VLM-guided navigation.

    This tool handles the complete purchase flow:
    1. Verifies you're in a Mart and near the clerk
    2. Faces the clerk if needed
    3. Navigates through the purchase UI using VLM

    Requirements:
    - Must be inside a Mart (*_Mart map)
    - Must be within interaction range of the clerk (1-2 tiles)

    Args:
        state: State object with VLM support
        item_name: Item to buy (e.g., "Poké Ball", "Potion", "Antidote")
        quantity: Number of items to buy (default: 1)

    Returns:
        str: Next action ('up', 'down', 'left', 'right', 'a', 'b', 'no_op')

    Example:
        # Buy 5 Poké Balls when near the clerk
        action = buy_item_at_mart(state, item_name="Poké Ball", quantity=5)
        return action

        # Buy a single Potion (quantity defaults to 1)
        action = buy_item_at_mart(state, item_name="Potion")
        return action

    Common items in early game marts:
    - Poké Ball
    - Potion
    - Antidote
    - Paralyze Heal
    - Awakening
    - Escape Rope
    """
    try:
        from utils.vlm_state import add_to_state_schema
        from tools.ui_navigation import navigate_ui

        # 1. Check if we're in a mart
        # Try multiple ways to detect mart
        map_data = state.get('map', {})
        location = state.get('location', '')
        map_name = map_data.get('name', location)

        # Check for mart keywords or SHOP_SHELF tiles
        is_mart = False
        if '_Mart' in map_name or 'Mart' in map_name or 'MART' in map_name.upper():
            is_mart = True
        else:
            # Fallback: Check for SHOP_SHELF behavior (unique to marts)
            behaviors = map_data.get('metatile_behaviors', [])
            for row in behaviors:
                if 'SHOP_SHELF' in row:
                    is_mart = True
                    logger.info("Detected mart by SHOP_SHELF tiles")
                    break

        if not is_mart:
            logger.warning(f"Not in a mart (location: '{location}', map_name: '{map_name}')")
            return 'no_op'

        logger.info(f"In mart (location: {location or map_name}), attempting to buy {quantity}x {item_name}")

        # 2. VLM: Check if clerk is nearby (within interaction range)
        add_to_state_schema(
            key="clerk_nearby",
            vlm_prompt="Is there a mart clerk or shop employee NPC visible on screen and close enough to interact with (within 1-2 tiles of the player)? Answer true if the clerk is nearby and reachable, false otherwise.",
            return_type=bool
        )

        if not state["clerk_nearby"]:
            logger.info("Clerk not nearby - cannot purchase items")
            return 'no_op'

        logger.debug("Clerk is nearby")

        # 3. VLM: Check if we're facing the clerk
        add_to_state_schema(
            key="facing_clerk",
            vlm_prompt="Is the player character currently facing toward the mart clerk? Answer true if the player is facing the clerk, false if facing away or to the side.",
            return_type=bool
        )

        if not state["facing_clerk"]:
            # Need to turn to face clerk
            add_to_state_schema(
                key="clerk_direction",
                vlm_prompt="In which direction is the mart clerk from the player's current position? Answer with exactly one of: 'up', 'down', 'left', or 'right'.",
                return_type=str
            )

            direction = state["clerk_direction"].lower().strip()

            # Validate direction
            valid_directions = ['up', 'down', 'left', 'right']
            if direction not in valid_directions:
                logger.warning(f"Invalid direction from VLM: '{direction}', defaulting to 'up'")
                direction = 'up'

            logger.info(f"Turning to face clerk: {direction}")
            return direction

        logger.debug("Already facing clerk")

        # 4. Use navigate_ui with action_guidance for the purchase
        # VLM will handle the entire purchase flow based on current UI state
        action_guidance = f"I'm buying {quantity} {item_name} at the Poké Mart"

        logger.debug(f"Using navigate_ui with action_guidance: {action_guidance}")
        action = navigate_ui(state, action_guidance=action_guidance)

        logger.info(f"Purchase action: {action}")
        return action

    except Exception as e:
        logger.error(f"Mart purchase error: {e}")
        import traceback
        traceback.print_exc()
        return 'no_op'
