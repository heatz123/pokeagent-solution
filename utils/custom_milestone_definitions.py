#!/usr/bin/env python3
"""
Custom milestone definitions (state-based checks)

Additional milestones beyond server milestones for finer-grained progress tracking.
"""


def check_clock_interact(game_state, action):
    """
    Check if player interacted with clock in bedroom
    - Player at position (5, 2) - the tile in front of the clock
    - Player facing north (from previous 'up' action)
    - Action: 'a' (interact with the clock while facing it)

    Note: facing direction is determined by prev_action (directional keys).
    If prev_action was 'up', facing will be 'north'.
    """
    player = game_state.get("player", {})
    pos = player.get("position", {})

    # Must be at position (5,2) - in front of the clock
    if not (pos.get("x") == 5 and pos.get("y") == 2):
        return False

    # Must be facing north (facing the clock)
    facing = game_state.get("facing", "")
    if facing != "north":
        return False

    # Action must be 'a' (interact)
    if isinstance(action, str):
        return action == 'a'

    # Also support list actions with 'a' (in case of multi-action lists)
    if isinstance(action, list):
        return 'a' in action

    return False


def check_downstairs_to_1f(game_state, action):
    """
    Check if player went downstairs from bedroom (2F) to 1F:
    - Player location must be Brendan's house 1F

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Brendan's house 1F (after coming down from 2F)
    return ("LITTLEROOT" in location_upper and
            "BRENDAN" in location_upper and
            "HOUSE" in location_upper and
            "1F" in location_upper)


def check_exit_brendan_house(game_state, action):
    """
    Check if player exited Brendan's house to outside:
    - Player location must be in Littleroot Town
    - NOT in any house or lab (outside)

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # In Littleroot but NOT in either house or lab (outside)
    return ("LITTLEROOT" in location_upper and
            "HOUSE" not in location_upper and
            "LAB" not in location_upper)


def check_rival_bedroom_to_1f(game_state, action):
    """
    Check if player went downstairs from May's bedroom (2F) to 1F:
    - Player location must be May's house 1F

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in May's house 1F (after coming down from 2F)
    return ("LITTLEROOT" in location_upper and
            "MAY" in location_upper and
            "HOUSE" in location_upper and
            "1F" in location_upper)


def check_exit_rival_house(game_state, action):
    """
    Check if player exited May's house to Littleroot Town (outside):
    - Player location must be in Littleroot Town
    - NOT in any house or lab (outside)

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # In Littleroot but NOT in either house or lab (outside)
    return ("LITTLEROOT" in location_upper and
            "HOUSE" not in location_upper and
            "LAB" not in location_upper)


def check_littleroot_to_route101(game_state, action):
    """
    Check if player left Littleroot Town to Route 101 (after leaving May's house):
    - Player location must be Route 101

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "ROUTE 101" in location_upper or "ROUTE_101" in location_upper


def check_may_interaction(game_state, action):
    """
    Check if May interaction/battle happened on Route 103:
    - Must be on Route 103
    - Dialog text contains 'MAY: I think I know'

    Note: This is a dialog-based check, action parameter is not used.
    """
    # Check location first
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be on Route 103
    if not "ROUTE 103" in location_upper:
        return False

    # Check dialog text for May
    dialog_text = game_state.get("game", {}).get("dialog_text") or ""

    # Check for May's specific dialog
    return "MAY: I think I know" in dialog_text


def check_pokedex_dialog(game_state, action):
    """
    Check if Pokedex dialog text appeared in Birch's lab:
    - Must be in Birch's lab location
    - Dialog text contains 'POKéDEX', 'POKEDEX', or variants

    Note: This is a dialog-based check, action parameter is not used.
    """
    # Check location first
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Birch's lab
    if "LITTLEROOT TOWN PROFESSOR BIRCHS LAB" not in location_upper:
        return False

    # Check dialog text
    dialog_text = game_state.get("game", {}).get("dialog_text") or ""
    dialog_upper = str(dialog_text).upper()

    # Check multiple variants (é with acute accent, regular e, etc.)
    return ("POKEDEX" in dialog_upper or
            "POKÉDEX" in dialog_upper or
            "POKéDEX" in dialog_upper or
            "POKEDE'X" in dialog_upper or
            # Fallback: contains both "POK" and "DEX"
            ("POK" in dialog_upper and "DEX" in dialog_upper))


def check_dad_dialog(game_state, action):
    """
    Check if Dad dialog appeared at Petalburg Gym:
    - Must be in Petalburg City Gym
    - Dialog text contains 'DAD:'

    Note: This is a dialog-based check, action parameter is not used.
    """
    # Check location first
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg City Gym
    if not ("PETALBURG CITY GYM" in location_upper or "PETALBURG_CITY_GYM" in location_upper):
        return False

    # Check dialog text for Dad
    dialog_text = game_state.get("game", {}).get("dialog_text") or ""
    dialog_upper = str(dialog_text).upper()

    return "DAD:" in dialog_upper


def check_back_to_oldale_from_route103(game_state, action):
    """
    Check if player returned to Oldale Town from Route 103
    (to go back to Littleroot for Pokedex)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "OLDALE" in location_upper


def check_back_to_route101_from_oldale(game_state, action):
    """
    Check if player returned to Route 101 from Oldale
    (on the way back to Littleroot for Pokedex)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "ROUTE 101" in location_upper or "ROUTE_101" in location_upper


def check_back_to_littleroot_for_pokedex(game_state, action):
    """
    Check if player returned to Littleroot Town after Route 103
    (to get Pokedex from Birch's lab)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Littleroot but NOT in lab yet (outside or in house)
    return "LITTLEROOT" in location_upper


def check_enter_birch_lab_for_pokedex(game_state, action):
    """
    Check if player entered Birch's lab to get Pokedex
    (after returning from Route 103)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "LITTLEROOT" in location_upper and "LAB" in location_upper


def check_leave_littleroot_to_route101(game_state, action):
    """
    Check if player left Littleroot to Route 101 after getting Pokedex
    (heading toward Oldale and Route 102)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "ROUTE 101" in location_upper or "ROUTE_101" in location_upper


def check_oldale_to_route102(game_state, action):
    """
    Check if player reached Oldale Town on the way to Route 102
    (after getting Pokedex)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "OLDALE" in location_upper


def check_exit_petalburg_gym(game_state, action):
    """
    Check if player exited Petalburg Gym after meeting Dad
    (heading to Route 104 South)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg City but NOT in gym
    return "PETALBURG" in location_upper and "GYM" not in location_upper


# Custom milestones list
CUSTOM_MILESTONES = [
    # Clock set sequence - broken down into detailed steps
    {
        "id": "CLOCK_INTERACT",
        "description": "At position (5,2) in player bedroom, face north and press 'a' to interact with the clock",
        "insert_after": "PLAYER_BEDROOM",
        "check_fn": check_clock_interact,
        "category": "custom"
    },
    {
        "id": "GO_DOWNSTAIRS_TO_1F",
        "description": "Go downstairs from bedroom (2F) to Brendan's house 1F",
        "insert_after": "CLOCK_INTERACT",
        "check_fn": check_downstairs_to_1f,
        "category": "navigation"
    },
    {
        "id": "LEAVE_HOUSE",
        "description": "Exit Brendan's house to Littleroot Town (outside)",
        "insert_after": "GO_DOWNSTAIRS_TO_1F",
        "check_fn": check_exit_brendan_house,
        "category": "navigation"
    },

    # May's house navigation - going downstairs and out to Route 101
    {
        "id": "GO_DOWNSTAIRS_RIVAL_HOUSE",
        "description": "Go downstairs from May's bedroom (2F) to May's house 1F",
        "insert_after": "RIVAL_BEDROOM",
        "check_fn": check_rival_bedroom_to_1f,
        "category": "navigation"
    },
    {
        "id": "EXIT_RIVAL_HOUSE",
        "description": "Exit May's house to Littleroot Town (outside)",
        "insert_after": "GO_DOWNSTAIRS_RIVAL_HOUSE",
        "check_fn": check_exit_rival_house,
        "category": "navigation"
    },
    {
        "id": "LITTLEROOT_TO_ROUTE101",
        "description": "Leave Littleroot Town and head to Route 101",
        "insert_after": "EXIT_RIVAL_HOUSE",
        "check_fn": check_littleroot_to_route101,
        "category": "navigation"
    },

    # Route 103 interaction
    {
        "id": "MAY_ROUTE103_INTERACTION",
        "description": "Interact with May on Route 103 (dialog contains 'MAY: I think I know')",
        "insert_after": "ROUTE_103",
        "check_fn": check_may_interaction,
        "category": "dialog"
    },

    # Return journey from Route 103 to Littleroot for Pokedex - broken down into steps
    {
        "id": "BACK_TO_OLDALE_FROM_ROUTE103",
        "description": "Return to Oldale Town from Route 103 (on the way back to get Pokedex)",
        "insert_after": "MAY_ROUTE103_INTERACTION",
        "check_fn": check_back_to_oldale_from_route103,
        "category": "navigation"
    },
    {
        "id": "BACK_TO_ROUTE101_FROM_OLDALE",
        "description": "Return to Route 101 from Oldale Town (heading back to Littleroot)",
        "insert_after": "BACK_TO_OLDALE_FROM_ROUTE103",
        "check_fn": check_back_to_route101_from_oldale,
        "category": "navigation"
    },
    {
        "id": "BACK_TO_LITTLEROOT_FOR_POKEDEX",
        "description": "Return to Littleroot Town after Route 103 (to get Pokedex)",
        "insert_after": "BACK_TO_ROUTE101_FROM_OLDALE",
        "check_fn": check_back_to_littleroot_for_pokedex,
        "category": "navigation"
    },
    {
        "id": "ENTER_BIRCH_LAB_FOR_POKEDEX",
        "description": "Enter Professor Birch's lab to receive Pokedex",
        "insert_after": "BACK_TO_LITTLEROOT_FOR_POKEDEX",
        "check_fn": check_enter_birch_lab_for_pokedex,
        "category": "navigation"
    },
    {
        "id": "POKEDEX_DIALOG_CONFIRMED",
        "description": "Pokedex dialog text appeared in Birch's lab (POKEDEX or POKEDE'X in dialog at PROFESSOR BIRCHS LAB)",
        "insert_after": "ENTER_BIRCH_LAB_FOR_POKEDEX",
        "check_fn": check_pokedex_dialog,
        "category": "dialog"
    },

    # Journey from Littleroot to Route 102 after getting Pokedex - broken down into steps
    {
        "id": "LEAVE_LITTLEROOT_TO_ROUTE101",
        "description": "Leave Littleroot Town toward Route 101 (after getting Pokedex)",
        "insert_after": "RECEIVED_POKEDEX",
        "check_fn": check_leave_littleroot_to_route101,
        "category": "navigation"
    },
    {
        "id": "OLDALE_TO_ROUTE102",
        "description": "Pass through Oldale Town on the way to Route 102",
        "insert_after": "LEAVE_LITTLEROOT_TO_ROUTE101",
        "check_fn": check_oldale_to_route102,
        "category": "navigation"
    },

    # Dad interaction at Petalburg Gym
    {
        "id": "DAD_DIALOG_CONFIRMED",
        "description": "Dad dialog appeared at Petalburg Gym (DAD: in dialog at PETALBURG CITY GYM)",
        "insert_after": "DAD_FIRST_MEETING",
        "check_fn": check_dad_dialog,
        "category": "dialog"
    },
    {
        "id": "EXIT_PETALBURG_GYM",
        "description": "Exit Petalburg Gym after meeting Dad (heading to Route 104 South)",
        "insert_after": "DAD_DIALOG_CONFIRMED",
        "check_fn": check_exit_petalburg_gym,
        "category": "navigation"
    }
]
