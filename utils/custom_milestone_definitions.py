#!/usr/bin/env python3
"""
Custom milestone definitions (state-based checks)

Additional milestones beyond server milestones for finer-grained progress tracking.
"""


def check_clock_interact(game_state, action=None):
    """
    Check if player interacted with clock in bedroom
    - Player at position (5, 2) - the tile in front of the clock
    - Player facing north (from previous 'up' action)
    - Previous action: 'a' (interacted with the clock while facing it)

    Note: This uses game_state.prev_action instead of the action parameter.
    The action parameter is kept for API compatibility but not used.
    """
    player = game_state.get("player", {})
    pos = player.get("position", {})

    # Must be at position (5,2) - in front of the clock
    if not (pos.get("x") == 5 and pos.get("y") == 2):
        return False

    # Must be facing north (facing the clock)
    # Check both top-level and player.facing (for different state formats)
    facing = game_state.get("facing", "") or player.get("facing", "")
    if facing.lower() != "north":
        return False

    # Previous action must be 'a' (interacted with clock)
    prev_action = game_state.get("prev_action", None)
    print(pos, facing, prev_action)
    if prev_action != 'a':
        return False

    return True


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


def check_exit_birch_lab_to_littleroot(game_state, action):
    """
    Check if player exited Birch's lab to Littleroot Town (outside):
    - Player location must be in Littleroot Town
    - NOT in lab or house (outside)

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # In Littleroot but NOT in lab or house (outside)
    return ("LITTLEROOT" in location_upper and
            "LAB" not in location_upper and
            "HOUSE" not in location_upper)


def check_littleroot_to_route101_first_time(game_state, action):
    """
    Check if player left Littleroot to Route 101 (after visiting Birch's lab, before Oldale):
    - Player location must be Route 101

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "ROUTE 101" in location_upper or "ROUTE_101" in location_upper


def check_route101_to_oldale(game_state, action):
    """
    Check if player reached Oldale Town from Route 101 (first time):
    - Player location must be Oldale Town

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "OLDALE" in location_upper


# def check_level_7_achieved(game_state, action):
#     """
#     Check if any Pokemon in the party has reached level 7 or higher:
#     - At least one Pokemon in party with level >= 7
#
#     Note: This is a state-based check, action parameter is not used.
#     """
#     player = game_state.get("player", {})
#     party = player.get("party", [])
#
#     # Check if any Pokemon has level >= 7
#     for pokemon in party:
#         level = pokemon.get("level", 0)
#         if level >= 7:
#             return True
#
#     return False


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


def check_received_pokedex(game_state, action):
    """
    Check if player received Pokedex and exited lab to Littleroot Town:
    - Player must have starter Pokemon
    - Player must be in Littleroot Town but NOT in the lab (outside)

    Note: This is a location-based check, action parameter is not used.
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()
    party = player.get("party", [])

    # Player must have starter (already received)
    has_starter = len(party) >= 1

    # Player must be in Littleroot Town but NOT in the lab (outside)
    in_littleroot_outside = "LITTLEROOT" in location_upper and "LAB" not in location_upper and "HOUSE" not in location_upper

    return has_starter and in_littleroot_outside


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


def check_route101_after_pokedex(game_state, action):
    """
    Check if player reached Route 101 after getting Pokedex
    (on the way to Oldale and Route 102)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "ROUTE 101" in location_upper or "ROUTE_101" in location_upper


def check_oldale_after_pokedex(game_state, action):
    """
    Check if player reached Oldale Town after getting Pokedex
    (on the way to Route 102)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "OLDALE" in location_upper


def check_oldale_to_route102(game_state, action):
    """
    Check if player reached Oldale Town on the way to Route 102
    (after getting Pokedex)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    return "OLDALE" in location_upper


def check_gym_cutscene_outside(game_state, action):
    """
    Check if player is outside during gym cutscene (after Dad dialog):
    - Must be in Petalburg City
    - NOT in gym (cutscene moved player outside)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg City but NOT in gym (cutscene)
    return "PETALBURG" in location_upper and "GYM" not in location_upper


def check_back_in_gym_after_cutscene(game_state, action):
    """
    Check if player is back in gym after cutscene ends:
    - Must be in Petalburg Gym
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg Gym (back inside after cutscene)
    return "PETALBURG" in location_upper and "GYM" in location_upper


def check_exit_petalburg_gym(game_state, action):
    """
    Check if player exited Petalburg Gym after receiving gym explanation
    (heading to Pokemon Center or Route 104 South)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg City but NOT in gym
    return "PETALBURG" in location_upper and "GYM" not in location_upper


def check_enter_petalburg_center(game_state, action):
    """
    Check if player entered Petalburg Pokemon Center:
    - Must be in Petalburg Pokemon Center
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg Pokemon Center
    return ("PETALBURG" in location_upper and
            ("POKEMON CENTER" in location_upper or "POKÉMON CENTER" in location_upper or "POKECENTER" in location_upper))


def check_heal_at_petalburg_center(game_state, action):
    """
    Check if player healed Pokemon at Petalburg Pokemon Center:
    - Must be in Petalburg Pokemon Center
    - All Pokemon in party must have full HP
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg Pokemon Center
    if not ("PETALBURG" in location_upper and
            ("POKEMON CENTER" in location_upper or "POKÉMON CENTER" in location_upper or "POKECENTER" in location_upper)):
        return False

    # Check all Pokemon have full HP
    party = player.get("party", [])
    if not party:
        return False

    for pokemon in party:
        current_hp = pokemon.get("current_hp", 0)
        max_hp = pokemon.get("max_hp", 1)

        # Pokemon must have full HP
        if current_hp < max_hp:
            return False

    return True


def check_exit_petalburg_center(game_state, action):
    """
    Check if player exited Petalburg Pokemon Center after healing:
    - Must be in Petalburg City
    - NOT in Pokemon Center (outside)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Petalburg but NOT in Pokemon Center
    return ("PETALBURG" in location_upper and
            "POKEMON CENTER" not in location_upper and
            "POKÉMON CENTER" not in location_upper and
            "POKECENTER" not in location_upper)


def check_enter_rustboro_center(game_state, action):
    """
    Check if player entered Rustboro Pokemon Center:
    - Must be in Rustboro Pokemon Center
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Rustboro Pokemon Center
    return ("RUSTBORO" in location_upper and
            ("POKEMON CENTER" in location_upper or "POKÉMON CENTER" in location_upper or "POKECENTER" in location_upper))


def check_heal_at_rustboro_center(game_state, action):
    """
    Check if player healed Pokemon at Rustboro Pokemon Center:
    - Must be in Rustboro Pokemon Center
    - All Pokemon in party must have full HP
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Rustboro Pokemon Center
    if not ("RUSTBORO" in location_upper and
            ("POKEMON CENTER" in location_upper or "POKÉMON CENTER" in location_upper or "POKECENTER" in location_upper)):
        return False

    # Check all Pokemon have full HP
    party = player.get("party", [])
    if not party:
        return False

    for pokemon in party:
        current_hp = pokemon.get("current_hp", 0)
        max_hp = pokemon.get("max_hp", 1)

        # Pokemon must have full HP
        if current_hp < max_hp:
            return False

    return True


def check_exit_rustboro_center(game_state, action):
    """
    Check if player exited Rustboro Pokemon Center after healing:
    - Must be in Rustboro City
    - NOT in Pokemon Center (outside)
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Rustboro but NOT in Pokemon Center
    return ("RUSTBORO" in location_upper and
            "POKEMON CENTER" not in location_upper and
            "POKÉMON CENTER" not in location_upper and
            "POKECENTER" not in location_upper)


def check_trainer_josh_battle(game_state, action):
    """
    Check if player is in battle with Trainer Josh at Rustboro Gym:
    - Must be in Rustboro Gym
    - Near coordinates (5, 13) - Josh's location
    - is_in_battle = True
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Rustboro Gym
    if not ("RUSTBORO" in location_upper and "GYM" in location_upper):
        return False

    # Must be in battle
    is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
    if not is_in_battle:
        return False

    # Check coordinates near Josh (5, 13)
    position = player.get("position", {})
    x = position.get("x", 0)
    y = position.get("y", 0)

    # Within 2 tiles of Josh's position
    return abs(x - 5) <= 2 and abs(y - 13) <= 2


def check_roxanne_battle(game_state, action):
    """
    Check if player is in battle with Gym Leader Roxanne at Rustboro Gym:
    - Must be in Rustboro Gym
    - Near coordinates (5, 2) - Roxanne's location
    - is_in_battle = True
    """
    player = game_state.get("player", {})
    location = player.get("location", "")
    location_upper = str(location).upper()

    # Must be in Rustboro Gym
    if not ("RUSTBORO" in location_upper and "GYM" in location_upper):
        return False

    # Must be in battle
    is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
    if not is_in_battle:
        return False

    # Check coordinates near Roxanne (5, 2)
    position = player.get("position", {})
    x = position.get("x", 0)
    y = position.get("y", 0)

    # Within 2 tiles of Roxanne's position
    return abs(x - 5) <= 2 and abs(y - 2) <= 2


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

    # Birch Lab to Oldale Town - first journey
    {
        "id": "EXIT_BIRCH_LAB",
        "description": "Exit Birch's lab to Littleroot Town (outside)",
        "insert_after": "BIRCH_LAB_VISITED",
        "check_fn": check_exit_birch_lab_to_littleroot,
        "category": "navigation"
    },
    {
        "id": "LITTLEROOT_TO_ROUTE101_AFTER_LAB",
        "description": "Leave Littleroot Town and head to Route 101 (after visiting Birch's lab)",
        "insert_after": "EXIT_BIRCH_LAB",
        "check_fn": check_littleroot_to_route101_first_time,
        "category": "navigation"
    },
    {
        "id": "ROUTE101_TO_OLDALE_FIRST_TIME",
        "description": "Travel from Route 101 to Oldale Town (first time)",
        "insert_after": "LITTLEROOT_TO_ROUTE101_AFTER_LAB",
        "check_fn": check_route101_to_oldale,
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

    # Level 7 achievement on Route 103 (before battling May) - DISABLED
    # {
    #     "id": "LEVEL_7_ACHIEVED",
    #     "description": "Train Pokemon to level 7 or higher on Route 103",
    #     "insert_after": "ROUTE_103",
    #     "check_fn": "check_level_7_achieved",  # Function is commented out above
    #     "category": "training"
    # },

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
    {
        "id": "RECEIVED_POKEDEX",
        "description": "Receive Pokedex and exit Birch's lab to Littleroot Town",
        "insert_after": "POKEDEX_DIALOG_CONFIRMED",
        "check_fn": check_received_pokedex,
        "category": "navigation"
    },

    # Journey from Littleroot to Route 102 after getting Pokedex - broken down into steps
    {
        "id": "ROUTE101_AFTER_POKEDEX",
        "description": "Travel to Route 101 (after getting Pokedex, heading to Oldale)",
        "insert_after": "RECEIVED_POKEDEX",
        "check_fn": check_route101_after_pokedex,
        "category": "navigation"
    },
    {
        "id": "OLDALE_AFTER_POKEDEX",
        "description": "Arrive at Oldale Town (after getting Pokedex, heading to Route 102)",
        "insert_after": "ROUTE101_AFTER_POKEDEX",
        "check_fn": check_oldale_after_pokedex,
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
        "id": "GYM_CUTSCENE_OUTSIDE",
        "description": "Gym cutscene starts and player is moved outside gym",
        "insert_after": "DAD_DIALOG_CONFIRMED",
        "check_fn": check_gym_cutscene_outside,
        "category": "cutscene"
    },
    {
        "id": "BACK_IN_GYM_AFTER_CUTSCENE",
        "description": "Cutscene ends and player is back inside Petalburg Gym",
        "insert_after": "GYM_CUTSCENE_OUTSIDE",
        "check_fn": check_back_in_gym_after_cutscene,
        "category": "cutscene"
    },
    {
        "id": "EXIT_PETALBURG_GYM",
        "description": "Exit Petalburg Gym after receiving gym explanation",
        "insert_after": "BACK_IN_GYM_AFTER_CUTSCENE",
        "check_fn": check_exit_petalburg_gym,
        "category": "navigation"
    },

    # Petalburg Pokemon Center - heal before Route 104
    {
        "id": "ENTER_PETALBURG_CENTER",
        "description": "Enter Petalburg City Pokemon Center",
        "insert_after": "EXIT_PETALBURG_GYM",
        "check_fn": check_enter_petalburg_center,
        "category": "navigation"
    },
    {
        "id": "HEAL_AT_PETALBURG_CENTER",
        "description": "Heal Pokemon at Petalburg City Pokemon Center (all Pokemon must have full HP)",
        "insert_after": "ENTER_PETALBURG_CENTER",
        "check_fn": check_heal_at_petalburg_center,
        "category": "healing"
    },
    {
        "id": "EXIT_PETALBURG_CENTER",
        "description": "Exit Petalburg City Pokemon Center after healing",
        "insert_after": "HEAL_AT_PETALBURG_CENTER",
        "check_fn": check_exit_petalburg_center,
        "category": "navigation"
    },

    # Rustboro Pokemon Center - heal before Gym
    {
        "id": "RUSTBORO_CENTER_ENTERED",
        "description": "Enter Rustboro City Pokemon Center",
        "insert_after": "RUSTBORO_CITY",
        "check_fn": check_enter_rustboro_center,
        "category": "navigation"
    },
    {
        "id": "HEAL_AT_RUSTBORO_CENTER",
        "description": "Heal Pokemon at Rustboro City Pokemon Center (all Pokemon must have full HP)",
        "insert_after": "RUSTBORO_CENTER_ENTERED",
        "check_fn": check_heal_at_rustboro_center,
        "category": "healing"
    },
    {
        "id": "RUSTBORO_CENTER_EXITED",
        "description": "Exit Rustboro City Pokemon Center after healing",
        "insert_after": "HEAL_AT_RUSTBORO_CENTER",
        "check_fn": check_exit_rustboro_center,
        "category": "navigation"
    },

    # Rustboro Gym trainer battles
    {
        "id": "TRAINER_JOSH_BATTLE",
        "description": "Battle with Trainer Josh at Rustboro Gym (coordinates 5,13)",
        "insert_after": "RUSTBORO_GYM_ENTERED",
        "check_fn": check_trainer_josh_battle,
        "category": "battle"
    },
    {
        "id": "ROXANNE_BATTLE",
        "description": "Battle with Gym Leader Roxanne at coordinates (5,2)",
        "insert_after": "TRAINER_JOSH_BATTLE",
        "check_fn": check_roxanne_battle,
        "category": "battle"
    }
]
