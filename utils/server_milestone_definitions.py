#!/usr/bin/env python3
"""
Server milestone definitions as custom milestones (state-based checks only)

All milestones are converted to Python check functions that examine game state.
This allows full Python control over milestone completion conditions for RL training.
"""


def check_game_running(state, action):
    """Check if game is running (always true if we have state)"""
    return state is not None and "player" in state


def check_player_name_set(state, action):
    """Check if player name is set"""
    name = state.get("player", {}).get("name", "").strip()
    return name not in ["", "UNKNOWN", "PLAYER"]


def check_intro_cutscene_complete(state, action):
    """Check if intro cutscene is complete (in moving van)"""
    location = state.get("player", {}).get("location", "").upper()
    return "MOVING" in location and "VAN" in location


def check_littleroot_town(state, action):
    """Check if player reached Littleroot Town"""
    location = state.get("player", {}).get("location", "").upper()
    return "LITTLEROOT" in location


def check_player_house_entered(state, action):
    """Check if player entered their house"""
    location = state.get("player", {}).get("location", "").upper()
    # Brendan's house 1F (not May's house, not 2F bedroom)
    return "LITTLEROOT" in location and "BRENDAN" in location and "HOUSE" in location and "1F" in location


def check_player_bedroom(state, action):
    """Check if player reached their bedroom (2F)"""
    location = state.get("player", {}).get("location", "").upper()
    # Brendan's house 2F (player's bedroom)
    return "LITTLEROOT" in location and "BRENDAN" in location and "HOUSE" in location and "2F" in location


def check_rival_house(state, action):
    """Check if player entered rival's house"""
    location = state.get("player", {}).get("location", "").upper()
    # May's house 1F only (rival's house, not player's house)
    return "LITTLEROOT" in location and "MAY" in location and "HOUSE" in location and "1F" in location


def check_rival_bedroom(state, action):
    """Check if player reached rival's bedroom"""
    location = state.get("player", {}).get("location", "").upper()
    # May's house 2F only (rival's bedroom)
    return "LITTLEROOT" in location and "MAY" in location and "HOUSE" in location and "2F" in location


def check_route_101(state, action):
    """Check if player reached Route 101"""
    location = state.get("player", {}).get("location", "").upper()
    return "ROUTE 101" in location or "ROUTE_101" in location


def check_starter_chosen(state, action):
    """Check if player has chosen Mudkip as starter Pokemon"""
    party = state.get("player", {}).get("party", [])
    if len(party) == 0:
        return False

    # Check if any Pokemon in party is Mudkip
    for pokemon in party:
        species_name = pokemon.get("species_name", "").strip().upper()
        if species_name == "MUDKIP":
            return True

    return False


def check_birch_lab_visited(state, action):
    """Check if player visited Birch's lab"""
    location = state.get("player", {}).get("location", "").upper()
    return "LITTLEROOT" in location and "LAB" in location


def check_oldale_town(state, action):
    """Check if player reached Oldale Town"""
    location = state.get("player", {}).get("location", "").upper()
    return "OLDALE" in location


def check_route_103(state, action):
    """Check if player reached Route 103"""
    location = state.get("player", {}).get("location", "").upper()
    return "ROUTE 103" in location or "ROUTE_103" in location


def check_route_102(state, action):
    """Check if player reached Route 102"""
    location = state.get("player", {}).get("location", "").upper()
    return "ROUTE 102" in location or "ROUTE_102" in location


def check_petalburg_city(state, action):
    """Check if player reached Petalburg City"""
    location = state.get("player", {}).get("location", "").upper()
    return "PETALBURG" in location


def check_dad_first_meeting(state, action):
    """Check if player met Dad at Petalburg Gym"""
    location = state.get("player", {}).get("location", "").upper()
    return "PETALBURG" in location and "GYM" in location


def check_route_104_south(state, action):
    """Check if player reached Route 104 (south section)"""
    location = state.get("player", {}).get("location", "").upper()
    return "ROUTE 104" in location or "ROUTE_104" in location


def check_petalburg_woods(state, action):
    """Check if player entered Petalburg Woods"""
    location = state.get("player", {}).get("location", "").upper()
    return "MAP_18_0B" in location or ("PETALBURG" in location and "WOODS" in location)


def check_team_aqua_grunt_defeated(state, action):
    """Check if Team Aqua grunt was defeated in Petalburg Woods"""
    location = state.get("player", {}).get("location", "").upper()
    # Must be in Petalburg Woods
    is_in_petalburg = "MAP_18_0B" in location or ("PETALBURG" in location and "WOODS" in location)
    if is_in_petalburg:
        # Check specific coordinates where grunt battle occurs
        position = state.get("player", {}).get("position", {})
        x = position.get("x", 0)
        y = position.get("y", 0)
        # Team Aqua grunt is at coordinates (26,23) or (27,23)
        if y == 23 and x in [26, 27]:
            return True
    return False


def check_route_104_north(state, action):
    """Check if player reached Route 104 North (after Petalburg Woods)"""
    location = state.get("player", {}).get("location", "").upper()
    # Must be on Route 104
    return "ROUTE 104" in location or "ROUTE_104" in location


def check_rustboro_city(state, action):
    """Check if player reached Rustboro City"""
    location = state.get("player", {}).get("location", "").upper()
    return "RUSTBORO" in location


def check_rustboro_center_entered(state, action):
    """Check if player entered Rustboro City Pokemon Center"""
    location = state.get("player", {}).get("location", "").upper()
    return "RUSTBORO" in location and (
        "POKEMON CENTER" in location or "POKÉMON CENTER" in location or "POKECENTER" in location
    )


def check_heal_at_rustboro_center(state, action):
    """Check if player healed Pokemon at Rustboro Pokemon Center (all Pokemon must have full HP)"""
    location = state.get("player", {}).get("location", "").upper()

    # Must be in Rustboro Pokemon Center
    if not (
        "RUSTBORO" in location
        and ("POKEMON CENTER" in location or "POKÉMON CENTER" in location or "POKECENTER" in location)
    ):
        return False

    # Check all Pokemon have full HP
    party = state.get("player", {}).get("party", [])
    if not party:
        return False

    for pokemon in party:
        current_hp = pokemon.get("current_hp", 0)
        max_hp = pokemon.get("max_hp", 1)

        # Pokemon must have full HP
        if current_hp < max_hp:
            return False

    return True


def check_rustboro_center_exited(state, action):
    """Check if player exited Rustboro City Pokemon Center back to Rustboro City"""
    location = state.get("player", {}).get("location", "").upper()
    # Should be in Rustboro but NOT in Pokemon Center
    return (
        "RUSTBORO" in location
        and "POKEMON CENTER" not in location
        and "POKÉMON CENTER" not in location
        and "POKECENTER" not in location
    )


def check_rustboro_gym_entered(state, action):
    """Check if player entered Rustboro Gym"""
    location = state.get("player", {}).get("location", "").upper()
    return "RUSTBORO" in location and "GYM" in location


def check_stone_badge(state, action):
    """Check if player exited Rustboro Gym"""
    location = state.get("player", {}).get("location", "")

    # Player must be outside the gym
    # outside_gym = location != "RUSTBOROCITY GYM"
    return "GYM" not in location and "RUSTBORO" in location


# List of all server milestones to be registered as custom milestones
# Order matters: insert_after creates dependency chain
SERVER_MILESTONES = [
    {
        "id": "GAME_RUNNING",
        "insert_after": None,  # First milestone
        "description": "Complete title sequence and begin the game",
        "check_fn": check_game_running,
        "category": "system",
    },
    {
        "id": "PLAYER_NAME_SET",
        "insert_after": "GAME_RUNNING",
        "description": "Player has chosen their character name",
        "check_fn": check_player_name_set,
        "category": "intro",
    },
    {
        "id": "INTRO_CUTSCENE_COMPLETE",
        "insert_after": "PLAYER_NAME_SET",
        "description": "Complete intro cutscene with moving van",
        "check_fn": check_intro_cutscene_complete,
        "category": "intro",
    },
    {
        "id": "LITTLEROOT_TOWN",
        "insert_after": "INTRO_CUTSCENE_COMPLETE",
        "description": "Arrive at Littleroot Town",
        "check_fn": check_littleroot_town,
        "category": "location",
    },
    {
        "id": "PLAYER_HOUSE_ENTERED",
        "insert_after": "LITTLEROOT_TOWN",
        "description": "Enter player's house for the first time",
        "check_fn": check_player_house_entered,
        "category": "location",
    },
    {
        "id": "PLAYER_BEDROOM",
        "insert_after": "PLAYER_HOUSE_ENTERED",
        "description": "Go upstairs to player's bedroom",
        "check_fn": check_player_bedroom,
        "category": "location",
    },
    {
        "id": "RIVAL_HOUSE",
        "insert_after": "LEAVE_HOUSE",  # After custom LEAVE_HOUSE milestone
        "description": "Visit May's house next door",
        "check_fn": check_rival_house,
        "category": "location",
    },
    {
        "id": "RIVAL_BEDROOM",
        "insert_after": "RIVAL_HOUSE",
        "description": "Visit May's bedroom on the second floor",
        "check_fn": check_rival_bedroom,
        "category": "location",
    },
    {
        "id": "ROUTE_101",
        "insert_after": "RIVAL_BEDROOM",
        "description": "Travel to Route 101",
        "check_fn": check_route_101,
        "category": "location",
    },
    {
        "id": "STARTER_CHOSEN",
        "insert_after": "ROUTE_101",
        "description": "Choose starter Pokemon as **Mudkip**",
        "check_fn": check_starter_chosen,
        "category": "pokemon",
    },
    {
        "id": "BIRCH_LAB_VISITED",
        "insert_after": "STARTER_CHOSEN",
        "description": "Visit Professor Birch's lab",
        "check_fn": check_birch_lab_visited,
        "category": "location",
    },
    {
        "id": "OLDALE_TOWN",
        "insert_after": "BIRCH_LAB_VISITED",
        "description": "Arrive at Oldale Town",
        "check_fn": check_oldale_town,
        "category": "location",
    },
    {
        "id": "ROUTE_103",
        "insert_after": "OLDALE_TOWN",
        "description": "Travel to Route 103 and battle with May",
        "check_fn": check_route_103,
        "category": "location",
    },
    {
        "id": "ROUTE_102",
        "insert_after": "RECEIVED_POKEDEX",  # RECEIVED_POKEDEX is now in CUSTOM_MILESTONES
        "description": "Travel to Route 102",
        "check_fn": check_route_102,
        "category": "location",
    },
    {
        "id": "PETALBURG_CITY",
        "insert_after": "ROUTE_102",
        "description": "Arrive at Petalburg City",
        "check_fn": check_petalburg_city,
        "category": "location",
    },
    {
        "id": "DAD_FIRST_MEETING",
        "insert_after": "PETALBURG_CITY",
        "description": "Meet Dad at Petalburg Gym",
        "check_fn": check_dad_first_meeting,
        "category": "story",
    },
    {
        "id": "ROUTE_104_SOUTH",
        "insert_after": "EXIT_PETALBURG_GYM",
        "description": "Travel to Route 104 (South)",
        "check_fn": check_route_104_south,
        "category": "location",
    },
    {
        "id": "PETALBURG_WOODS",
        "insert_after": "ROUTE_104_SOUTH",
        "description": "Navigate through Petalburg Woods",
        "check_fn": check_petalburg_woods,
        "category": "location",
    },
    {
        "id": "TEAM_AQUA_GRUNT_DEFEATED",
        "insert_after": "PETALBURG_WOODS",
        "description": "Defeat Team Aqua grunt in Petalburg Woods",
        "check_fn": check_team_aqua_grunt_defeated,
        "category": "event",
    },
    {
        "id": "ROUTE_104_NORTH",
        "insert_after": "TEAM_AQUA_GRUNT_DEFEATED",
        "description": "Exit Petalburg Woods to Route 104 North",
        "check_fn": check_route_104_north,
        "category": "location",
    },
    {
        "id": "RUSTBORO_CITY",
        "insert_after": "ROUTE_104_NORTH",
        "description": "Arrive at Rustboro City (x=16) from Route 104 north section",
        "check_fn": check_rustboro_city,
        "category": "location",
    },
    {
        "id": "RUSTBORO_CENTER_ENTERED",
        "insert_after": "RUSTBORO_CITY",
        "description": "Enter Rustboro City Pokemon Center",
        "check_fn": check_rustboro_center_entered,
        "category": "location",
    },
    {
        "id": "HEAL_AT_RUSTBORO_CENTER",
        "insert_after": "RUSTBORO_CENTER_ENTERED",
        "description": "Heal Pokemon at Rustboro City Pokemon Center (all Pokemon must have full HP)",
        "check_fn": check_heal_at_rustboro_center,
        "category": "healing",
    },
    {
        "id": "RUSTBORO_CENTER_EXITED",
        "insert_after": "HEAL_AT_RUSTBORO_CENTER",
        "description": "Exit Rustboro City Pokemon Center after healing",
        "check_fn": check_rustboro_center_exited,
        "category": "location",
    },
    {
        "id": "RUSTBORO_GYM_ENTERED",
        "insert_after": "RUSTBORO_CENTER_EXITED",
        "description": "Enter Rustboro Gym",
        "check_fn": check_rustboro_gym_entered,
        "category": "location",
    },
    {
        "id": "STONE_BADGE",
        "insert_after": "RUSTBORO_GYM_ENTERED",
        "description": "Exit Rustboro Gym after defeating Roxanne",
        "check_fn": check_stone_badge,
        "category": "badge",
    },
]
