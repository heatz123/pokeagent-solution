#!/usr/bin/env python3
"""
Server milestone definitions as custom milestones (state-based checks only)

All milestones are converted to Python check functions that examine game state.
This allows full Python control over milestone completion conditions for RL training.
"""


def check_game_running(state, action):
    """Check if game is running (always true if we have state)"""
    return state is not None and 'player' in state


def check_player_name_set(state, action):
    """Check if player name is set"""
    name = state.get('player', {}).get('name', '').strip()
    return name not in ['', 'UNKNOWN', 'PLAYER']


def check_intro_cutscene_complete(state, action):
    """Check if intro cutscene is complete (in moving van)"""
    location = state.get('player', {}).get('location', '').upper()
    return 'MOVING' in location and 'VAN' in location


def check_littleroot_town(state, action):
    """Check if player reached Littleroot Town"""
    location = state.get('player', {}).get('location', '').upper()
    return 'LITTLEROOT' in location


def check_player_house_entered(state, action):
    """Check if player entered their house"""
    location = state.get('player', {}).get('location', '').upper()
    # Brendan's house 1F (not May's house, not 2F bedroom)
    return 'LITTLEROOT' in location and 'BRENDAN' in location and 'HOUSE' in location and '1F' in location


def check_player_bedroom(state, action):
    """Check if player reached their bedroom (2F)"""
    location = state.get('player', {}).get('location', '').upper()
    # Brendan's house 2F (player's bedroom)
    return 'LITTLEROOT' in location and 'BRENDAN' in location and 'HOUSE' in location and '2F' in location


def check_clock_set(state, action):
    """Check if player set the clock and left the house"""
    location = state.get('player', {}).get('location', '').upper()
    # In Littleroot but NOT in any house or lab (outside)
    return 'LITTLEROOT' in location and 'HOUSE' not in location and 'LAB' not in location


def check_rival_house(state, action):
    """Check if player entered rival's house"""
    location = state.get('player', {}).get('location', '').upper()
    # May's house 1F only (rival's house, not player's house)
    return 'LITTLEROOT' in location and 'MAY' in location and 'HOUSE' in location and '1F' in location


def check_rival_bedroom(state, action):
    """Check if player reached rival's bedroom"""
    location = state.get('player', {}).get('location', '').upper()
    # May's house 2F only (rival's bedroom)
    return 'LITTLEROOT' in location and 'MAY' in location and 'HOUSE' in location and '2F' in location


def check_route_101(state, action):
    """Check if player reached Route 101"""
    location = state.get('player', {}).get('location', '').upper()
    return 'ROUTE 101' in location or 'ROUTE_101' in location


def check_starter_chosen(state, action):
    """Check if player has a starter Pokemon (party size >= 1 with valid species)"""
    party = state.get('player', {}).get('party', [])
    # Check party size AND that at least one Pokemon has a valid species name
    return len(party) >= 1 and any(p.get('species', '').strip() or p.get('species_name', '').strip() for p in party)


def check_birch_lab_visited(state, action):
    """Check if player visited Birch's lab"""
    location = state.get('player', {}).get('location', '').upper()
    return 'LITTLEROOT' in location and 'LAB' in location


def check_oldale_town(state, action):
    """Check if player reached Oldale Town"""
    location = state.get('player', {}).get('location', '').upper()
    return 'OLDALE' in location


def check_route_103(state, action):
    """Check if player reached Route 103"""
    location = state.get('player', {}).get('location', '').upper()
    return 'ROUTE 103' in location or 'ROUTE_103' in location


def check_received_pokedex(state, action):
    """Check if player received Pokedex (back at Birch's lab after Route 103)"""
    # This is tricky - need to have been to Route 103 AND be back at lab
    # For simplicity, check for Pokedex in bag or just use dialog-based check
    # Alternative: Check if player is in Oldale/Route 102 area (after receiving Pokedex)
    location = state.get('player', {}).get('location', '').upper()
    party = state.get('player', {}).get('party', [])

    # Heuristic: Has starter + visited areas beyond Route 103
    # (Route 102, Petalburg, etc. are only accessible after Pokedex)
    has_starter = len(party) >= 1
    beyond_route_103 = any(loc in location for loc in ['ROUTE 102', 'ROUTE_102', 'PETALBURG', 'ROUTE 104', 'ROUTE_104'])

    return has_starter and beyond_route_103


def check_route_102(state, action):
    """Check if player reached Route 102"""
    location = state.get('player', {}).get('location', '').upper()
    return 'ROUTE 102' in location or 'ROUTE_102' in location


def check_petalburg_city(state, action):
    """Check if player reached Petalburg City"""
    location = state.get('player', {}).get('location', '').upper()
    return 'PETALBURG' in location


def check_dad_first_meeting(state, action):
    """Check if player met Dad at Petalburg Gym"""
    location = state.get('player', {}).get('location', '').upper()
    return 'PETALBURG' in location and 'GYM' in location


def check_gym_explanation(state, action):
    """Check if player received gym explanation from Dad (heuristic: left gym after meeting)"""
    location = state.get('player', {}).get('location', '').upper()
    # Heuristic: After meeting Dad in gym, player is now outside (Petalburg City but not in gym)
    # Or use dialog-based check from CUSTOM_MILESTONES (DAD_DIALOG_CONFIRMED)
    # For simplicity: assume completed after leaving gym
    return 'PETALBURG' in location and 'GYM' not in location and 'ROUTE' not in location


def check_route_104_south(state, action):
    """Check if player reached Route 104 (south section)"""
    location = state.get('player', {}).get('location', '').upper()
    return 'ROUTE 104' in location or 'ROUTE_104' in location


def check_petalburg_woods(state, action):
    """Check if player entered Petalburg Woods"""
    location = state.get('player', {}).get('location', '').upper()
    return 'PETALBURG' in location and 'WOODS' in location


def check_rustboro_city(state, action):
    """Check if player reached Rustboro City"""
    location = state.get('player', {}).get('location', '').upper()
    return 'RUSTBORO' in location


def check_rustboro_gym_entered(state, action):
    """Check if player entered Rustboro Gym"""
    location = state.get('player', {}).get('location', '').upper()
    return 'RUSTBORO' in location and 'GYM' in location


def check_stone_badge(state, action):
    """Check if player obtained Stone Badge (first gym badge)"""
    badges = state.get('game', {}).get('badges', [])
    # Stone badge is typically the first badge
    return len(badges) >= 1


# List of all server milestones to be registered as custom milestones
# Order matters: insert_after creates dependency chain
SERVER_MILESTONES = [
    {
        "id": "GAME_RUNNING",
        "insert_after": None,  # First milestone
        "description": "Complete title sequence and begin the game",
        "check_fn": check_game_running,
        "category": "system"
    },
    {
        "id": "PLAYER_NAME_SET",
        "insert_after": "GAME_RUNNING",
        "description": "Player has chosen their character name",
        "check_fn": check_player_name_set,
        "category": "intro"
    },
    {
        "id": "INTRO_CUTSCENE_COMPLETE",
        "insert_after": "PLAYER_NAME_SET",
        "description": "Complete intro cutscene with moving van",
        "check_fn": check_intro_cutscene_complete,
        "category": "intro"
    },
    {
        "id": "LITTLEROOT_TOWN",
        "insert_after": "INTRO_CUTSCENE_COMPLETE",
        "description": "Arrive at Littleroot Town",
        "check_fn": check_littleroot_town,
        "category": "location"
    },
    {
        "id": "PLAYER_HOUSE_ENTERED",
        "insert_after": "LITTLEROOT_TOWN",
        "description": "Enter player's house for the first time",
        "check_fn": check_player_house_entered,
        "category": "location"
    },
    {
        "id": "PLAYER_BEDROOM",
        "insert_after": "PLAYER_HOUSE_ENTERED",
        "description": "Go upstairs to player's bedroom",
        "check_fn": check_player_bedroom,
        "category": "location"
    },
    {
        "id": "CLOCK_SET",
        "insert_after": "CLOCK_INTERACT",  # After custom CLOCK_INTERACT milestone
        "description": "Set the clock in player's bedroom and leave the house",
        "check_fn": check_clock_set,
        "category": "task"
    },
    {
        "id": "RIVAL_HOUSE",
        "insert_after": "LEAVE_HOUSE",  # After custom LEAVE_HOUSE milestone
        "description": "Visit May's house next door",
        "check_fn": check_rival_house,
        "category": "location"
    },
    {
        "id": "RIVAL_BEDROOM",
        "insert_after": "RIVAL_HOUSE",
        "description": "Visit May's bedroom on the second floor",
        "check_fn": check_rival_bedroom,
        "category": "location"
    },
    {
        "id": "ROUTE_101",
        "insert_after": "RIVAL_BEDROOM",
        "description": "Travel to Route 101",
        "check_fn": check_route_101,
        "category": "location"
    },
    {
        "id": "STARTER_CHOSEN",
        "insert_after": "ROUTE_101",
        "description": "Choose starter Pokemon",
        "check_fn": check_starter_chosen,
        "category": "pokemon"
    },
    {
        "id": "BIRCH_LAB_VISITED",
        "insert_after": "STARTER_CHOSEN",
        "description": "Visit Professor Birch's lab",
        "check_fn": check_birch_lab_visited,
        "category": "location"
    },
    {
        "id": "OLDALE_TOWN",
        "insert_after": "BIRCH_LAB_VISITED",
        "description": "Arrive at Oldale Town",
        "check_fn": check_oldale_town,
        "category": "location"
    },
    {
        "id": "ROUTE_103",
        "insert_after": "OLDALE_TOWN",
        "description": "Travel to Route 103 and battle with May",
        "check_fn": check_route_103,
        "category": "location"
    },
    {
        "id": "RECEIVED_POKEDEX",
        "insert_after": "POKEDEX_DIALOG_CONFIRMED",  # After custom POKEDEX_DIALOG_CONFIRMED milestone
        "description": "Visit Birch's lab to receive Pokedex from Professor Birch",
        "check_fn": check_received_pokedex,
        "category": "item"
    },
    {
        "id": "ROUTE_102",
        "insert_after": "RECEIVED_POKEDEX",
        "description": "Travel to Route 102",
        "check_fn": check_route_102,
        "category": "location"
    },
    {
        "id": "PETALBURG_CITY",
        "insert_after": "ROUTE_102",
        "description": "Arrive at Petalburg City",
        "check_fn": check_petalburg_city,
        "category": "location"
    },
    {
        "id": "DAD_FIRST_MEETING",
        "insert_after": "PETALBURG_CITY",
        "description": "Meet Dad at Petalburg Gym",
        "check_fn": check_dad_first_meeting,
        "category": "story"
    },
    {
        "id": "GYM_EXPLANATION",
        "insert_after": "DAD_DIALOG_CONFIRMED",  # After custom DAD_DIALOG_CONFIRMED milestone
        "description": "Receive gym explanation from Dad",
        "check_fn": check_gym_explanation,
        "category": "story"
    },
    {
        "id": "ROUTE_104_SOUTH",
        "insert_after": "GYM_EXPLANATION",
        "description": "Travel to Route 104 (South)",
        "check_fn": check_route_104_south,
        "category": "location"
    },
    {
        "id": "PETALBURG_WOODS",
        "insert_after": "ROUTE_104_SOUTH",
        "description": "Navigate through Petalburg Woods",
        "check_fn": check_petalburg_woods,
        "category": "location"
    },
    {
        "id": "RUSTBORO_CITY",
        "insert_after": "PETALBURG_WOODS",
        "description": "Arrive at Rustboro City",
        "check_fn": check_rustboro_city,
        "category": "location"
    },
    {
        "id": "RUSTBORO_GYM_ENTERED",
        "insert_after": "RUSTBORO_CITY",
        "description": "Enter Rustboro Gym",
        "check_fn": check_rustboro_gym_entered,
        "category": "location"
    },
    {
        "id": "STONE_BADGE",
        "insert_after": "RUSTBORO_GYM_ENTERED",
        "description": "Defeat Gym Leader Roxanne and receive Stone Badge (first gym badge)",
        "check_fn": check_stone_badge,
        "category": "badge"
    },
]