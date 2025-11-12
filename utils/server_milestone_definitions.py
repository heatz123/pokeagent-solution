#!/usr/bin/env python3
"""
Server milestone definitions as custom milestones (state-based checks only)

All milestones are converted to Python check functions that examine game state.
This allows full Python control over milestone completion conditions for RL training.
"""


def check_game_running(state, action):
    """Check if game is running (always true if we have state)"""
    return state is not None and 'player' in state


def check_littleroot_town(state, action):
    """Check if player reached Littleroot Town"""
    location = state.get('player', {}).get('location', '').upper()
    return 'LITTLEROOT' in location


def check_player_house_entered(state, action):
    """Check if player entered their house"""
    location = state.get('player', {}).get('location', '').upper()
    # Littleroot + House, but NOT May's house (rival house)
    return 'LITTLEROOT' in location and 'HOUSE' in location and 'MAY' not in location and 'BRENDAN' not in location


def check_player_bedroom(state, action):
    """Check if player reached their bedroom (2F)"""
    location = state.get('player', {}).get('location', '').upper()
    # Player's house + 2F indicates bedroom
    return 'LITTLEROOT' in location and '2F' in location and 'MAY' not in location and 'BRENDAN' not in location


def check_rival_house(state, action):
    """Check if player entered rival's house"""
    location = state.get('player', {}).get('location', '').upper()
    # May's or Brendan's house
    return 'LITTLEROOT' in location and ('MAY' in location or 'BRENDAN' in location) and 'HOUSE' in location


def check_rival_bedroom(state, action):
    """Check if player reached rival's bedroom"""
    location = state.get('player', {}).get('location', '').upper()
    # Rival's house + 2F
    return 'LITTLEROOT' in location and ('MAY' in location or 'BRENDAN' in location) and '2F' in location


def check_route_101(state, action):
    """Check if player reached Route 101"""
    location = state.get('player', {}).get('location', '').upper()
    return 'ROUTE 101' in location or 'ROUTE_101' in location


def check_starter_chosen(state, action):
    """Check if player has a starter Pokemon (party size >= 1)"""
    party = state.get('player', {}).get('party', [])
    return len(party) >= 1


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
        "description": "Game is running",
        "check_fn": check_game_running,
        "category": "system"
    },
    {
        "id": "LITTLEROOT_TOWN",
        "insert_after": "GAME_RUNNING",
        "description": "Arrived at Littleroot Town",
        "check_fn": check_littleroot_town,
        "category": "location"
    },
    {
        "id": "PLAYER_HOUSE_ENTERED",
        "insert_after": "LITTLEROOT_TOWN",
        "description": "Entered player's house",
        "check_fn": check_player_house_entered,
        "category": "location"
    },
    {
        "id": "PLAYER_BEDROOM",
        "insert_after": "PLAYER_HOUSE_ENTERED",
        "description": "Reached player's bedroom (2F)",
        "check_fn": check_player_bedroom,
        "category": "location"
    },
    {
        "id": "RIVAL_HOUSE",
        "insert_after": "PLAYER_BEDROOM",
        "description": "Entered rival's house",
        "check_fn": check_rival_house,
        "category": "location"
    },
    {
        "id": "RIVAL_BEDROOM",
        "insert_after": "RIVAL_HOUSE",
        "description": "Reached rival's bedroom",
        "check_fn": check_rival_bedroom,
        "category": "location"
    },
    {
        "id": "ROUTE_101",
        "insert_after": "RIVAL_BEDROOM",
        "description": "Reached Route 101",
        "check_fn": check_route_101,
        "category": "location"
    },
    {
        "id": "STARTER_CHOSEN",
        "insert_after": "ROUTE_101",
        "description": "Chose starter Pokemon",
        "check_fn": check_starter_chosen,
        "category": "pokemon"
    },
    {
        "id": "BIRCH_LAB_VISITED",
        "insert_after": "STARTER_CHOSEN",
        "description": "Visited Professor Birch's lab",
        "check_fn": check_birch_lab_visited,
        "category": "location"
    },
    {
        "id": "OLDALE_TOWN",
        "insert_after": "BIRCH_LAB_VISITED",
        "description": "Reached Oldale Town",
        "check_fn": check_oldale_town,
        "category": "location"
    },
    {
        "id": "ROUTE_103",
        "insert_after": "OLDALE_TOWN",
        "description": "Reached Route 103",
        "check_fn": check_route_103,
        "category": "location"
    },
    {
        "id": "RECEIVED_POKEDEX",
        "insert_after": "ROUTE_103",
        "description": "Received Pokedex from Professor Birch",
        "check_fn": check_received_pokedex,
        "category": "item"
    },
    {
        "id": "ROUTE_102",
        "insert_after": "RECEIVED_POKEDEX",
        "description": "Reached Route 102",
        "check_fn": check_route_102,
        "category": "location"
    },
    {
        "id": "PETALBURG_CITY",
        "insert_after": "ROUTE_102",
        "description": "Reached Petalburg City",
        "check_fn": check_petalburg_city,
        "category": "location"
    },
    {
        "id": "ROUTE_104_SOUTH",
        "insert_after": "PETALBURG_CITY",
        "description": "Reached Route 104 (south section)",
        "check_fn": check_route_104_south,
        "category": "location"
    },
    {
        "id": "PETALBURG_WOODS",
        "insert_after": "ROUTE_104_SOUTH",
        "description": "Entered Petalburg Woods",
        "check_fn": check_petalburg_woods,
        "category": "location"
    },
    {
        "id": "RUSTBORO_CITY",
        "insert_after": "PETALBURG_WOODS",
        "description": "Reached Rustboro City",
        "check_fn": check_rustboro_city,
        "category": "location"
    },
    {
        "id": "STONE_BADGE",
        "insert_after": "RUSTBORO_CITY",
        "description": "Obtained Stone Badge (defeated Roxanne)",
        "check_fn": check_stone_badge,
        "category": "badge"
    },
]