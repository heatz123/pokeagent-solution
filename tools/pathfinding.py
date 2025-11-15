#!/usr/bin/env python3
"""
Pathfinding tool for CodeAgent

Provides A* pathfinding to navigate around obstacles.
This file is auto-loaded - no import needed in LLM code!
"""

import heapq
import logging
from typing import Tuple, Set, List, Optional, Dict

logger = logging.getLogger(__name__)


def find_path_action(state: dict, goal_x: int, goal_y: int, use_vlm_fallback: bool = True) -> str:
    """
    Find next action to reach goal using A* pathfinding.

    Automatically handles:
    - Dialogs: Press 'a' to advance/dismiss (state-based + VLM fallback)
    - NPCs: When goal is NPC, press 'a' when adjacent and facing

    Args:
        state: CodeAgent state dict
        goal_x: Target x coordinate (absolute world coordinates)
        goal_y: Target y coordinate (absolute world coordinates)
        use_vlm_fallback: If True, use VLM to double-check for dialogs (slower but more accurate)

    Returns:
        str: Next action ('up', 'down', 'left', 'right', 'a', 'no_op')

    Example:
        # Navigate to stairs at position (10, 5)
        action = find_path_action(state, goal_x=10, goal_y=5)
        return action

        # Navigate to NPC and interact
        action = find_path_action(state, goal_x=12, goal_y=8)
        return action

        # Common pattern: Navigate to another map
        # when warp coordinates are unknown
        if map == "PALLET_TOWN":
            # Connection shows next map is at left edge y=7
            action = find_path_action(state, goal_x=-1, goal_y=7)
            return action
    """
    try:
        # 0. Check for dialogs/text boxes
        # First try state dict (fast, no VLM cost)
        dialog_text = state.get('game', {}).get('dialog_text')

        # Dialog exists if not None and not empty string
        if dialog_text and len(str(dialog_text).strip()) > 0:
            logger.info(f"Dialog detected (state): '{str(dialog_text)[:50]}...', pressing 'a'")
            return 'a'

        # VLM fallback: Sometimes state.dialog_text is not updated properly
        # Use VLM to detect dialogs from screenshot
        if use_vlm_fallback:
            try:
                from tools.dialog_checker import is_dialog_open
                if is_dialog_open(state):
                    logger.info("Dialog detected (VLM fallback), pressing 'a'")
                    return 'a'
            except ImportError:
                logger.warning("dialog_checker not available, skipping VLM fallback")
            except Exception as e:
                logger.warning(f"VLM dialog detection failed: {e}")

        # 1. Extract current position (world coordinates)
        player_data = state.get('player', {})
        player_pos = player_data.get('position', {})
        start_x = player_pos.get('x')
        start_y = player_pos.get('y')

        if start_x is None or start_y is None:
            logger.warning("No player position in state")
            return 'no_op'

        # Check if already at goal
        if start_x == goal_x and start_y == goal_y:
            logger.info(f"Already at goal ({goal_x}, {goal_y})")
            return 'no_op'

        # 2. Parse ASCII map to get blocked, ledges, NPCs, grass, sight ranges, and bounds
        blocked, ledges, npcs, grass, sight_range, bounds = _parse_ascii_map(state)

        # 3. Check if goal is NPC and can interact -> press 'a'
        if (goal_x, goal_y) in npcs:
            player_facing = state.get('facing', 'north')

            # Case 1: Adjacent and facing (distance = 1)
            if _is_adjacent_and_facing(
                start_x, start_y, goal_x, goal_y, player_facing
            ):
                logger.info(
                    f"Adjacent to NPC at ({goal_x}, {goal_y}) "
                    f"and facing, pressing 'a'"
                )
                return 'a'

            # Case 2: Distance 2 with counter/desk between (e.g., mart, pokemon center)
            if _can_interact_through_counter(
                start_x, start_y, goal_x, goal_y, player_facing, blocked, state
            ):
                logger.info(
                    f"Can interact with NPC at ({goal_x}, {goal_y}) "
                    f"through counter, pressing 'a'"
                )
                return 'a'

        # 4. Run A* pathfinding with ledge awareness, grass avoidance (penalty=2), and sight range avoidance (penalty=10)
        path = _astar(start_x, start_y, goal_x, goal_y, blocked, ledges, grass, sight_range, bounds, 50)

        if not path:
            logger.info(
                f"No path found from "
                f"({start_x}, {start_y}) to ({goal_x}, {goal_y})"
            )
            return 'no_op'

        # 5. Convert first step to action
        if len(path) >= 2:
            # Normal case: follow path
            action = _path_to_action(path[0], path[1])
            logger.info(
                f"Pathfinding: ({start_x}, {start_y}) → "
                f"({goal_x}, {goal_y}), action: {action}"
            )
            return action

        # Special case: reached end of path (len(path) == 1)
        # If goal is outside map bounds, move towards it to trigger map transition
        min_x, max_x, min_y, max_y = bounds
        if goal_x < min_x or goal_x > max_x or goal_y < min_y or goal_y > max_y:
            # Determine which boundary to cross based on goal position
            action = None
            if goal_x < min_x:
                action = 'left'
            elif goal_x > max_x:
                action = 'right'
            elif goal_y < min_y:
                action = 'up'
            elif goal_y > max_y:
                action = 'down'
            else:
                action = 'no_op'  # Shouldn't happen

            logger.info(
                f"At map boundary, moving towards goal outside map: "
                f"({start_x}, {start_y}) → ({goal_x}, {goal_y}), action: {action}"
            )
            return action

        # Already at goal (shouldn't happen due to check at top, but just in case)
        logger.info(f"Already at goal ({goal_x}, {goal_y})")
        return 'no_op'

    except Exception as e:
        logger.error(f"Pathfinding error: {e}")
        import traceback
        traceback.print_exc()
        return 'no_op'


def _parse_ascii_map(state: dict) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], str], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], Tuple[int, int, int, int]]:
    """
    Parse ASCII map from state to extract blocked positions, ledges, NPCs, grass, sight ranges, and map bounds.

    Returns:
        tuple: (blocked_set, ledges_dict, npc_set, grass_set, sight_range_set, bounds)
            - blocked_set: Set of (x, y) coordinates that are impassable
            - ledges_dict: Dict mapping (x, y) to ledge direction symbol
            - npc_set: Set of (x, y) coordinates with NPCs or Trainers
            - grass_set: Set of (x, y) coordinates with tall grass (wild encounters)
            - sight_range_set: Set of (x, y) coordinates in NPC sight ranges (high cost but passable)
            - bounds: Tuple (min_x, max_x, min_y, max_y) defining the map boundaries
    """
    map_data = state.get('map', {})
    ascii_map = map_data.get('ascii_map', '')

    if not ascii_map:
        logger.warning("No ASCII map in state")
        return set(), {}, set(), set(), set(), (0, 0, 0, 0)

    # Parse ASCII map into lines
    if isinstance(ascii_map, list):
        lines = ascii_map
    else:
        lines = ascii_map.strip().split('\n')

    if not lines:
        return set(), {}, set(), set(), set(), (0, 0, 0, 0)

    # Get player world coordinates (always use player['position'] as source of truth)
    player_data = state.get('player', {})
    player_pos = player_data.get('position', {})
    player_world_x = player_pos.get('x')
    player_world_y = player_pos.get('y')

    if player_world_x is None or player_world_y is None:
        logger.warning("No player world coordinates in state")
        return set(), {}, set(), set(), set(), (0, 0, 0, 0)

    # In this coordinate system, grid coordinates = world coordinates
    # So player's grid position is the same as world position
    player_row = player_world_y
    player_col = player_world_x

    blocked = set()
    ledges = {}
    npcs = set()
    grass = set()  # Track grass tiles for avoidance (grass_penalty=2 hardcoded)
    sight_range = set()  # Track NPC sight ranges (sight_penalty=10 hardcoded)

    # Track map bounds (min/max world coordinates)
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Ledge symbols (one-way directional jumps)
    ledge_symbols = {'↓', '↑', '←', '→', '↗', '↖', '↘', '↙'}

    # Interactable object symbols (NPCs and objects that can be interacted with)
    npc_symbols = {
        'N',   # NPC (actual position)
        'c',   # PC/Computer
        'T',   # Television
        'B',   # Bookshelf
        '?',   # Sign/Information
        'C',   # Counter/Desk
        'K',   # Clock (Wall)
        'O',   # Clock
        '^',   # Picture/Painting
        'U',   # Trash can
        'V',   # Pot/Vase
        'M',   # Machine/Device
    }

    # Walkable symbols (from map_formatter.py)
    # '@' is now walkable but with high cost (NPC sight range)
    walkable_symbols = {'.', '~', 'D', 'S', 'P', '?', 'F', '@'} | ledge_symbols

    for row_idx, line in enumerate(lines):
        for col_idx, symbol in enumerate(line):
            # Convert grid coordinates to world coordinates
            world_x = player_world_x + (col_idx - player_col)
            world_y = player_world_y + (row_idx - player_row)

            # Update bounds
            min_x = min(min_x, world_x)
            max_x = max(max_x, world_x)
            min_y = min(min_y, world_y)
            max_y = max(max_y, world_y)

            # Track NPC sight ranges (high cost, passable)
            if symbol == '@':
                sight_range.add((world_x, world_y))

            # Track NPCs and interactable objects
            if symbol in npc_symbols:
                npcs.add((world_x, world_y))

            # Track grass tiles (for encounter avoidance)
            if symbol == '~':
                grass.add((world_x, world_y))

            # Track ledges separately
            if symbol in ledge_symbols:
                ledges[(world_x, world_y)] = symbol

            # Mark as blocked if not walkable
            # (objects are both interactable AND blocked, except '@')
            if symbol not in walkable_symbols:
                blocked.add((world_x, world_y))

    bounds = (int(min_x), int(max_x), int(min_y), int(max_y))
    return blocked, ledges, npcs, grass, sight_range, bounds


def _can_interact_through_counter(
    player_x: int,
    player_y: int,
    target_x: int,
    target_y: int,
    facing: str,
    blocked: Set[Tuple[int, int]],
    state: dict
) -> bool:
    """
    Check if player can interact with NPC through a counter (distance 2).

    This handles cases like Pokemon Centers and Marts where the NPC is behind
    a counter, requiring interaction from distance 2 instead of 1.

    Args:
        player_x: Player's x coordinate
        player_y: Player's y coordinate
        target_x: Target NPC's x coordinate
        target_y: Target NPC's y coordinate
        facing: Player's facing direction
        blocked: Set of blocked positions
        state: Full game state (for ASCII map)

    Returns:
        True if can interact through counter
    """
    # Calculate direction and distance
    dx = target_x - player_x
    dy = target_y - player_y
    manhattan_dist = abs(dx) + abs(dy)

    # Must be exactly distance 2
    if manhattan_dist != 2:
        return False

    # Must be in a straight line (same x or same y, not diagonal)
    if dx != 0 and dy != 0:
        return False

    # Check if facing the right direction
    facing_lower = facing.lower() if facing else ""

    # Determine expected facing based on direction to target
    if dy == -2 and dx == 0:  # Target is 2 tiles north
        if facing_lower not in ['north', 'up']:
            return False
        middle_tile = (player_x, player_y - 1)
    elif dy == 2 and dx == 0:  # Target is 2 tiles south
        if facing_lower not in ['south', 'down']:
            return False
        middle_tile = (player_x, player_y + 1)
    elif dx == -2 and dy == 0:  # Target is 2 tiles west
        if facing_lower not in ['west', 'left']:
            return False
        middle_tile = (player_x - 1, player_y)
    elif dx == 2 and dy == 0:  # Target is 2 tiles east
        if facing_lower not in ['east', 'right']:
            return False
        middle_tile = (player_x + 1, player_y)
    else:
        return False

    # Check if the middle tile is a counter or desk
    # Get ASCII map to check the symbol
    map_data = state.get('map', {})
    ascii_map = map_data.get('ascii_map', '')

    if not ascii_map:
        return False

    # Parse ASCII map
    if isinstance(ascii_map, list):
        lines = ascii_map
    else:
        lines = ascii_map.strip().split('\n')

    if not lines:
        return False

    # Get player position in grid
    player_data = state.get('player', {})
    player_pos = player_data.get('position', {})
    player_world_x = player_pos.get('x')
    player_world_y = player_pos.get('y')

    if player_world_x is None or player_world_y is None:
        return False

    # Convert middle tile world coords to grid coords
    middle_world_x, middle_world_y = middle_tile

    # Find grid position for middle tile
    # In the ASCII map, player is typically at center
    player_row = player_world_y
    player_col = player_world_x

    for row_idx, line in enumerate(lines):
        for col_idx, symbol in enumerate(line):
            # Convert grid to world coords
            world_x = player_world_x + (col_idx - player_col)
            world_y = player_world_y + (row_idx - player_row)

            # Check if this is the middle tile
            if world_x == middle_world_x and world_y == middle_world_y:
                # Check if it's a counter, desk, or similar interactable obstacle
                # 'C' = Counter/Desk (from map_formatter.py line 89-90)
                if symbol == 'C':
                    return True
                # Also allow interaction through regular blocked tiles (#)
                # This handles cases where Counter behavior wasn't detected
                # (e.g., missing tileset files for Pokemon Centers)
                # Safe because we already check:
                # 1. Goal must be NPC (not just any blocked tile)
                # 2. Player must be facing the NPC
                # 3. Distance must be exactly 2 in straight line
                if symbol == '#':
                    return True
                # Not a counter or blocked tile
                return False

    return False


def _is_adjacent_and_facing(player_x: int, player_y: int, target_x: int, target_y: int, facing: str) -> bool:
    """
    Check if player is adjacent to target and facing it.

    Args:
        player_x: Player's x coordinate
        player_y: Player's y coordinate
        target_x: Target's x coordinate
        target_y: Target's y coordinate
        facing: Player's facing direction (e.g., "North", "South", "East", "West")

    Returns:
        True if player is adjacent to target and facing it
    """
    # Calculate direction from player to target
    dx = target_x - player_x
    dy = target_y - player_y

    # Check if adjacent (Manhattan distance = 1, no diagonal)
    if abs(dx) + abs(dy) != 1:
        return False

    # Map facing direction to expected delta
    facing_lower = facing.lower() if facing else ""

    if dy == -1 and dx == 0:  # Target is north
        return facing_lower in ['north', 'up']
    elif dy == 1 and dx == 0:  # Target is south
        return facing_lower in ['south', 'down']
    elif dx == -1 and dy == 0:  # Target is west
        return facing_lower in ['west', 'left']
    elif dx == 1 and dy == 0:  # Target is east
        return facing_lower in ['east', 'right']

    return False


def _astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked: Set[Tuple[int, int]],
    ledges: Dict[Tuple[int, int], str],
    grass: Set[Tuple[int, int]],
    sight_range: Set[Tuple[int, int]],
    bounds: Tuple[int, int, int, int],
    max_distance: int
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm with ledge awareness, grass avoidance, and NPC sight range avoidance.

    Cost penalties:
    - Grass tiles: 2x cost (avoid wild Pokemon encounters)
    - NPC sight range (@): 10x cost (avoid triggering trainer battles)

    Map bounds enforcement:
    - Only explores positions within bounds (min_x, max_x, min_y, max_y)
    - Goals outside bounds will find closest reachable position at boundary

    If goal is unreachable, returns path to closest reachable position.
    """
    min_x, max_x, min_y, max_y = bounds
    start_node = (
        _heuristic(start_x, start_y, goal_x, goal_y),  # f_cost
        0,  # g_cost
        start_x,
        start_y,
        None  # parent
    )

    open_list = [start_node]
    closed_set = set()
    came_from = {}
    g_score = {(start_x, start_y): 0}

    # Track closest position to goal (for best-effort pathfinding)
    best_pos = (start_x, start_y)
    best_distance = _heuristic(start_x, start_y, goal_x, goal_y)

    while open_list:
        current = heapq.heappop(open_list)
        f_cost, g_cost, x, y, parent = current

        # Check if reached goal
        if x == goal_x and y == goal_y:
            return _reconstruct_path(came_from, (x, y))

        # Check if searched too far
        if g_cost > max_distance:
            continue

        # Mark as visited
        pos = (x, y)
        if pos in closed_set:
            continue
        closed_set.add(pos)

        # Update best position if closer to goal
        current_distance = _heuristic(x, y, goal_x, goal_y)
        if current_distance < best_distance:
            best_pos = pos
            best_distance = current_distance

        # Check all neighbors (up, down, left, right)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor_x = x + dx
            neighbor_y = y + dy
            neighbor_pos = (neighbor_x, neighbor_y)

            # Check if neighbor is outside map bounds
            if neighbor_x < min_x or neighbor_x > max_x or neighbor_y < min_y or neighbor_y > max_y:
                continue

            if neighbor_pos in blocked:
                continue

            # Check ledge restrictions (one-way movement)
            if neighbor_pos in ledges:
                if not _can_enter_ledge(ledges[neighbor_pos], dx, dy):
                    continue

            if neighbor_pos in closed_set:
                continue

            # Apply movement cost penalties
            move_cost = 1

            # Grass penalty: 2x cost to avoid wild encounters
            if neighbor_pos in grass:
                move_cost = 2

            # NPC sight range penalty: 10x cost to avoid trainer battles
            # This is higher priority than grass avoidance
            if neighbor_pos in sight_range:
                move_cost = 10

            tentative_g = g_cost + move_cost

            if neighbor_pos in g_score and tentative_g >= g_score[neighbor_pos]:
                continue

            came_from[neighbor_pos] = pos
            g_score[neighbor_pos] = tentative_g

            h_cost = _heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
            f_cost = tentative_g + h_cost
            neighbor_node = (f_cost, tentative_g, neighbor_x, neighbor_y, pos)
            heapq.heappush(open_list, neighbor_node)

    # Goal unreachable - return path to closest position
    if best_pos != (start_x, start_y):
        logger.info(f"Goal ({goal_x}, {goal_y}) unreachable, moving to closest point {best_pos} (distance: {best_distance})")
        return _reconstruct_path(came_from, best_pos)

    return None


def _can_enter_ledge(ledge_symbol: str, dx: int, dy: int) -> bool:
    """
    Check if movement in direction (dx, dy) can enter a ledge.

    Ledges are one-way jumps:
    - ↓ (Jump South): Can only enter from north (moving down, dy > 0)
    - ↑ (Jump North): Can only enter from south (moving up, dy < 0)
    - ← (Jump West): Can only enter from east (moving left, dx < 0)
    - → (Jump East): Can only enter from west (moving right, dx > 0)
    - Diagonal ledges: combination of above

    Args:
        ledge_symbol: Ledge direction symbol
        dx: X movement delta (-1, 0, or 1)
        dy: Y movement delta (-1, 0, or 1)

    Returns:
        True if can enter ledge from this direction
    """
    # Map ledge symbols to allowed entry directions
    ledge_rules = {
        '↓': (0, 1),    # Jump south: enter from north (moving down)
        '↑': (0, -1),   # Jump north: enter from south (moving up)
        '←': (-1, 0),   # Jump west: enter from east (moving left)
        '→': (1, 0),    # Jump east: enter from west (moving right)
        '↗': (1, -1),   # Jump northeast: enter moving right+up
        '↖': (-1, -1),  # Jump northwest: enter moving left+up
        '↘': (1, 1),    # Jump southeast: enter moving right+down
        '↙': (-1, 1),   # Jump southwest: enter moving left+down
    }

    allowed_direction = ledge_rules.get(ledge_symbol)
    if not allowed_direction:
        # Unknown ledge symbol, treat as walkable
        return True

    # Check if movement direction matches allowed direction
    return (dx, dy) == allowed_direction


def _heuristic(x1: int, y1: int, x2: int, y2: int) -> int:
    """Manhattan distance heuristic for grid movement."""
    return abs(x1 - x2) + abs(y1 - y2)


def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from dict."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _path_to_action(current_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> str:
    """Convert position step to action."""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    if dx > 0:
        return 'right'
    elif dx < 0:
        return 'left'
    elif dy > 0:
        return 'down'
    elif dy < 0:
        return 'up'
    else:
        return 'no_op'
