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


def find_path_action(state: dict, goal_x: int, goal_y: int, max_distance: int = 50) -> str:
    """
    Find next action to reach goal using A* pathfinding.

    Args:
        state: CodeAgent state dict
        goal_x: Target x coordinate (absolute world coordinates)
        goal_y: Target y coordinate (absolute world coordinates)
        max_distance: Maximum search distance (default: 50)

    Returns:
        str: Next action ('up', 'down', 'left', 'right', 'no_op')

    Example:
        # Navigate to stairs at position (10, 5)
        action = find_path_action(state, goal_x=10, goal_y=5)
        return action

        # Common pattern: Navigate to another map when warp coordinates are unknown
        # but connection data shows it's connected to the left/right/top/bottom edge
        if map == "PALLET_TOWN":
            # Connection shows next map is connected to left edge at y=7
            action = find_path_action(state, goal_x=-1, goal_y=7)
            return action
    """
    try:
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

        # 2. Parse ASCII map to get blocked positions and ledges
        blocked, ledges = _parse_ascii_map(state)

        # 3. Run A* pathfinding with ledge awareness
        path = _astar(start_x, start_y, goal_x, goal_y, blocked, ledges, max_distance)

        if not path or len(path) < 2:
            logger.info(f"No path found from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
            return 'no_op'

        # 4. Convert first step to action
        action = _path_to_action(path[0], path[1])
        logger.info(f"Pathfinding: ({start_x}, {start_y}) → ({goal_x}, {goal_y}), action: {action}")

        return action

    except Exception as e:
        logger.error(f"Pathfinding error: {e}")
        import traceback
        traceback.print_exc()
        return 'no_op'


def _parse_ascii_map(state: dict) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    """
    Parse ASCII map from state to extract blocked positions and ledges.

    Returns:
        tuple: (blocked_set, ledges_dict)
            - blocked_set: Set of (x, y) coordinates that are impassable
            - ledges_dict: Dict mapping (x, y) to ledge direction symbol
    """
    map_data = state.get('map', {})
    ascii_map = map_data.get('ascii_map', '')
    player_position = map_data.get('player_position', {})

    if not ascii_map or not player_position:
        logger.warning("No ASCII map or player position in state")
        return set(), {}

    # Get player position in ASCII map grid (row, col)
    # Note: player_position uses 'y' for row and 'x' for col
    player_row = player_position.get('y', 0)
    player_col = player_position.get('x', 0)

    # Get player world coordinates
    player_data = state.get('player', {})
    player_pos = player_data.get('position', {})
    player_world_x = player_pos.get('x', 0)
    player_world_y = player_pos.get('y', 0)

    # Parse ASCII map into 2D grid
    # ascii_map can be either a string or a list of strings
    if isinstance(ascii_map, list):
        lines = ascii_map
    else:
        lines = ascii_map.strip().split('\n')

    blocked = set()
    ledges = {}

    # Ledge symbols (one-way directional jumps)
    ledge_symbols = {'↓', '↑', '←', '→', '↗', '↖', '↘', '↙'}

    # Walkable symbols (from map_formatter.py)
    walkable_symbols = {'.', '~', 'D', 'S', 'P', 'c', 'T', '?', 'F', 't'} | ledge_symbols

    for row_idx, line in enumerate(lines):
        for col_idx, symbol in enumerate(line):
            # Convert grid coordinates to world coordinates
            world_x = player_world_x + (col_idx - player_col)
            world_y = player_world_y + (row_idx - player_row)

            # Track ledges separately
            if symbol in ledge_symbols:
                ledges[(world_x, world_y)] = symbol

            # Mark as blocked if not walkable
            if symbol not in walkable_symbols:
                blocked.add((world_x, world_y))

    return blocked, ledges


def _astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked: Set[Tuple[int, int]],
    ledges: Dict[Tuple[int, int], str],
    max_distance: int
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm with ledge awareness.

    If goal is unreachable, returns path to closest reachable position.
    """
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

            if neighbor_pos in blocked:
                continue

            # Check ledge restrictions (one-way movement)
            if neighbor_pos in ledges:
                if not _can_enter_ledge(ledges[neighbor_pos], dx, dy):
                    continue

            if neighbor_pos in closed_set:
                continue

            tentative_g = g_cost + 1

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
