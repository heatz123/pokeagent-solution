#!/usr/bin/env python3
"""
Pathfinding tool for CodeAgent

Provides A* pathfinding to navigate around obstacles.
This file is auto-loaded - no import needed in LLM code!
"""

import heapq
import logging
from typing import Tuple, Set, List, Optional

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

        # 2. Parse ASCII map to get blocked positions
        blocked = _parse_ascii_map(state)

        # 3. Run A* pathfinding
        path = _astar(start_x, start_y, goal_x, goal_y, blocked, max_distance)

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


def _parse_ascii_map(state: dict) -> Set[Tuple[int, int]]:
    """Parse ASCII map from state to extract blocked positions."""
    map_data = state.get('map', {})
    ascii_map = map_data.get('ascii_map', '')
    player_position = map_data.get('player_position', {})

    if not ascii_map or not player_position:
        logger.warning("No ASCII map or player position in state")
        return set()

    # Get player position in ASCII map grid (row, col)
    player_row = player_position.get('row', 0)
    player_col = player_position.get('col', 0)

    # Get player world coordinates
    player_data = state.get('player', {})
    player_pos = player_data.get('position', {})
    player_world_x = player_pos.get('x', 0)
    player_world_y = player_pos.get('y', 0)

    # Parse ASCII map into 2D grid
    lines = ascii_map.strip().split('\n')
    blocked = set()

    # Walkable symbols (from map_formatter.py)
    walkable_symbols = {'.', '~', 'D', 'S', 'P', 'c', 'T', '?', 'F', 't',
                        '↓', '↑', '←', '→', '↗', '↖', '↘', '↙'}

    for row_idx, line in enumerate(lines):
        for col_idx, symbol in enumerate(line):
            # Convert grid coordinates to world coordinates
            world_x = player_world_x + (col_idx - player_col)
            world_y = player_world_y + (row_idx - player_row)

            # Mark as blocked if not walkable
            if symbol not in walkable_symbols:
                blocked.add((world_x, world_y))

    return blocked


def _astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked: Set[Tuple[int, int]],
    max_distance: int
) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding algorithm."""
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

        # Check all neighbors (up, down, left, right)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor_x = x + dx
            neighbor_y = y + dy
            neighbor_pos = (neighbor_x, neighbor_y)

            if neighbor_pos in blocked:
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

    return None


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
