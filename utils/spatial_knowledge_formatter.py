"""
Spatial Knowledge Formatter

Provides utilities to format and parse structured spatial knowledge entries
for objects, NPCs, and warps with coordinates.

Format Convention:
- [OBJECT] <description> at (<x>, <y>) in <location>
- [OBJECT] <description> at (<x>, <y>) in <location>: <extra_desc>
- [NPC] <name> at (<x>, <y>) in <location>
- [NPC] <name> at (<x>, <y>) in <location>: <extra_desc>
- [WARP] <type> at (<x>, <y>) in <location>
- [WARP] <type> at (<x>, <y>) in <location> -> <destination>
- [WARP] <type> at (<x>, <y>) in <location> -> <destination>: <extra_desc>
"""

import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Prefix constants
PREFIX_OBJECT = "[OBJECT]"
PREFIX_NPC = "[NPC]"
PREFIX_WARP = "[WARP]"


def format_object(description: str, x: int, y: int, location: str, extra_desc: str = None) -> str:
    """
    Format an object knowledge entry

    Args:
        description: Object description (e.g., "Television", "PC", "Sign")
        x: X coordinate
        y: Y coordinate
        location: Location name
        extra_desc: Optional additional description

    Returns:
        Formatted knowledge string

    Example:
        format_object("Television", 4, 1, "Brendan's House 2F")
        -> "[OBJECT] Television at (4, 1) in Brendan's House 2F"

        format_object("PC", 5, 1, "Brendan's House 2F", "Used to store Pokemon")
        -> "[OBJECT] PC at (5, 1) in Brendan's House 2F: Used to store Pokemon"
    """
    base = f"{PREFIX_OBJECT} {description} at ({x}, {y}) in {location}"
    if extra_desc:
        return f"{base}: {extra_desc}"
    return base


def format_npc(name: str, x: int, y: int, location: str, trainer_type: int = None, extra_desc: str = None) -> str:
    """
    Format an NPC knowledge entry

    Args:
        name: NPC name or description
        x: X coordinate
        y: Y coordinate
        location: Location name
        trainer_type: Optional trainer type (0=NPC, >0=trainer)
        extra_desc: Optional additional description

    Returns:
        Formatted knowledge string

    Example:
        format_npc("Mom", 8, 2, "Brendan's House 1F")
        -> "[NPC] Mom at (8, 2) in Brendan's House 1F"

        format_npc("Mom", 8, 2, "Brendan's House 1F", extra_desc="Blocks stairs initially")
        -> "[NPC] Mom at (8, 2) in Brendan's House 1F: Blocks stairs initially"

        format_npc("Youngster", 5, 10, "Route 103", trainer_type=1)
        -> "[NPC] Youngster (trainer) at (5, 10) in Route 103"
    """
    npc_name = name
    if trainer_type is not None and trainer_type > 0:
        npc_name = f"{name} (trainer)"

    base = f"{PREFIX_NPC} {npc_name} at ({x}, {y}) in {location}"
    if extra_desc:
        return f"{base}: {extra_desc}"
    return base


def format_warp(warp_type: str, x: int, y: int, location: str, destination: str = None, extra_desc: str = None) -> str:
    """
    Format a warp/door/stairs knowledge entry

    Args:
        warp_type: Type of warp (e.g., "door", "stairs", "warp")
        x: X coordinate
        y: Y coordinate
        location: Current location name
        destination: Optional destination location
        extra_desc: Optional additional description

    Returns:
        Formatted knowledge string

    Example:
        format_warp("door", 7, 1, "Brendan's House 2F", "Brendan's House 1F")
        -> "[WARP] door at (7, 1) in Brendan's House 2F -> Brendan's House 1F"

        format_warp("stairs", 8, 2, "Brendan's House 1F", extra_desc="Leads to bedroom after Mom moves")
        -> "[WARP] stairs at (8, 2) in Brendan's House 1F: Leads to bedroom after Mom moves"

        format_warp("door", 7, 1, "House 2F", "House 1F", "Main entrance")
        -> "[WARP] door at (7, 1) in House 2F -> House 1F: Main entrance"
    """
    base = f"{PREFIX_WARP} {warp_type} at ({x}, {y}) in {location}"

    if destination:
        base = f"{base} -> {destination}"

    if extra_desc:
        return f"{base}: {extra_desc}"
    return base


def parse_spatial_entry(content: str) -> Optional[Dict]:
    """
    Parse a spatial knowledge entry to extract structured data

    Args:
        content: Knowledge content string

    Returns:
        Dict with type, description, coordinates, location, etc.
        None if not a spatial entry

    Example:
        Input: "[OBJECT] Television at (4, 1) in Brendan's House 2F"
        Output: {
            'type': 'object',
            'description': 'Television',
            'x': 4,
            'y': 1,
            'location': "Brendan's House 2F"
        }

        Input: "[NPC] Mom at (8, 2) in House 1F: Blocks stairs"
        Output: {
            'type': 'npc',
            'name': 'Mom',
            'is_trainer': False,
            'x': 8,
            'y': 2,
            'location': 'House 1F',
            'extra_desc': 'Blocks stairs'
        }

        Input: "[WARP] door at (7, 1) in House 2F -> House 1F: Main entrance"
        Output: {
            'type': 'warp',
            'warp_type': 'door',
            'x': 7,
            'y': 1,
            'location': 'House 2F',
            'destination': 'House 1F',
            'extra_desc': 'Main entrance'
        }
    """
    # Try OBJECT pattern
    object_match = re.match(
        r'\[OBJECT\]\s+(.+?)\s+at\s+\((\d+),\s*(\d+)\)\s+in\s+(.+?)(?::\s+(.+))?$',
        content
    )
    if object_match:
        result = {
            'type': 'object',
            'description': object_match.group(1),
            'x': int(object_match.group(2)),
            'y': int(object_match.group(3)),
            'location': object_match.group(4)
        }
        if object_match.group(5):
            result['extra_desc'] = object_match.group(5)
        return result

    # Try NPC pattern
    npc_match = re.match(
        r'\[NPC\]\s+(.+?)\s+at\s+\((\d+),\s*(\d+)\)\s+in\s+(.+?)(?::\s+(.+))?$',
        content
    )
    if npc_match:
        name = npc_match.group(1)
        is_trainer = '(trainer)' in name.lower()
        if is_trainer:
            name = name.replace('(trainer)', '').replace('(Trainer)', '').strip()

        result = {
            'type': 'npc',
            'name': name,
            'is_trainer': is_trainer,
            'x': int(npc_match.group(2)),
            'y': int(npc_match.group(3)),
            'location': npc_match.group(4)
        }
        if npc_match.group(5):
            result['extra_desc'] = npc_match.group(5)
        return result

    # Try WARP pattern (with optional destination and extra_desc)
    warp_match = re.match(
        r'\[WARP\]\s+(\w+)\s+at\s+\((\d+),\s*(\d+)\)\s+in\s+(.+?)(?:\s+->\s+(.+?))?(?::\s+(.+))?$',
        content
    )
    if warp_match:
        result = {
            'type': 'warp',
            'warp_type': warp_match.group(1),
            'x': int(warp_match.group(2)),
            'y': int(warp_match.group(3)),
            'location': warp_match.group(4)
        }

        # Handle destination (group 5)
        destination = warp_match.group(5)
        if destination:
            # Check if destination contains the extra_desc (if -> was not used)
            # This handles cases where there's no -> but there's a :
            result['destination'] = destination

        # Handle extra_desc (group 6)
        extra_desc = warp_match.group(6)
        if extra_desc:
            result['extra_desc'] = extra_desc

        return result

    # Not a spatial entry
    return None


def is_spatial_entry(content: str) -> bool:
    """
    Check if a knowledge entry is a spatial entry

    Args:
        content: Knowledge content string

    Returns:
        True if content starts with spatial prefix
    """
    return any(content.startswith(prefix) for prefix in [
        PREFIX_OBJECT,
        PREFIX_NPC,
        PREFIX_WARP
    ])


def get_spatial_type(content: str) -> Optional[str]:
    """
    Get the type of spatial entry

    Args:
        content: Knowledge content string

    Returns:
        'object', 'npc', 'warp', or None
    """
    if content.startswith(PREFIX_OBJECT):
        return 'object'
    elif content.startswith(PREFIX_NPC):
        return 'npc'
    elif content.startswith(PREFIX_WARP):
        return 'warp'
    return None
