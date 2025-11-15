#!/usr/bin/env python3
"""
Generate Static ASCII Maps from Pokemon Emerald Data

This script processes the pokeemerald decompilation data to generate
static ASCII representations of all map layouts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pokeemerald_parser import (
    parse_map_bin,
    parse_metatile_attributes,
    load_tileset_behaviors,
    get_tile_behavior
)
from pokemon_env.enums import MetatileBehavior
from utils.map_formatter import format_tile_to_symbol


def extract_tileset_name(tileset_label: str) -> str:
    """Extract tileset name from label like 'gTileset_General' -> 'general'"""
    if tileset_label.startswith("gTileset_"):
        return tileset_label[9:].lower()
    return tileset_label.lower()


def get_tileset_path(tileset_name: str, is_primary: bool) -> Path:
    """Get the path to a tileset's metatile_attributes.bin file"""
    base = Path("data/tilesets")
    category = "primary" if is_primary else "secondary"
    return base / category / tileset_name / "metatile_attributes.bin"


def convert_map_to_ascii(
    layout: Dict,
    behavior_cache: Dict[Tuple[str, str], Dict[int, int]]
) -> str:
    """
    Convert a single map layout to ASCII representation.

    Args:
        layout: Layout metadata from layouts.json
        behavior_cache: Cache of loaded tileset behaviors

    Returns:
        ASCII map as a string with newline-separated rows
    """
    # Parse map.bin
    map_path = layout['blockdata_filepath']
    width = layout['width']
    height = layout['height']

    try:
        tile_grid = parse_map_bin(map_path, width, height)
    except Exception as e:
        print(f"  Error parsing {map_path}: {e}", file=sys.stderr)
        return ""

    # Load tileset behaviors (with caching)
    primary_name = extract_tileset_name(layout['primary_tileset'])
    secondary_name = extract_tileset_name(layout['secondary_tileset'])
    cache_key = (primary_name, secondary_name)

    if cache_key not in behavior_cache:
        primary_path = get_tileset_path(primary_name, is_primary=True)
        secondary_path = get_tileset_path(secondary_name, is_primary=False)

        try:
            behaviors = load_tileset_behaviors(str(primary_path), str(secondary_path))
            behavior_cache[cache_key] = behaviors
        except Exception as e:
            print(f"  Error loading tilesets {primary_name}/{secondary_name}: {e}", file=sys.stderr)
            behavior_cache[cache_key] = {}

    behaviors = behavior_cache[cache_key]

    # Convert to ASCII using existing map_formatter logic
    ascii_rows = []
    for y, row in enumerate(tile_grid):
        ascii_row = []
        for x, (metatile_id, collision, elevation) in enumerate(row):
            # Look up behavior
            behavior = get_tile_behavior(metatile_id, behaviors)

            # Create tile tuple in the format expected by format_tile_to_symbol
            tile = (metatile_id, behavior, collision, elevation)

            # Convert to symbol
            symbol = format_tile_to_symbol(
                tile,
                x=x,
                y=y,
                location_name=layout['name']
            )

            ascii_row.append(symbol)

        ascii_rows.append(''.join(ascii_row))

    return '\n'.join(ascii_rows)


def load_map_npcs(maps_dir: str = "data/maps") -> Dict[str, List[Dict]]:
    """
    Load NPC positions and metadata from all map.json files.

    Args:
        maps_dir: Directory containing map subdirectories

    Returns:
        Dictionary mapping layout ID to list of NPC metadata dicts
        Each NPC dict contains: {x, y, sight_range, facing, trainer_type, movement_range}
    """
    layout_npcs = defaultdict(list)
    maps_path = Path(maps_dir)

    if not maps_path.exists():
        print(f"Warning: Maps directory not found: {maps_dir}", file=sys.stderr)
        return {}

    # Scan all subdirectories for map.json files
    map_json_files = list(maps_path.glob("*/map.json"))
    print(f"Scanning {len(map_json_files)} map.json files for NPCs...")

    for map_file in map_json_files:
        try:
            with open(map_file, 'r') as f:
                map_data = json.load(f)

            layout_id = map_data.get('layout')
            if not layout_id:
                continue

            # Extract NPC positions from object_events
            object_events = map_data.get('object_events', [])
            for obj in object_events:
                graphics_id = obj.get('graphics_id', '')

                # Skip item balls and other non-NPC objects
                if 'ITEM_BALL' in graphics_id:
                    continue

                x = obj.get('x')
                y = obj.get('y')

                if x is not None and y is not None:
                    # Parse facing direction from movement_type
                    movement_type = obj.get('movement_type', '')
                    facing = 'down'  # default
                    if 'FACE_DOWN' in movement_type or 'LOOK_DOWN' in movement_type:
                        facing = 'down'
                    elif 'FACE_UP' in movement_type or 'LOOK_UP' in movement_type:
                        facing = 'up'
                    elif 'FACE_LEFT' in movement_type or 'LOOK_LEFT' in movement_type:
                        facing = 'left'
                    elif 'FACE_RIGHT' in movement_type or 'LOOK_RIGHT' in movement_type:
                        facing = 'right'

                    # Get sight range (trainer_sight_or_berry_tree_id)
                    sight_str = obj.get('trainer_sight_or_berry_tree_id', '0')
                    try:
                        sight_range = int(sight_str)
                    except (ValueError, TypeError):
                        sight_range = 0

                    # Get trainer type
                    trainer_type = obj.get('trainer_type', 'TRAINER_TYPE_NONE')

                    # Get movement range
                    movement_range_x = obj.get('movement_range_x', 0)
                    movement_range_y = obj.get('movement_range_y', 0)

                    npc_data = {
                        'x': x,
                        'y': y,
                        'sight_range': sight_range,
                        'facing': facing,
                        'trainer_type': trainer_type,
                        'movement_range_x': movement_range_x,
                        'movement_range_y': movement_range_y
                    }
                    layout_npcs[layout_id].append(npc_data)

        except Exception as e:
            print(f"  Warning: Failed to process {map_file}: {e}", file=sys.stderr)
            continue

    # Print statistics
    total_npcs = sum(len(npcs) for npcs in layout_npcs.values())
    print(f"Loaded {total_npcs} NPCs across {len(layout_npcs)} layouts")

    return dict(layout_npcs)


def overlay_npcs(ascii_map: str, npc_data_list: List[Dict], layout_id: str = None) -> str:
    """
    Overlay NPC positions onto an ASCII map.
    For Rustboro Gym, also overlay NPC sight ranges.

    Args:
        ascii_map: ASCII map string with newline-separated rows
        npc_data_list: List of NPC metadata dicts with x, y, sight_range,
                       facing, trainer_type, movement_range
        layout_id: Layout ID to determine special handling

    Returns:
        ASCII map with NPCs marked as 'N' and sight ranges as '@' (Rustboro Gym only)
    """
    if not npc_data_list:
        return ascii_map

    # Convert to 2D array
    rows = ascii_map.split('\n')
    grid = [list(row) for row in rows]
    map_height = len(grid)
    map_width = len(grid[0]) if map_height > 0 else 0

    # Special handling for Rustboro Gym: mark sight ranges
    if layout_id == 'LAYOUT_RUSTBORO_CITY_GYM':
        # First pass: Mark sight ranges with '@'
        for npc in npc_data_list:
            x = npc['x']
            y = npc['y']
            sight_range = npc['sight_range']
            facing = npc['facing']
            trainer_type = npc['trainer_type']

            # Only mark sight range for trainers
            if trainer_type != 'TRAINER_TYPE_NONE' and sight_range > 0:
                # Calculate sight tiles
                sight_tiles = []
                for distance in range(1, sight_range + 1):
                    if facing == 'down':
                        sight_tiles.append((x, y + distance))
                    elif facing == 'up':
                        sight_tiles.append((x, y - distance))
                    elif facing == 'left':
                        sight_tiles.append((x - distance, y))
                    elif facing == 'right':
                        sight_tiles.append((x + distance, y))

                # Mark sight tiles with '@'
                for sx, sy in sight_tiles:
                    if 0 <= sy < map_height and 0 <= sx < map_width:
                        current = grid[sy][sx]
                        # Only mark walkable tiles
                        if current == '.':
                            grid[sy][sx] = '@'

    # Second pass: Mark actual NPC positions with 'N'
    for npc in npc_data_list:
        x = npc['x']
        y = npc['y']

        # Check bounds
        if 0 <= y < map_height and 0 <= x < map_width:
            grid[y][x] = 'N'

    # Convert back to string
    return '\n'.join(''.join(row) for row in grid)


def generate_all_ascii_maps(
    layouts_path: str = "data/layouts/layouts.json",
    npc_data: Dict[str, Set[Tuple[int, int]]] = None
) -> Dict:
    """
    Generate ASCII maps for all layouts.

    Args:
        layouts_path: Path to layouts.json file
        npc_data: Optional dictionary mapping layout ID to NPC positions

    Returns:
        Dictionary mapping layout ID to ASCII map data
    """
    # Load layouts metadata
    with open(layouts_path, 'r') as f:
        layouts_data = json.load(f)

    results = {}
    behavior_cache = {}
    npc_data = npc_data or {}

    total = len(layouts_data['layouts'])
    print(f"Processing {total} layouts...")

    for i, layout in enumerate(layouts_data['layouts'], 1):
        layout_id = layout['id']
        layout_name = layout['name']

        # Rename specific layouts for consistency
        if layout_id == 'LAYOUT_INSIDE_OF_TRUCK':
            layout_id = 'LAYOUT_MOVING_VAN'

        if i % 50 == 0:
            print(f"  Progress: {i}/{total} ({i*100//total}%)")

        # Convert to ASCII
        ascii_map = convert_map_to_ascii(layout, behavior_cache)

        if ascii_map:
            # Overlay NPCs if available
            original_id = layout['id']  # Use original ID for NPC lookup
            if original_id in npc_data:
                ascii_map = overlay_npcs(ascii_map, npc_data[original_id], layout_id)

            results[layout_id] = {
                "name": layout_name,
                "width": layout['width'],
                "height": layout['height'],
                "primary_tileset": layout['primary_tileset'],
                "secondary_tileset": layout['secondary_tileset'],
                "ascii_map": ascii_map
            }
        else:
            print(f"  Skipping {layout_name} (conversion failed)", file=sys.stderr)

    print(f"Successfully generated {len(results)}/{total} ASCII maps")
    print(f"Tileset combinations cached: {len(behavior_cache)}")

    return results


def main():
    """Main entry point"""
    print("Pokemon Emerald ASCII Map Generator")
    print("=" * 50)

    # Load existing LAYOUT_TITLE_SEQUENCE if it exists
    output_path = "maps_ascii.json"
    title_sequence_data = None

    if Path(output_path).exists():
        print(f"\nLoading existing {output_path} to preserve LAYOUT_TITLE_SEQUENCE...")
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                if 'LAYOUT_TITLE_SEQUENCE' in existing_data:
                    title_sequence_data = existing_data['LAYOUT_TITLE_SEQUENCE']
                    print(f"  Found LAYOUT_TITLE_SEQUENCE")
        except Exception as e:
            print(f"  Warning: Could not load existing file: {e}", file=sys.stderr)

    # Load NPC data
    print()
    npc_data = load_map_npcs()

    # Generate maps
    print()
    maps = generate_all_ascii_maps(npc_data=npc_data)

    # Add LAYOUT_TITLE_SEQUENCE if it was preserved
    if title_sequence_data:
        print(f"\nAdding LAYOUT_TITLE_SEQUENCE to output...")
        maps['LAYOUT_TITLE_SEQUENCE'] = title_sequence_data

    # Save to file
    print(f"\nSaving to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(maps, f, indent=2)

    # Print statistics
    total_size = Path(output_path).stat().st_size
    print(f"\nDone!")
    print(f"  Maps generated: {len(maps)}")
    if title_sequence_data:
        print(f"  (including LAYOUT_TITLE_SEQUENCE)")
    print(f"  Output file: {output_path}")
    print(f"  File size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

    # Show sample
    if maps:
        sample_id = list(maps.keys())[0]
        sample = maps[sample_id]
        print(f"\nSample map: {sample['name']} ({sample['width']}x{sample['height']})")
        print("=" * 50)
        lines = sample['ascii_map'].split('\n')
        for line in lines[:10]:  # Show first 10 lines
            print(line)
        if len(lines) > 10:
            print(f"... ({len(lines) - 10} more lines)")


if __name__ == "__main__":
    main()
