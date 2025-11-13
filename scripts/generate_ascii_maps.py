#!/usr/bin/env python3
"""
Generate Static ASCII Maps from Pokemon Emerald Data

This script processes the pokeemerald decompilation data to generate
static ASCII representations of all map layouts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def generate_all_ascii_maps(layouts_path: str = "data/layouts/layouts.json") -> Dict:
    """
    Generate ASCII maps for all layouts.

    Args:
        layouts_path: Path to layouts.json file

    Returns:
        Dictionary mapping layout ID to ASCII map data
    """
    # Load layouts metadata
    with open(layouts_path, 'r') as f:
        layouts_data = json.load(f)

    results = {}
    behavior_cache = {}

    total = len(layouts_data['layouts'])
    print(f"Processing {total} layouts...")

    for i, layout in enumerate(layouts_data['layouts'], 1):
        layout_id = layout['id']
        layout_name = layout['name']

        if i % 50 == 0:
            print(f"  Progress: {i}/{total} ({i*100//total}%)")

        # Convert to ASCII
        ascii_map = convert_map_to_ascii(layout, behavior_cache)

        if ascii_map:
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

    # Generate maps
    maps = generate_all_ascii_maps()

    # Save to file
    output_path = "maps_ascii.json"
    print(f"\nSaving to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(maps, f, indent=2)

    # Print statistics
    total_size = Path(output_path).stat().st_size
    print(f"\nDone!")
    print(f"  Maps generated: {len(maps)}")
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
