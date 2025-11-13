"""
Pokemon Emerald Binary Data Parser

This module provides utilities to parse binary data files from the
pokeemerald decompilation project, including map layouts and tileset attributes.
"""

import struct
from typing import List, Tuple, Dict
from pathlib import Path


def parse_map_bin(filepath: str, width: int, height: int) -> List[List[Tuple[int, int, int]]]:
    """
    Parse a map.bin file into a 2D grid of tile data.

    Args:
        filepath: Path to the map.bin file
        width: Width of the map in tiles
        height: Height of the map in tiles

    Returns:
        2D list of (metatile_id, collision, elevation) tuples
        Dimensions: [height][width]

    Format:
        Each tile is a 16-bit little-endian value:
        - Bits 0-9: Metatile ID (0-1023)
        - Bits 10-11: Collision (0-3)
        - Bits 12-15: Elevation (0-15)
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    expected_size = width * height * 2  # 2 bytes per tile
    if len(data) != expected_size:
        raise ValueError(
            f"Map data size mismatch: expected {expected_size} bytes "
            f"for {width}x{height} map, got {len(data)} bytes"
        )

    # Parse as 16-bit little-endian values
    num_tiles = width * height
    tiles_1d = struct.unpack(f'<{num_tiles}H', data)

    # Convert to 2D grid and extract components
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            idx = y * width + x
            value = tiles_1d[idx]

            # Extract packed components
            metatile_id = value & 0x03FF  # Bits 0-9
            collision = (value >> 10) & 0x03  # Bits 10-11
            elevation = (value >> 12) & 0x0F  # Bits 12-15

            row.append((metatile_id, collision, elevation))
        grid.append(row)

    return grid


def parse_metatile_attributes(filepath: str) -> Dict[int, int]:
    """
    Parse a metatile_attributes.bin file into a behavior lookup table.

    Args:
        filepath: Path to the metatile_attributes.bin file

    Returns:
        Dictionary mapping metatile_id (0-511) to behavior value (0-255)

    Format:
        Each metatile has 2 bytes (some older formats may use 1 byte):
        - Byte 0: Behavior value (MetatileBehavior enum)
        - Byte 1: Layer type (terrain type, unused for our purposes)
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # Determine format based on file size
    # Primary tilesets: typically 512 metatiles = 1024 bytes (2 bytes each)
    # Secondary tilesets: typically 128-144 metatiles = 256-288 bytes

    bytes_per_metatile = 2  # Modern pokeemerald uses 2 bytes
    num_metatiles = len(data) // bytes_per_metatile

    behaviors = {}
    for i in range(num_metatiles):
        offset = i * bytes_per_metatile
        behavior = data[offset]  # First byte is the behavior
        behaviors[i] = behavior

    return behaviors


def load_tileset_behaviors(primary_path: str, secondary_path: str) -> Dict[int, int]:
    """
    Load and merge primary and secondary tileset behaviors.

    Args:
        primary_path: Path to primary tileset's metatile_attributes.bin
        secondary_path: Path to secondary tileset's metatile_attributes.bin

    Returns:
        Combined behavior lookup table where:
        - IDs 0-511: Primary tileset behaviors
        - IDs 512-639: Secondary tileset behaviors (offset by 512)

    Note:
        In Pokemon Emerald, maps use both a primary and secondary tileset.
        Primary tileset provides common tiles (0-511), secondary provides
        map-specific tiles (512+).
    """
    primary_behaviors = parse_metatile_attributes(primary_path)
    secondary_behaviors = parse_metatile_attributes(secondary_path)

    # Combine with offset for secondary
    combined = {}

    # Add primary tileset (0-511)
    for tile_id, behavior in primary_behaviors.items():
        combined[tile_id] = behavior

    # Add secondary tileset (offset by 512)
    for tile_id, behavior in secondary_behaviors.items():
        combined[512 + tile_id] = behavior

    return combined


def get_tile_behavior(metatile_id: int, behavior_table: Dict[int, int]) -> int:
    """
    Look up the behavior for a given metatile ID.

    Args:
        metatile_id: The metatile ID (0-1023)
        behavior_table: Combined behavior lookup table from load_tileset_behaviors

    Returns:
        Behavior value (0-255), or 0 (NORMAL) if not found
    """
    return behavior_table.get(metatile_id, 0)


if __name__ == "__main__":
    # Test with a sample map
    import json

    # Load layouts.json
    with open('data/layouts/layouts.json', 'r') as f:
        layouts_data = json.load(f)

    # Test with first layout
    if layouts_data['layouts']:
        layout = layouts_data['layouts'][0]
        print(f"Testing with: {layout['name']}")
        print(f"  Size: {layout['width']}x{layout['height']}")
        print(f"  Primary tileset: {layout['primary_tileset']}")
        print(f"  Secondary tileset: {layout['secondary_tileset']}")

        # Parse map
        map_path = layout['blockdata_filepath']
        grid = parse_map_bin(map_path, layout['width'], layout['height'])
        print(f"  Parsed {len(grid)}x{len(grid[0])} grid")

        # Sample tile
        tile = grid[0][0]
        print(f"  Tile at (0,0): metatile_id={tile[0]}, collision={tile[1]}, elevation={tile[2]}")
