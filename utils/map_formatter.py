#!/usr/bin/env python3
"""
Centralized Map Formatting Utility

Single source of truth for all map formatting across the codebase.
"""

from pokemon_env.enums import MetatileBehavior


def format_tile_to_symbol(tile, x=None, y=None, location_name=None, player_pos=None, stairs_pos=None):
    """
    Convert a single tile to its display symbol.
    
    Args:
        tile: Tuple of (tile_id, behavior, collision, elevation)
        x: Optional x coordinate for context-specific symbols
        y: Optional y coordinate for context-specific symbols
        location_name: Optional location name for context-specific symbols
        player_pos: Optional player position tuple (px, py) for relative positioning
        stairs_pos: Optional stairs position tuple (sx, sy) for relative positioning
        
    Returns:
        str: Single character symbol representing the tile
    """
    if len(tile) >= 4:
        tile_id, behavior, collision, _ = tile  # elevation not used
    elif len(tile) >= 2:
        tile_id, behavior = tile[:2]
        collision = 0
    else:
        tile_id = tile[0] if tile else 0
        behavior = MetatileBehavior.NORMAL
        collision = 0
    
    # Convert behavior to symbol using unified logic
    if hasattr(behavior, 'name'):
        behavior_name = behavior.name
    elif isinstance(behavior, int):
        try:
            behavior_enum = MetatileBehavior(behavior)
            behavior_name = behavior_enum.name
        except ValueError:
            behavior_name = "UNKNOWN"
    else:
        behavior_name = "UNKNOWN"
    
    # Special handling for Brendan's House 2F wall clock
    # The clock is a wall tile with no special behavior, just tile ID 1023
    # Position it relative to the stairs dynamically
    if location_name and "BRENDAN" in location_name.upper() and "2F" in location_name.upper():
        if x is not None and y is not None and stairs_pos:
            sx, sy = stairs_pos
            # Clock is 2 tiles west of stairs on the same row
            if (x, y) == (sx - 2, sy):
                return "K"  # K for Klock (C is taken by Computer)
    
    # Map to symbol - SINGLE SOURCE OF TRUTH
    # tile_id 1023 (0x3FF) is usually invalid/out-of-bounds
    if tile_id == 1023:
        return "#"  # Always show as blocked/wall
    elif behavior_name == "NORMAL":
        return "." if collision == 0 else "#"
    # Fix for reversed door/stairs mapping in Brendan's house
    # NON_ANIMATED_DOOR (96) appears at top and should show as 'S' 
    # SOUTH_ARROW_WARP (101) appears at bottom and should show as 'D'
    elif behavior == 96 or "NON_ANIMATED_DOOR" in behavior_name:
        return "S"  # This is actually stairs going upstairs
    elif behavior == 101 or "SOUTH_ARROW_WARP" in behavior_name:
        return "D"  # This is actually the exit door
    elif "DOOR" in behavior_name:
        return "D"  # Other doors remain as doors
    elif "STAIRS" in behavior_name or "WARP" in behavior_name:
        return "S"  # Other stairs/warps remain as stairs
    elif "WATER" in behavior_name:
        return "W"
    elif "TALL_GRASS" in behavior_name:
        return "~"
    elif "COMPUTER" in behavior_name or "PC" in behavior_name:
        return "c"  # PC/Computer
    elif "TELEVISION" in behavior_name or "TV" in behavior_name:
        return "T"  # Television
    elif "BOOKSHELF" in behavior_name or "SHELF" in behavior_name:
        return "B"  # Bookshelf
    elif "SIGN" in behavior_name or "SIGNPOST" in behavior_name:
        return "?"  # Sign/Information
    elif "FLOWER" in behavior_name or "PLANT" in behavior_name:
        return "F"  # Flowers/Plants
    elif "COUNTER" in behavior_name or "DESK" in behavior_name:
        return "C"  # Counter/Desk
    elif "BED" in behavior_name or "SLEEP" in behavior_name:
        return "="  # Bed
    elif "TABLE" in behavior_name or "CHAIR" in behavior_name:
        return "t"  # Table/Chair
    elif "CLOCK" in behavior_name:
        return "O"  # Clock (O for clock face)
    elif "PICTURE" in behavior_name or "PAINTING" in behavior_name:
        return "^"  # Picture/Painting on wall
    elif "TRASH" in behavior_name or "BIN" in behavior_name:
        return "U"  # Trash can/bin
    elif "POT" in behavior_name or "VASE" in behavior_name:
        return "V"  # Pot/Vase
    elif "MACHINE" in behavior_name or "DEVICE" in behavior_name:
        return "M"  # Machine/Device
    elif "JUMP" in behavior_name:
        if "SOUTH" in behavior_name:
            return "↓"
        elif "EAST" in behavior_name:
            return "→"
        elif "WEST" in behavior_name:
            return "←"
        elif "NORTH" in behavior_name:
            return "↑"
        elif "NORTHEAST" in behavior_name:
            return "↗"
        elif "NORTHWEST" in behavior_name:
            return "↖"
        elif "SOUTHEAST" in behavior_name:
            return "↘"
        elif "SOUTHWEST" in behavior_name:
            return "↙"
        else:
            return "J"
    elif "IMPASSABLE" in behavior_name or "SEALED" in behavior_name:
        return "#"  # Blocked
    elif "INDOOR" in behavior_name:
        return "."  # Indoor tiles are walkable
    elif "DECORATION" in behavior_name or "HOLDS" in behavior_name:
        return "."  # Decorations are walkable
    else:
        # For unknown behavior, mark as blocked for safety
        return "#"


def format_map_grid(raw_tiles, player_facing="South", npcs=None, player_coords=None, trim_padding=True, location_name=None):
    """
    Format raw tile data into a traversability grid with NPCs.
    
    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction for center marker
        npcs: List of NPC/object events with positions
        player_coords: Player coordinates for relative positioning
        trim_padding: If True, remove padding rows/columns that are all walls
        location_name: Optional location name for context-specific symbols
        
    Returns:
        list: 2D list of symbol strings
    """
    if not raw_tiles or len(raw_tiles) == 0:
        return []
    
    # First pass: find the stairs position if in Brendan's house 2F
    stairs_pos = None
    if location_name and "BRENDAN" in location_name.upper() and "2F" in location_name.upper():
        for y, row in enumerate(raw_tiles):
            for x, tile in enumerate(row):
                if len(tile) >= 2:
                    _, behavior = tile[:2]
                    # Stairs have behavior 96 (NON_ANIMATED_DOOR which we mapped to 'S')
                    if behavior == 96:
                        stairs_pos = (x, y)
                        break
            if stairs_pos:
                break
    
    grid = []
    center_y = len(raw_tiles) // 2
    center_x = len(raw_tiles[0]) // 2
    
    # Player is always at the center of the 15x15 grid view
    # but we need the actual player coordinates for NPC positioning
    player_map_x = center_x  # Grid position (always 7,7 in 15x15)
    player_map_y = center_y
    
    # Always use P for player instead of direction arrows
    player_symbol = "P"
    
    # Create NPC position lookup (convert to relative grid coordinates)
    npc_positions = {}
    if npcs and player_coords:
        try:
            # Handle both tuple and dict formats for player_coords
            if isinstance(player_coords, dict):
                player_abs_x = player_coords.get('x', 0)
                player_abs_y = player_coords.get('y', 0)
            else:
                player_abs_x, player_abs_y = player_coords
            
            # Ensure coordinates are integers
            player_abs_x = int(player_abs_x) if player_abs_x is not None else 0
            player_abs_y = int(player_abs_y) if player_abs_y is not None else 0
            
            for npc in npcs:
                # NPCs have absolute world coordinates, convert to relative grid position
                npc_abs_x = npc.get('current_x', 0)
                npc_abs_y = npc.get('current_y', 0)
                
                # Ensure NPC coordinates are integers
                npc_abs_x = int(npc_abs_x) if npc_abs_x is not None else 0
                npc_abs_y = int(npc_abs_y) if npc_abs_y is not None else 0
                
                # Calculate offset from player in absolute coordinates
                offset_x = npc_abs_x - player_abs_x
                offset_y = npc_abs_y - player_abs_y
                
                # Convert offset to grid position (player is at center)
                grid_x = center_x + offset_x
                grid_y = center_y + offset_y
                
                # Check if NPC is within our grid view
                if 0 <= grid_x < len(raw_tiles[0]) and 0 <= grid_y < len(raw_tiles):
                    npc_positions[(grid_y, grid_x)] = npc
                    
        except (ValueError, TypeError) as e:
            # If coordinate conversion fails, skip NPC positioning
            print(f"Warning: Failed to convert coordinates for NPC positioning: {e}")
            print(f"  player_coords: {player_coords}")
            if npcs:
                print(f"  npc coords: {[(npc.get('current_x'), npc.get('current_y')) for npc in npcs]}")
            npc_positions = {}
    
    for y, row in enumerate(raw_tiles):
        grid_row = []
        for x, tile in enumerate(row):
            if y == center_y and x == center_x:
                # Player position
                grid_row.append(player_symbol)
            elif (y, x) in npc_positions:
                # NPC position - use NPC symbol
                npc = npc_positions[(y, x)]
                # Use different symbols for different NPC types
                if npc.get('trainer_type', 0) > 0:
                    grid_row.append("@")  # Trainer
                else:
                    grid_row.append("N")  # Regular NPC
            else:
                # Regular tile - pass coordinates and context for special handling
                symbol = format_tile_to_symbol(tile, x=x, y=y, location_name=location_name, 
                                               player_pos=(center_x, center_y), stairs_pos=stairs_pos)
                grid_row.append(symbol)
        grid.append(grid_row)
    
    # Trim padding if requested - but keep room boundaries!
    if trim_padding and len(grid) > 0:
        # First pass: Remove obvious padding (rows/columns that are ALL walls with no variation)
        # But we need to be careful to keep actual room walls
        
        # Check if we have any content in the middle
        has_walkable = False
        for row in grid:
            if any(cell in ['.', 'P', 'D', 'N', 'T', 'S'] for cell in row):
                has_walkable = True
                break
        
        if has_walkable:
            # Only trim extra padding beyond the first wall layer
            # Count consecutive wall rows from top
            top_wall_rows = 0
            for row in grid:
                if all(cell == '#' for cell in row):
                    top_wall_rows += 1
                else:
                    break
            
            # Remove extra top padding but keep one wall row
            while top_wall_rows > 1 and len(grid) > 1:
                grid.pop(0)
                top_wall_rows -= 1
            
            # Count consecutive wall rows from bottom
            bottom_wall_rows = 0
            for row in reversed(grid):
                if all(cell == '#' for cell in row):
                    bottom_wall_rows += 1
                else:
                    break
            
            # Remove extra bottom padding but keep one wall row
            while bottom_wall_rows > 1 and len(grid) > 1:
                grid.pop()
                bottom_wall_rows -= 1
            
            # Similar for left/right but be more conservative
            # Don't trim sides if we have doors or other features in the walls
    
    return grid


def format_stitched_map_simple(area, player_world_x, player_world_y):
    """
    Format stitched map from MapArea - using world coordinates.

    Shows explored_bounds rectangle with world coordinates.
    Grid (0, 0) represents world coordinate (min_x - origin_x, min_y - origin_y).

    Args:
        area: MapArea object with map_data and explored_bounds
        player_world_x: Player's world X coordinate
        player_world_y: Player's world Y coordinate

    Returns:
        list of str: Each row as a string (for list format)
    """
    if not area or not area.map_data:
        return []

    # Get explored bounds (absolute coordinates)
    bounds = area.explored_bounds if hasattr(area, 'explored_bounds') else {}
    min_x = bounds.get('min_x', 0)
    min_y = bounds.get('min_y', 0)
    max_x = bounds.get('max_x', 20)
    max_y = bounds.get('max_y', 20)

    # Get origin offset for world coordinate calculation
    offset = area.origin_offset if hasattr(area, 'origin_offset') else {'x': 0, 'y': 0}
    offset_x = offset.get('x', 0)
    offset_y = offset.get('y', 0)

    # Calculate world coordinate origin
    # Grid (0, 0) = World coordinate (min_x - offset_x, min_y - offset_y)
    world_origin_x = min_x - offset_x
    world_origin_y = min_y - offset_y

    # Calculate grid dimensions
    grid_width = max_x - min_x
    grid_height = max_y - min_y

    lines = []

    # Header: world X coordinates
    world_x_coords = [world_origin_x + x for x in range(grid_width + 1)]
    x_coords = " ".join(f"{x:2}" for x in world_x_coords)
    lines.append(f"    {x_coords}")

    # Each row
    for grid_y in range(grid_height + 1):
        # Calculate world Y coordinate for this row
        world_y = world_origin_y + grid_y

        row_symbols = []
        for grid_x in range(grid_width + 1):
            # Calculate world X coordinate
            world_x = world_origin_x + grid_x

            # Convert to absolute coordinates for map_data access
            abs_x = grid_x + min_x
            abs_y = grid_y + min_y

            # Check if this is player position (compare world coordinates)
            if world_y == player_world_y and world_x == player_world_x:
                row_symbols.append(" P")
            # Direct access to map_data using absolute coordinates
            elif (abs_y < len(area.map_data) and abs_x < len(area.map_data[0]) and
                  area.map_data[abs_y] and area.map_data[abs_y][abs_x]):
                tile = area.map_data[abs_y][abs_x]
                symbol = format_tile_to_symbol(tile)
                row_symbols.append(f" {symbol}")
            else:
                # Unexplored or out of bounds
                row_symbols.append(" ?")

        # Row with world Y coordinate
        lines.append(f"{world_y:3} {''.join(row_symbols)}")

    return lines


def format_map_grid_with_coords(raw_tiles, player_grid_x, player_grid_y, player_world_x, player_world_y,
                                 player_facing="South", npcs=None, location_name=None, trim_padding=True):
    """
    Format raw tile data into a traversability grid with explicit player grid coordinates.
    Used for stitched maps where player is not at the center.

    Args:
        raw_tiles: 2D list of tile tuples (full stitched map)
        player_grid_x: Player's X position in the grid (0-based index)
        player_grid_y: Player's Y position in the grid (0-based index)
        player_world_x: Player's actual game world X coordinate (for display)
        player_world_y: Player's actual game world Y coordinate (for display)
        player_facing: Player facing direction
        npcs: List of NPC/object events with positions
        location_name: Optional location name
        trim_padding: If True, remove padding rows/columns

    Returns:
        str: Formatted grid as string
    """
    if not raw_tiles or len(raw_tiles) == 0:
        return ""

    height = len(raw_tiles)
    width = len(raw_tiles[0]) if height > 0 else 0

    # Validate player position
    if player_grid_y < 0 or player_grid_y >= height or player_grid_x < 0 or player_grid_x >= width:
        # Player is outside the grid, fallback to format_map_for_llm
        return format_map_for_llm(raw_tiles, player_facing, npcs, (player_world_x, player_world_y), location_name)

    grid = []
    player_symbol = "P"

    # Process each row
    for y, row in enumerate(raw_tiles):
        grid_row = []
        for x, tile in enumerate(row):
            if y == player_grid_y and x == player_grid_x:
                # Player position
                grid_row.append(player_symbol)
            else:
                # Regular tile
                symbol = format_tile_to_symbol(tile)
                grid_row.append(symbol)
        grid.append(grid_row)

    # Trim padding if requested
    if trim_padding:
        grid = _trim_grid_padding(grid)

    # Convert grid to string with coordinate labels
    lines = []

    # Calculate coordinate range to display (world coordinates)
    # We need to map grid coordinates to world coordinates
    # player_grid_x/y corresponds to player_world_x/y
    # So: world_x = (grid_x - player_grid_x) + player_world_x

    # Find grid bounds after trimming
    if not grid or not grid[0]:
        return ""

    trimmed_height = len(grid)
    trimmed_width = len(grid[0])

    # Calculate offset due to trimming (count removed top rows and left columns)
    trim_offset_y = 0
    trim_offset_x = 0

    # Find how many rows were trimmed from top
    for y in range(height):
        if all(cell == '#' for cell in [format_tile_to_symbol(raw_tiles[y][x]) for x in range(width)]):
            trim_offset_y += 1
        else:
            break

    # Find how many columns were trimmed from left
    for x in range(width):
        if all(format_tile_to_symbol(raw_tiles[y][x]) == '#' for y in range(height)):
            trim_offset_x += 1
        else:
            break

    # Adjust player grid position for trimming
    adjusted_player_grid_x = player_grid_x - trim_offset_x
    adjusted_player_grid_y = player_grid_y - trim_offset_y

    # Calculate world coordinate for top-left of trimmed grid
    world_start_x = player_world_x - adjusted_player_grid_x
    world_start_y = player_world_y - adjusted_player_grid_y

    # Add coordinate header (X axis)
    x_coords = [str(world_start_x + i) for i in range(trimmed_width)]
    # Pad numbers to align properly
    max_coord_len = max(len(c) for c in x_coords)
    lines.append("  " + " ".join(c.rjust(max_coord_len) for c in x_coords))

    # Add each row with Y coordinate
    for i, row in enumerate(grid):
        y_coord = world_start_y + i
        row_str = " ".join(cell.rjust(max_coord_len) for cell in row)
        lines.append(f"{y_coord} {row_str}")

    return "\n".join(lines)


def format_map_for_display(raw_tiles, player_facing="South", title="Map", npcs=None, player_coords=None):
    """
    Format raw tiles into a complete display string with headers and legend.
    
    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction
        title: Title for the map display
        npcs: List of NPC/object events with positions
        player_coords: Dict with player absolute coordinates {'x': x, 'y': y}
        
    Returns:
        str: Formatted map display
    """
    if not raw_tiles:
        return f"{title}: No map data available"
    
    # Convert player_coords to tuple if it's a dict
    if player_coords and isinstance(player_coords, dict):
        player_coords_tuple = (player_coords['x'], player_coords['y'])
    else:
        player_coords_tuple = player_coords
    
    grid = format_map_grid(raw_tiles, player_facing, npcs, player_coords_tuple)
    
    lines = [f"{title} ({len(grid)}x{len(grid[0])}):", ""]
    
    # Add column headers
    header = "      "
    for i in range(len(grid[0])):
        header += f"{i:2} "
    lines.append(header)
    lines.append("     " + "--" * len(grid[0]))
    
    # Add grid with row numbers
    for y, row in enumerate(grid):
        row_str = f"  {y:2}: " + " ".join(f"{cell:2}" for cell in row)
        lines.append(row_str)
    
    # Add dynamic legend based on symbols that appear
    lines.append("")
    lines.append(generate_dynamic_legend(grid))
    
    return "\n".join(lines)


def get_symbol_legend():
    """
    Get the complete symbol legend for map displays.
    
    Returns:
        dict: Symbol -> description mapping
    """
    return {
        "P": "Player",
        ".": "Walkable path",
        "#": "Wall/Blocked/Unknown",
        "D": "Door",
        "S": "Stairs/Warp",
        "W": "Water",
        "~": "Tall grass",
        "c": "PC/Computer",
        "T": "Television",
        "B": "Bookshelf", 
        "?": "Unexplored area",
        "F": "Flowers/Plants",
        "C": "Counter/Desk",
        "=": "Bed",
        "t": "Table/Chair",
        "K": "Clock (Wall)",
        "O": "Clock",
        "^": "Picture/Painting",
        "U": "Trash can",
        "V": "Pot/Vase",
        "M": "Machine/Device",
        "J": "Jump ledge",
        "↓": "Jump South",
        "↑": "Jump North",
        "←": "Jump West",
        "→": "Jump East",
        "↗": "Jump Northeast",
        "↖": "Jump Northwest", 
        "↘": "Jump Southeast",
        "↙": "Jump Southwest",
        "N": "NPC",
        "@": "Trainer"
    }


def generate_dynamic_legend(grid):
    """
    Generate a legend based on symbols that actually appear in the grid.
    
    Args:
        grid: 2D list of symbol strings
        
    Returns:
        str: Formatted legend string
    """
    if not grid:
        return ""
    
    symbol_legend = get_symbol_legend()
    symbols_used = set()
    
    # Collect all unique symbols in the grid
    for row in grid:
        for symbol in row:
            symbols_used.add(symbol)
    
    # Build legend for used symbols
    legend_lines = ["Legend:"]
    
    # Group symbols by category for better organization
    player_symbols = ["P"]
    terrain_symbols = [".", "#", "W", "~", "?"] 
    structure_symbols = ["D", "S"]
    jump_symbols = ["J", "↓", "↑", "←", "→", "↗", "↖", "↘", "↙"]
    furniture_symbols = ["PC", "T", "B", "F", "C", "=", "t", "K", "O", "^", "U", "V", "M"]
    npc_symbols = ["N", "@"]
    
    categories = [
        ("Movement", player_symbols),
        ("Terrain", terrain_symbols),
        ("Structures", structure_symbols), 
        ("Jump ledges", jump_symbols),
        ("Furniture", furniture_symbols),
        ("NPCs", npc_symbols)
    ]
    
    for category_name, symbol_list in categories:
        category_items = []
        for symbol in symbol_list:
            if symbol in symbols_used and symbol in symbol_legend:
                category_items.append(f"{symbol}={symbol_legend[symbol]}")
        
        if category_items:
            legend_lines.append(f"  {category_name}: {', '.join(category_items)}")
    
    return "\n".join(legend_lines)


def format_map_for_llm(raw_tiles, player_facing="South", npcs=None, player_coords=None, location_name=None):
    """
    Format raw tiles into LLM-friendly grid format (no headers/legends).

    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Direction player is facing
        npcs: List of NPC/object events
        player_coords: Player position for relative positioning
        location_name: Location name for context-specific symbols
        player_facing: Player facing direction
        npcs: List of NPC/object events with positions
        player_coords: Tuple of (player_x, player_y) in absolute world coordinates

    Returns:
        str: Grid format suitable for LLM
    """
    if not raw_tiles:
        return "No map data available"

    grid = format_map_grid(raw_tiles, player_facing, npcs, player_coords, location_name=location_name)

    # Simple grid format for LLM
    lines = []
    for row in grid:
        lines.append(" ".join(row))

    return "\n".join(lines)


def format_map_for_llm_json(map_stitcher, location_name: str, player_coords=None, npcs=None, connections=None):
    """
    Format map as structured JSON data for LLM consumption.

    This provides explicit tile information including coordinates, type, and walkability
    instead of relying on the LLM to interpret ASCII symbols.

    Args:
        map_stitcher: MapStitcher instance with map data
        location_name: Name of the location
        player_coords: Tuple of (player_x, player_y)
        npcs: List of NPC/object events
        connections: List of location connections

    Returns:
        str: Formatted text representation of map JSON data
    """
    if not map_stitcher:
        return "No map data available"

    # Generate JSON map data
    map_json = map_stitcher.generate_location_map_json(
        location_name=location_name,
        player_pos=player_coords,
        npcs=npcs,
        connections=connections
    )

    # Format as readable text
    return map_stitcher.format_map_json_as_text(map_json)