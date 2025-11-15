#!/usr/bin/env python3
"""
Battle UI Parser tool for CodeAgent

Parses battle UI and returns structured UI information with highlighted items.
This file is auto-loaded - no import needed in LLM code!
"""

import logging

logger = logging.getLogger(__name__)


def get_battle_ui(state) -> dict:
    """
    Parse and return the battle UI structure with highlighted items marked.

    Args:
        state: State object with VLM support

    Returns:
        dict: UI structure with the following format:
            {
                "ui_type": "main_menu" | "move_selection" | "none",
                "ui_grid": list of list (2x2 grid),
                "raw_text": str (description of UI)
            }

        For main menu (Fight/Pokemon/Bag/Run):
            ui_grid = [
                ["* Fight", "Pokemon"],  # * indicates highlighted
                ["Bag", "Run"]
            ]

        For move selection (4 moves with PP):
            ui_grid = [
                ["* Tackle PP 35/35", "Water Gun PP 25/25"],
                ["Growl PP 40/40", "Tail Whip PP 30/30"]
            ]

    Example:
        ui_info = get_battle_ui(state)
        print(ui_info["ui_grid"])
        # [["* Fight", "Pokemon"], ["Bag", "Run"]]
    """
    try:
        from utils.vlm_state import add_to_state_schema

        # Check if in battle
        add_to_state_schema(
            key="in_battle",
            vlm_prompt="Is the character currently in a Pokemon battle? Look for battle UI, HP bars, Pokemon sprites. Answer true only if battle screen is visible, false otherwise.",
            return_type=bool,
        )

        if not state["in_battle"]:
            logger.info("Not in battle - no UI to parse")
            return {
                "ui_type": "none",
                "ui_grid": [],
                "raw_text": "Not in battle"
            }

        # Check what type of UI is visible
        add_to_state_schema(
            key="battle_ui_type",
            vlm_prompt=(
                "What battle UI is currently visible? "
                "Answer with ONE of these exact strings:\n"
                "- 'main_menu': if Fight/Pokemon/Bag/Run menu is visible\n"
                "- 'move_selection': if move list with PP counts is visible\n"
                "- 'none': if no menu is visible (dialog/animation)"
            ),
            return_type=str,
        )

        ui_type = state["battle_ui_type"].lower()
        logger.info(f"Battle UI type detected: {ui_type}")

        if ui_type == "main_menu":
            return _parse_main_menu_ui(state)
        elif ui_type == "move_selection":
            return _parse_move_selection_ui(state)
        else:
            return {
                "ui_type": "none",
                "ui_grid": [],
                "raw_text": "No menu visible (dialog or animation)"
            }

    except Exception as e:
        logger.error(f"Battle UI parser error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ui_type": "error",
            "ui_grid": [],
            "raw_text": f"Error: {str(e)}"
        }


def _parse_main_menu_ui(state) -> dict:
    """
    Parse the main battle menu (Fight/Pokemon/Bag/Run).

    Returns:
        dict: Parsed UI structure
    """
    from utils.vlm_state import add_to_state_schema

    # Get the currently selected option
    add_to_state_schema(
        key="battle_menu_selection",
        vlm_prompt=(
            "Which option is currently highlighted/selected in the battle menu? "
            "Answer with ONE of: 'Fight', 'Pokemon', 'Bag', 'Run'. "
            "Look for cursor or highlighting."
        ),
        return_type=str,
    )

    selected = state["battle_menu_selection"]
    logger.info(f"Main menu selected: {selected}")

    # Build the UI grid (2x2 layout)
    options = ["Fight", "Pokemon", "Bag", "Run"]
    ui_grid = [
        [
            f"* {options[0]}" if selected == options[0] else options[0],
            f"* {options[1]}" if selected == options[1] else options[1]
        ],
        [
            f"* {options[2]}" if selected == options[2] else options[2],
            f"* {options[3]}" if selected == options[3] else options[3]
        ]
    ]

    return {
        "ui_type": "main_menu",
        "ui_grid": ui_grid,
        "raw_text": f"Battle menu: {selected} selected"
    }


def _parse_move_selection_ui(state) -> dict:
    """
    Parse the move selection screen (4 moves with PP).

    Returns:
        dict: Parsed UI structure
    """
    from utils.vlm_state import add_to_state_schema

    # Get all 4 moves with their PP
    add_to_state_schema(
        key="battle_moves_info",
        vlm_prompt=(
            "Extract all move information from the move selection screen. "
            "Return a JSON object with:\n"
            "- moves: array of 4 move objects, each with:\n"
            "  - name: move name (string)\n"
            "  - pp_current: current PP (number)\n"
            "  - pp_max: max PP (number)\n"
            "  - position: position in grid 1-4 (number)\n"
            "- selected_position: which position is highlighted (number 1-4)\n"
            "Example: {\n"
            '  "moves": [\n'
            '    {"name": "Tackle", "pp_current": 35, "pp_max": 35, "position": 1},\n'
            '    {"name": "Water Gun", "pp_current": 25, "pp_max": 25, "position": 2},\n'
            '    {"name": "Growl", "pp_current": 40, "pp_max": 40, "position": 3},\n'
            '    {"name": "Tail Whip", "pp_current": 30, "pp_max": 30, "position": 4}\n'
            '  ],\n'
            '  "selected_position": 1\n'
            "}"
        ),
        return_type=dict,
    )

    moves_info = state["battle_moves_info"]
    logger.info(f"Move selection info: {moves_info}")

    # Build move strings with PP
    move_strings = {}
    for move in moves_info["moves"]:
        pp_str = f"PP {move['pp_current']}/{move['pp_max']}"
        move_str = f"{move['name']} {pp_str}"

        # Add * if selected
        if move["position"] == moves_info["selected_position"]:
            move_str = f"* {move_str}"

        move_strings[move["position"]] = move_str

    # Build 2x2 grid (positions: 1 2 / 3 4)
    ui_grid = [
        [move_strings.get(1, "---"), move_strings.get(2, "---")],
        [move_strings.get(3, "---"), move_strings.get(4, "---")]
    ]

    selected_move = next(
        (m for m in moves_info["moves"] if m["position"] == moves_info["selected_position"]),
        None
    )
    selected_name = selected_move["name"] if selected_move else "Unknown"

    return {
        "ui_type": "move_selection",
        "ui_grid": ui_grid,
        "raw_text": f"Move selection: {selected_name} selected"
    }