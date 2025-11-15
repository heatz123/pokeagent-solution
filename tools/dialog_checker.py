#!/usr/bin/env python3
"""
Dialog Checker tool for CodeAgent

Checks if a dialog/text box is currently displayed using image processing.
This file is auto-loaded - no import needed in LLM code!
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def is_dialog_open(state: dict) -> bool:
    """
    Check if a dialog or text box is currently displayed using VLM with 3-point fallback.

    This function uses VLM to analyze the screenshot and detect dialogs, with a fast
    3-point pixel check as a reliable fallback.

    Args:
        state: State dict with VLM support and 'screenshot' key

    Returns:
        bool: True if dialog is open, False otherwise

    Example:
        # Check if dialog is open
        if is_dialog_open(state):
            # Handle dialog
            return 'a'
        else:
            # Continue normal movement
            return find_path_action(state, goal_x=10, goal_y=5)

        # Alternative: Skip dialog first, then move
        if is_dialog_open(state):
            return 'a'
        return find_path_action(state, goal_x=10, goal_y=5)
    """
    try:
        # Fast path: 3-point exact pixel check (most reliable for Pokemon Emerald)
        # Try to get screenshot from State object's internal field first (_screenshot)
        # This is needed because info["frame"] might not be set on first step
        frame = None
        if hasattr(state, '_screenshot'):
            frame = state._screenshot
        if frame is None:
            frame = state.get("screenshot")

        # Convert PIL Image to numpy array if needed
        if frame is not None:
            if hasattr(frame, 'mode'):  # PIL Image
                frame = np.array(frame)

        if frame is not None and _has_dialog_border(frame):
            logger.info("Dialog detected by 3-point check")
            return True

        # VLM path: Use vision model for detection
        from utils.vlm_state import add_to_state_schema

        # Register VLM query for dialog detection
        add_to_state_schema(
            key="has_dialog_box",
            vlm_prompt="Is there a dialog box, text message, or NPC speech displayed on screen? Look for text boxes, speech bubbles, or any UI overlay with text. Answer true if any dialog/text is visible, false if only the game world is shown.",
            return_type=bool
        )

        has_dialog = state["has_dialog_box"]

        if has_dialog:
            logger.info("Dialog detected by VLM")
        else:
            logger.debug("No dialog detected by VLM")

        return has_dialog

    except Exception as e:
        logger.error(f"Dialog checker error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _detect_green_triangle(frame: np.ndarray) -> bool:
    """Detect the small green advance-triangle indicator in dialog boxes."""
    try:
        h, w = frame.shape[:2]
        # Check bottom-right and bottom-left corners for green triangle
        for corner_region in [
            frame[int(h*0.85):, int(w*0.85):, :],  # bottom-right
            frame[int(h*0.85):, :int(w*0.15), :]   # bottom-left
        ]:
            if corner_region.size == 0:
                continue

            g = corner_region[:, :, 1].astype(int)
            r = corner_region[:, :, 0].astype(int)
            b = corner_region[:, :, 2].astype(int)

            green_mask = (g >= 120) & (g > (r + 30)) & (g > (b + 30))
            green_fraction = float(green_mask.sum()) / float(green_mask.size)

            if green_fraction >= 0.05:
                return True

        return False
    except Exception:
        return False


def _has_dialog_border(frame: np.ndarray) -> bool:
    """Detect dialog box by checking exact pixels at left corner.

    Checks 3 specific points that indicate a Pokemon dialog box:
    1. (10, 117): RGB(0, 255, 156) - Outer cyan/teal border
    2. (11, 117): RGB(231, 239, 231) - Inner gray border
    3. (10, 120): RGB(255, 255, 255) - White interior

    Returns True if all 3 points match the expected colors.
    """
    try:
        h, w = frame.shape[:2]

        # Check if frame is large enough for these coordinates
        if h < 121 or w < 12:
            return False

        # Check the 3 key points
        # Point 1: Outer border (cyan/teal)
        r1, g1, b1 = int(frame[117, 10, 0]), int(frame[117, 10, 1]), int(frame[117, 10, 2])
        if (r1, g1, b1) != (0, 255, 156):
            return False

        # Point 2: Inner border (light gray)
        r2, g2, b2 = int(frame[117, 11, 0]), int(frame[117, 11, 1]), int(frame[117, 11, 2])
        if (r2, g2, b2) != (231, 239, 231):
            return False

        # Point 3: White interior
        r3, g3, b3 = int(frame[120, 10, 0]), int(frame[120, 10, 1]), int(frame[120, 10, 2])
        if (r3, g3, b3) != (255, 255, 255):
            return False

        return True

    except Exception:
        return False


def _detect_dialogue_box(
    frame: np.ndarray,
    white_tolerance: int = 12,
) -> bool:
    """Detect whether a dialogue/text box is visible near the bottom of the frame.

    Detection strategy:
    1. Fast path: 3-point exact pixel check (most accurate for Pokemon Emerald)
    2. Fallback: Heuristic checks (white interior, green border, text detection)

    Improvements to reduce false positives on bright interiors:
    - Narrow the bottom-region crop to focus on the dialog area.
    - Increase the white luminance threshold so bright floor/panels aren't mis-classified.
    - Require stronger per-row dark/text concentration before accepting small white fractions.
    - Keep a green-triangle fast-accept path but require a higher-quality green match.
    """
    try:
        if frame is None or not hasattr(frame, "shape"):
            return False

        # Fast path 1: 3-point exact border check (most reliable for Pokemon Emerald)
        if _has_dialog_border(frame):
            return True

        # Fast path 2: detect the small green advance-triangle (either side)
        try:
            if _detect_green_triangle(frame):
                return True
        except Exception:
            pass

        h, w, c = frame.shape
        if h < 20 or w < 20:
            return False

        # Bottom region selection: a narrower slice focused on the dialogue area
        # Use ~16-20% of height anchored at the bottom (dialog typically occupies lowest rows)
        region_height = max(8, int(h * 0.18))
        top = max(0, h - region_height - 4)
        bottom = h - 4
        left = max(6, int(w * 0.04))
        right = min(w - 6, int(w * 0.96))

        region = frame[top:bottom, left:right, :]
        if region.size == 0:
            return False

        # Convert to luminance (Rec. 709 approx)
        r = region[:, :, 0].astype(np.float32)
        g = region[:, :, 1].astype(np.float32)
        b = region[:, :, 2].astype(np.float32)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Raise the white threshold to avoid bright floors/windows being treated as dialog interior
        white_thresh = 200.0
        white_mask = lum >= white_thresh
        white_fraction = float(white_mask.sum()) / float(white_mask.size)

        # Stricter dark/text pixel threshold (text is relatively dark compared to dialog interior)
        dark_mask = lum <= 95.0
        dark_fraction = float(dark_mask.sum()) / float(dark_mask.size)

        # Per-row dark concentration (detect text lines)
        if dark_mask.shape[0] > 0 and dark_mask.shape[1] > 0:
            row_dark_counts = dark_mask.sum(axis=1).astype(float)
            row_dark_fractions = row_dark_counts / float(dark_mask.shape[1])
            max_row_dark_fraction = float(np.max(row_dark_fractions))
            # Also compute how many rows show concentrated dark pixels (text often spans several rows)
            rows_with_dark = float((row_dark_fractions >= 0.06).sum())
        else:
            max_row_dark_fraction = 0.0
            rows_with_dark = 0.0

        # Compute uniformity (stddev) of the white area - dialog interior is usually smooth
        white_pixels = lum[white_mask]
        if white_pixels.size > 0:
            white_std = float(np.std(white_pixels))
        else:
            white_std = 255.0

        # Green-ish border detection: inspect the top rows of the region for a green border
        border_rows = region[0:3, :, :]
        if border_rows.size == 0:
            green_fraction = 0.0
        else:
            br = border_rows[:, :, 0].astype(int)
            bg = border_rows[:, :, 1].astype(int)
            bb = border_rows[:, :, 2].astype(int)
            green_mask = (bg >= 100) & (bg > (br + 18)) & (bg > (bb + 18))
            green_fraction = float(green_mask.sum()) / float(green_mask.size)

        # Heuristic decision rules (more conservative than before):
        # Rule A: Clear dialog signature: sufficiently white interior AND clear text-line evidence
        if (
            white_fraction >= 0.28
            and dark_fraction >= 0.015
            and max_row_dark_fraction >= 0.06
            and rows_with_dark >= 1
            and white_std <= 28.0
        ):
            return True

        # Rule B: Green border combined with some text evidence (less overall white needed)
        if green_fraction >= 0.010 and max_row_dark_fraction >= 0.03:
            return True

        # Rule C: Very large white fraction (rare) but must still show some text rows
        if white_fraction >= 0.52 and max_row_dark_fraction >= 0.03 and rows_with_dark >= 1:
            return True

        # Conservative single-row fallback removed/strengthened: require both a bright row and
        # a small but non-trivial dark concentration elsewhere to avoid bright UI elements.
        white_row_thresh = 0.95
        for r_idx in range(region.shape[0]):
            row_pixels = region[r_idx, :, :]
            rr = row_pixels[:, :, 0].astype(int)
            gg = row_pixels[:, :, 1].astype(int)
            bb = row_pixels[:, :, 2].astype(int)
            row_lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            row_white_mask = row_lum >= white_thresh
            row_white_fraction = float(row_white_mask.sum()) / float(row_white_mask.size)
            if row_white_fraction >= white_row_thresh:
                # Only accept this row-based hit if there's at least some dark pixels in the region (text evidence)
                if dark_fraction >= 0.02 or max_row_dark_fraction >= 0.04:
                    return True

        return False
    except Exception:
        # On errors be conservative and do NOT claim dialog
        return False


# # Original VLM-based implementation (commented out)
# def is_dialog_open_vlm(state: dict) -> bool:
#     """
#     Check if a dialog or text box is currently displayed using VLM.
#
#     This function uses VLM to analyze the screenshot and detect dialogs.
#
#     Args:
#         state: State dict with VLM support
#
#     Returns:
#         bool: True if dialog is open, False otherwise
#     """
#     try:
#         from utils.vlm_state import add_to_state_schema
#
#         # Register VLM query for dialog detection
#         add_to_state_schema(
#             key="has_dialog_box",
#             vlm_prompt="Is there a dialog box, text message, or NPC speech displayed on screen? Look for text boxes, speech bubbles, or any UI overlay with text. Answer true if any dialog/text is visible, false if only the game world is shown.",
#             return_type=bool
#         )
#
#         has_dialog = state["has_dialog_box"]
#
#         if has_dialog:
#             logger.info("Dialog detected by VLM")
#         else:
#             logger.debug("No dialog detected by VLM")
#
#         return has_dialog
#
#     except Exception as e:
#         logger.error(f"Dialog checker error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Original function signature (now integrated into is_dialog_open)
# def has_dialogue_box(
#     frame: np.ndarray,
#     white_tolerance: int = 12,
# ) -> bool:
#     """Detect whether a dialogue/text box is visible near the bottom of the frame.
#
#     This function is now integrated into is_dialog_open() as _detect_dialogue_box().
#     """
#     return _detect_dialogue_box(frame, white_tolerance)
