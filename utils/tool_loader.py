#!/usr/bin/env python3
"""
Simple tool loader for CodeAgent

Scans tools/ directory and loads all public functions.
No state, no caching, no complexity - just load and return.
"""

import os
import sys
import importlib.util
import inspect
import logging

logger = logging.getLogger(__name__)


def load_tools(tools_dir='tools', force_reload=False):
    """
    Load all functions from tools/ directory.

    Scans all .py files in the directory and extracts public functions
    (functions that don't start with underscore).

    Args:
        tools_dir: Path to tools directory (default: 'tools')
        force_reload: If True, remove from sys.modules to force reload (default: False)

    Returns:
        dict: {function_name: function_object}

    Example:
        >>> tools = load_tools('tools')
        >>> print(tools.keys())
        dict_keys(['find_path_action'])

        >>> # Force reload to pick up file changes
        >>> tools = load_tools('tools', force_reload=True)
    """
    tools = {}

    if not os.path.exists(tools_dir):
        logger.warning(f"Tools directory '{tools_dir}' not found")
        return tools

    # Scan for .py files
    for filename in os.listdir(tools_dir):
        if not filename.endswith('.py') or filename.startswith('_'):
            continue

        filepath = os.path.join(tools_dir, filename)
        module_name = f"tools_{filename[:-3]}"  # Unique module name

        try:
            # Check if already in cache and not forcing reload
            if not force_reload and module_name in sys.modules:
                # Use cached module
                module = sys.modules[module_name]
                logger.debug(f"Using cached module: {module_name}")
            else:
                # Force reload if requested (remove from cache)
                if force_reload and module_name in sys.modules:
                    del sys.modules[module_name]
                    logger.debug(f"Removed {module_name} from sys.modules for reload")

                # Import module dynamically
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not load spec for {filepath}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Add to sys.modules for caching
                sys.modules[module_name] = module
                logger.debug(f"Loaded and cached module: {module_name}")

            # Extract all public functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith('_'):
                    tools[name] = obj
                    logger.debug(f"Loaded tool: {name} from {filename}")

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    return tools


def format_tools_for_prompt(tools):
    """
    Format tool functions for prompt display.

    Extracts function signature and docstring for each tool.

    Args:
        tools: Dict of {function_name: function_object}

    Returns:
        str: Formatted documentation string

    Example:
        >>> tools = load_tools('tools')
        >>> doc = format_tools_for_prompt(tools)
        >>> print(doc)
        ### find_path_action(state, goal_x, goal_y, max_distance=50)

        Find next action to reach goal using A* pathfinding.
        ...
    """
    if not tools:
        return "No tools available."

    lines = []
    for name, func in sorted(tools.items()):
        # Get function signature
        try:
            sig = inspect.signature(func)
            lines.append(f"### {name}{sig}")
            lines.append("")
        except Exception as e:
            lines.append(f"### {name}(...)")
            lines.append("")
            logger.warning(f"Could not get signature for {name}: {e}")

        # Get docstring
        doc = inspect.getdoc(func)
        if doc:
            lines.append(doc)
        else:
            lines.append("(No documentation)")

        lines.append("")
        lines.append("---")
        lines.append("")

    return '\n'.join(lines)
