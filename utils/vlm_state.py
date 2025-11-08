#!/usr/bin/env python3
"""
Visual Language Model-aware State object
Supports lazy evaluation of visual queries via VLM
"""

from collections import UserDict
from typing import Any, Callable, Dict, Optional
import threading


class StateSchemaRegistry:
    """
    Thread-safe registry for state schema definitions
    Stores VLM query specifications (key, prompt, return_type)
    """

    def __init__(self):
        self._schemas: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, key: str, vlm_prompt: str, return_type: type):
        """
        Register a schema entry

        Args:
            key: State key to query
            vlm_prompt: Prompt for VLM
            return_type: Expected return type (bool, int, str, float, list, dict)

        Raises:
            ValueError: If return_type is not supported
        """
        with self._lock:
            if return_type not in [bool, int, str, float, list, dict]:
                raise ValueError(f"Unsupported return_type: {return_type}")

            self._schemas[key] = {
                'vlm_prompt': vlm_prompt,
                'return_type': return_type
            }

    def get(self, key: str) -> Optional[dict]:
        """Get schema for a key"""
        with self._lock:
            return self._schemas.get(key)

    def clear(self):
        """Clear all schemas (called before each code generation)"""
        with self._lock:
            self._schemas.clear()

    def has_schema(self, key: str) -> bool:
        """Check if key has a schema"""
        with self._lock:
            return key in self._schemas

    def get_all_keys(self) -> list:
        """Get all registered keys"""
        with self._lock:
            return list(self._schemas.keys())


class State(UserDict):
    """
    Enhanced state dictionary with lazy VLM evaluation

    Usage:
        state = State(base_data, schema_registry, vlm_caller, screenshot)

        # Regular dict access (no VLM call)
        pos = state['player']['position']['x']

        # Schema-registered key (triggers VLM call on first access)
        is_open = state['is_in_clock_ui']  # VLM called here!

        # Subsequent access uses cached result
        is_open_again = state['is_in_clock_ui']  # No VLM call
    """

    def __init__(
        self,
        base_data: dict,
        schema_registry: StateSchemaRegistry,
        vlm_caller: Optional[Callable] = None,
        screenshot = None
    ):
        """
        Initialize State object

        Args:
            base_data: Base game state dict (player, map, game, etc.)
            schema_registry: Schema registry with VLM queries
            vlm_caller: Function to call VLM (screenshot, prompt, return_type) -> value
            screenshot: PIL Image or None
        """
        super().__init__(base_data)
        self._schema_registry = schema_registry
        self._vlm_caller = vlm_caller
        self._screenshot = screenshot
        self._vlm_cache: Dict[str, Any] = {}  # Cache VLM results
        self._access_log: list = []  # Log which keys triggered VLM calls

    def __getitem__(self, key: str) -> Any:
        """
        Override __getitem__ to support lazy VLM evaluation

        Behavior:
        1. If key exists in base data, return it (normal dict behavior)
        2. If key has schema but not in base data:
           a. Check cache first
           b. If not cached, call VLM
           c. Cache and return result
        3. Otherwise, raise KeyError (normal dict behavior)
        """
        # Case 1: Key exists in base data (normal dict access)
        if key in self.data:
            return self.data[key]

        # Case 2: Key has schema (VLM query)
        if self._schema_registry.has_schema(key):
            # Check cache first
            if key in self._vlm_cache:
                return self._vlm_cache[key]

            # Call VLM
            schema = self._schema_registry.get(key)
            if self._vlm_caller is None:
                raise RuntimeError(f"VLM caller not provided for schema key '{key}'")

            if self._screenshot is None:
                raise RuntimeError(f"Screenshot not provided for VLM query '{key}'")

            try:
                result = self._vlm_caller(
                    screenshot=self._screenshot,
                    prompt=schema['vlm_prompt'],
                    return_type=schema['return_type']
                )

                # Cache result
                self._vlm_cache[key] = result

                # Log access
                self._access_log.append({
                    'key': key,
                    'prompt': schema['vlm_prompt'],
                    'return_type': schema['return_type'].__name__,
                    'result': result
                })

                return result

            except Exception as e:
                raise RuntimeError(f"VLM call failed for key '{key}': {e}")

        # Case 3: Key not found (normal dict behavior)
        raise KeyError(key)

    def get_vlm_access_log(self) -> list:
        """Return log of VLM accesses for debugging"""
        return self._access_log.copy()

    def get_vlm_cache(self) -> dict:
        """Return VLM cache for debugging"""
        return self._vlm_cache.copy()


# Global registry (accessible from exec'd code)
_global_schema_registry = StateSchemaRegistry()


def add_to_state_schema(key: str, vlm_prompt: str, return_type: type):
    """
    Global function to register state schema (called from generated code)

    Args:
        key: State key to query
        vlm_prompt: Prompt for VLM to answer
        return_type: Expected type (bool, int, str, float, list, dict)

    Example:
        add_to_state_schema(
            key="is_in_clock_ui",
            vlm_prompt="Is the clock setting UI currently open?",
            return_type=bool
        )
    """
    _global_schema_registry.register(key, vlm_prompt, return_type)


def get_global_schema_registry() -> StateSchemaRegistry:
    """Get the global schema registry"""
    return _global_schema_registry
