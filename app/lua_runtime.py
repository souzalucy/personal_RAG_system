"""Lua runtime engine — executes Lua scripts via lupa (LuaJIT bindings).

This module provides:
1. A LuaRuntime wrapper that loads, caches, and executes Lua scripts.
2. A feature-flag reader that reads lua/features.lua on EVERY call (no caching),
   so changes take effect immediately without redeployment.
3. Helper functions for the RAG pipeline to call Lua logic.
"""

import os
from pathlib import Path
from typing import Any

from lupa import LuaRuntime as _LuaRuntime

from app.config import settings

# ---------------------------------------------------------------------------
# Lua runtime singleton
# ---------------------------------------------------------------------------
_lua: _LuaRuntime | None = None
_script_cache: dict[str, Any] = {}


def _get_runtime() -> _LuaRuntime:
    """Get or create the Lua runtime (singleton)."""
    global _lua
    if _lua is None:
        _lua = _LuaRuntime(unpack_returned_tuples=True)
        # Add the lua/ directory to Lua's package.path so require() works
        lua_dir = str(Path(settings.lua_dir).resolve())
        _lua.execute(f'package.path = "{lua_dir}/?.lua;" .. package.path')
    return _lua


def _clear_cache():
    """Clear the compiled script cache (useful for testing)."""
    global _script_cache
    _script_cache = {}


# ---------------------------------------------------------------------------
# Feature flags — read from disk on EVERY call (hot-reload)
# ---------------------------------------------------------------------------

def get_features() -> dict[str, Any]:
    """Read lua/features.lua from disk and return the parsed config.

    This is called on every request — no caching — so editing the file
    takes effect immediately without redeployment.
    """
    lua = _get_runtime()
    features_path = Path(settings.lua_dir) / "features.lua"
    if not features_path.exists():
        return {}

    # Execute the Lua file fresh each time (no caching)
    try:
        features_table = lua.execute(features_path.read_text())
        return _table_to_dict(features_table)
    except Exception as exc:
        print(f"[lua_runtime] WARNING: Failed to load features.lua: {exc}")
        return {}


def is_enabled(feature_name: str) -> bool:
    """Shorthand: check if a feature flag is enabled.

    Returns False if the feature doesn't exist or is not a table with 'enabled'.
    """
    features = get_features()
    feature = features.get(feature_name)
    if isinstance(feature, dict):
        return bool(feature.get("enabled", False))
    return False


def get_feature_config(feature_name: str) -> dict[str, Any]:
    """Get the full config dict for a feature (including params like keyword_boost)."""
    features = get_features()
    feature = features.get(feature_name)
    if isinstance(feature, dict):
        return feature
    return {}


# ---------------------------------------------------------------------------
# Lua script execution
# ---------------------------------------------------------------------------

def load_script(name: str) -> Any:
    """Load a Lua script by filename (e.g. 'normalize_query.lua').

    Compiled scripts are cached in memory for performance.
    If you edit the Lua file, call clear_cache() or restart the process.
    """
    if name in _script_cache:
        return _script_cache[name]

    script_path = Path(settings.lua_dir) / name
    if not script_path.exists():
        raise FileNotFoundError(f"Lua script not found: {script_path}")

    lua = _get_runtime()
    try:
        # Execute the script and get the returned module table
        chunk = lua.execute(script_path.read_text())
        _script_cache[name] = chunk
        return chunk
    except Exception as exc:
        raise RuntimeError(f"Failed to load Lua script '{name}': {exc}") from exc


def _py_to_lua(value: Any, lua_runtime: _LuaRuntime) -> Any:
    """Recursively convert Python values to Lua-friendly values.

    - Python lists → Lua tables with 1-based integer keys (for ipairs compat)
    - Python dicts → Lua tables
    - Everything else → passed through
    """
    if isinstance(value, list):
        tbl = lua_runtime.table_from()
        for i, item in enumerate(value, start=1):
            tbl[i] = _py_to_lua(item, lua_runtime)
        return tbl
    if isinstance(value, dict):
        tbl = lua_runtime.table_from()
        for k, v in value.items():
            tbl[k] = _py_to_lua(v, lua_runtime)
        return tbl
    return value


def execute(name: str, func_name: str, *args) -> Any:
    """Execute a function from a loaded Lua script.

    Args:
        name: Lua script filename (e.g. 'normalize_query.lua').
        func_name: Function name within the script (e.g. 'normalize').
        *args: Arguments to pass to the Lua function.

    Returns:
        The return value from the Lua function.
    """
    module = load_script(name)
    func = getattr(module, func_name, None)
    if func is None:
        raise AttributeError(
            f"Lua script '{name}' has no function '{func_name}'"
        )
    lua = _get_runtime()
    lua_args = tuple(_py_to_lua(a, lua) for a in args)
    result = func(*lua_args)
    return _table_to_dict(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_to_dict(table: Any) -> Any:
    """Recursively convert a lupa Lua table to a Python dict/list."""
    if table is None:
        return None
    # Check if it's a lupa table object
    try:
        # Test if it behaves like a table
        _ = table.keys
        is_table = True
    except (AttributeError, TypeError):
        is_table = False

    if not is_table:
        # Primitive value
        return table

    # Check if it's a list (all integer keys starting from 1)
    try:
        keys = list(table.keys())
        if keys and all(isinstance(k, int) for k in keys):
            # Could be a list — check if keys are 1..n
            if set(keys) == set(range(1, len(keys) + 1)):
                return [_table_to_dict(table[k]) for k in sorted(keys)]
    except Exception:
        pass

    # It's a dict-like table
    result = {}
    try:
        for key in table.keys():
            try:
                result[str(key)] = _table_to_dict(table[key])
            except Exception:
                result[str(key)] = None
    except Exception:
        pass
    return result


def clear_cache():
    """Clear the compiled script cache.

    Call this if you modify Lua scripts at runtime and want changes
    to take effect without restarting the server.
    """
    _clear_cache()
