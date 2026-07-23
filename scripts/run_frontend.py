#!/usr/bin/env python3
"""Streamlit launcher with Python 3.14 asyncio compatibility fix."""
import asyncio
import sys

# Python 3.14 removed the implicit event loop creation in get_event_loop().
# Streamlit's bootstrap.py calls asyncio.get_event_loop() which now raises
# RuntimeError. We monkey-patch it to create a loop if none exists.
_original_get_event_loop = asyncio.get_event_loop


def _patched_get_event_loop():
    try:
        return _original_get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


asyncio.get_event_loop = _patched_get_event_loop

# Now run streamlit normally
from streamlit.web import cli

sys.argv = ["streamlit", "run", "frontend/streamlit_app.py"]
cli.main()
