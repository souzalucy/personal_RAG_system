#!/usr/bin/env bash
# Run the Streamlit frontend with Python 3.14 compatibility workaround
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec python3 scripts/run_frontend.py "$@"
