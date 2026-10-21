#!/usr/bin/env bash
# CorridorKey 2.0 GUI launcher. Extra arguments go to gui/app.py,
# e.g. ./gui/launch.sh --listen --auth me:secret
cd "$(dirname "$0")/.." || exit 1

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" gui/app.py "$@"
