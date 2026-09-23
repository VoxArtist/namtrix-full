#!/bin/zsh
# Double-click to start the NAMTRIX Full capture bridge.
cd "${0:A:h}"

PY="/Users/miguelmarques/Documents/Codex/2026-09-17/i-x20/work/nam-venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

echo "Starting the NAMTRIX bridge with: $PY"
echo "Open http://127.0.0.1:8765 once it says 'audio ready'."
echo "Leave this window open while recording; Ctrl-C to stop."
echo
exec "$PY" bridge/namtrix_bridge.py --port 8765
