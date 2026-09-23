#!/bin/zsh
# Run NAMTRIX Full from this folder, for development.
#
# Most people should use "NAMTRIX Full.app" instead - it carries its own Python
# and needs nothing installed. This script is for working on the source.
cd "${0:A:h}"

PY="${NAMTRIX_PYTHON:-python3}"
if ! "$PY" -c "import numpy, sounddevice" 2>/dev/null; then
  echo "This needs numpy and sounddevice:"
  echo "    $PY -m pip install numpy sounddevice"
  echo
  echo "Or just double-click \"NAMTRIX Full.app\", which needs neither."
  exit 1
fi

exec "$PY" bridge/namtrix_bridge.py
