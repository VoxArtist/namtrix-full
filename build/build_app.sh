#!/bin/zsh
# Build NAMTRIX Full.app. This is ours, not the user's: what they receive is the
# finished .app, which needs nothing installed to run.
set -e
cd "${0:A:h}/.."
ROOT="$PWD"
VENV="${NAMTRIX_BUILD_VENV:-$ROOT/build/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating build environment"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet numpy sounddevice pyinstaller pillow
fi

echo "==> icon"
"$VENV/bin/python" build/make_icon.py

echo "==> bundling"
rm -rf build/dist build/work
"$VENV/bin/pyinstaller" build/namtrix.spec \
  --distpath build/dist --workpath build/work --noconfirm --clean

APP="build/dist/NAMTRIX Full.app"

# Ad-hoc signing. It cannot replace notarisation - without a paid Developer ID
# the first launch still needs right-click > Open - but an unsigned bundle is
# refused outright on Apple silicon, so this is the difference between "one
# extra click" and "will not start".
echo "==> signing"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "    signature ok"

echo "==> zipping"
# The app and a first-run note together, because the one confusing moment -
# Gatekeeper refusing a plain double-click - happens before anyone reads a
# README on the web.
rm -rf build/dist/payload "build/dist/NAMTRIX-Full-macOS.zip"
mkdir -p build/dist/payload
ditto "$APP" "build/dist/payload/NAMTRIX Full.app"
cp build/FIRST-RUN.txt "build/dist/payload/Read me first.txt"
ditto -c -k --sequesterRsrc "build/dist/payload" "build/dist/NAMTRIX-Full-macOS.zip"
rm -rf build/dist/payload

du -sh "$APP" "build/dist/NAMTRIX-Full-macOS.zip"
echo "==> done: $APP"
