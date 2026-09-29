#!/bin/zsh
# Build NAMTRIX Full.app. This is ours, not the user's: what they receive is the
# finished .app, which needs nothing installed to run.
set -e
cd "${0:A:h}/.."
ROOT="$PWD"
VENV="${NAMTRIX_BUILD_VENV:-$ROOT/build/.venv-macos11}"
# The oldest macOS the app runs on. Everything bundled has to be built for it:
# the Python a Homebrew install provides is built for the macOS it was
# installed on (15, here), and an app carrying it refuses to open on 14.
MIN_MACOS="11.0"
PYTHON_VERSION="3.12.13"

# uv installs the trainer on the user's Mac when they ask for it. It ships
# inside the app, pinned, so what gets installed does not depend on whatever
# uv happens to be current on release day. It also provides the build's Python.
UV_VERSION="0.11.1"
if [[ ! -x build/uv ]] || [[ "$(build/uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
  echo "==> fetching uv $UV_VERSION"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" \
    | tar -xz -C build --strip-components=1 uv-aarch64-apple-darwin/uv
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating build environment (Python $PYTHON_VERSION, wheels for macOS $MIN_MACOS)"
  # uv's own CPython builds target macOS 11; the wheels are picked for
  # MIN_MACOS, not for this Mac, so numpy comes as its macOS 11 build.
  UV_PYTHON_PREFERENCE=only-managed build/uv venv --quiet "$VENV" --python "$PYTHON_VERSION"
  MACOSX_DEPLOYMENT_TARGET="$MIN_MACOS" build/uv pip install --quiet --python "$VENV/bin/python" \
    --python-platform aarch64-apple-darwin --only-binary :all: \
    numpy sounddevice cffi pyinstaller pillow
fi

echo "==> icon"
"$VENV/bin/python" build/make_icon.py

echo "==> bundling"
rm -rf build/dist build/work
"$VENV/bin/pyinstaller" build/namtrix.spec \
  --distpath build/dist --workpath build/work --noconfirm --clean

APP="build/dist/NAMTRIX Full.app"

# Nothing in the bundle may need a newer macOS than the app says it needs:
# macOS checks the declared minimum, then each library refuses on its own,
# and that second refusal is a crash with no message.
echo "==> checking every binary runs on macOS $MIN_MACOS"
declared=$(/usr/libexec/PlistBuddy -c "Print :LSMinimumSystemVersion" "$APP/Contents/Info.plist")
[[ "$declared" == "$MIN_MACOS" ]] || { echo "    Info.plist declares $declared, the build targets $MIN_MACOS"; exit 1; }
autoload -U is-at-least
checked=0; too_new=""
while IFS= read -r -d '' f; do
  v=$(vtool -show-build "$f" 2>/dev/null | awk '/minos/{print $2; exit}')
  [[ -n "$v" ]] || continue
  checked=$((checked + 1))
  is-at-least "$v" "$MIN_MACOS" || too_new+="$v ${f#$APP/Contents/}"$'\n'
done < <(find "$APP/Contents" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -111 \) -print0)
(( checked > 0 )) || { echo "    no binaries were checked - the check itself is broken"; exit 1; }
if [[ -n "$too_new" ]]; then
  echo "    these need a newer macOS than $MIN_MACOS:"; printf "%s" "$too_new" | head -20; exit 1
fi
echo "    all $checked binaries run on macOS $MIN_MACOS"

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
