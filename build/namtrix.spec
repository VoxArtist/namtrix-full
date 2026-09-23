# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for NAMTRIX Full.

Bundles CPython, numpy, sounddevice and PortAudio's own dylib alongside the page,
so the person who receives this installs nothing at all. The alternative - "have
Python and pip install sounddevice" - is a terminal session, and the point of
this build is that there isn't one.
"""

from pathlib import Path

import sounddevice as _sd

ROOT = Path(SPECPATH).parent

# PortAudio ships inside the sounddevice wheel as _sounddevice_data. PyInstaller
# does not find it on its own, so it is named here.
portaudio = Path(_sd.__file__).parent / "_sounddevice_data"

a = Analysis(
    [str(ROOT / "bridge" / "namtrix_bridge.py")],
    pathex=[str(ROOT / "bridge")],
    datas=[
        (str(ROOT / "index.html"), "."),
        (str(ROOT / "favicon.svg"), "."),
        # The reamp signals, so nobody has to find or point at a DI file.
        (str(ROOT / "signals"), "signals"),
        # Run by the trainer's own Python, not ours, so it ships as a plain file.
        (str(ROOT / "bridge" / "validate_model.py"), "."),
        # What "Install trainer" puts on the Mac, and the tool that puts it there.
        (str(ROOT / "bridge" / "trainer-requirements.txt"), "."),
        (str(portaudio), "_sounddevice_data"),
    ],
    binaries=[
        (str(ROOT / "build" / "uv"), "."),
    ],
    hiddenimports=["latency", "training", "_cffi_backend"],
    hookspath=[],
    runtime_hooks=[],
    # Nothing here wants a GUI toolkit, a plotting library or a test runner, and
    # each one dragged in costs tens of megabytes.
    excludes=["tkinter", "matplotlib", "PIL", "pytest", "IPython", "torch",
              "scipy", "pandas", "setuptools", "pip"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="NAMTRIX Full",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no Terminal window
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="NAMTRIX Full",
)
app = BUNDLE(
    coll,
    name="NAMTRIX Full.app",
    icon=str(ROOT / "build" / "namtrix.icns"),
    bundle_identifier="pt.voxartist.namtrix.full",
    info_plist={
        "CFBundleName": "NAMTRIX Full",
        "CFBundleDisplayName": "NAMTRIX Full",
        "CFBundleShortVersionString": "0.7.3",
        "CFBundleVersion": "0.7.3",
        "LSMinimumSystemVersion": "11.0",
        # Without this the microphone prompt never appears and recordings come
        # back as digital silence - an hour lost to something that looks like a
        # patching mistake.
        "NSMicrophoneUsageDescription":
            "NAMTRIX records the amp's response through your audio interface. "
            "macOS counts any audio input as microphone access.",
        "NSHighResolutionCapable": True,
    },
)
