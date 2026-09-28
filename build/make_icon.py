"""
Build the app icon, and the page's small marks, from the NAMTRIX Profiler logo.

The logo (brand/namtrix_profiler_icon.png, 1254 px) is a black rounded square
on a black ground. macOS wants the square alone on transparency, so it is cut
out along its own edge - measured on the artwork: x 110-1140, y 114-1144,
corner radius 214 - with an anti-aliased mask, and placed on Apple's icon grid
(an 824 px body on a 1024 px canvas).

Writes build/namtrix.icns, and brand/namtrix-profiler-appicon.png (1024 px),
brand/namtrix-profiler-mark.png (96 px, the page header) and
brand/namtrix-profiler-favicon.png (64 px, the browser tab).
"""

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "brand" / "namtrix_profiler_icon.png"
BOX = (110, 114, 1141, 1145)       # the rounded square, measured on the artwork
RADIUS = 214                        # its corner radius, likewise
MASTER = 1024
BODY = 824                          # Apple's grid: 100 px clear on every side
SUPERSAMPLE = 4


def cut_out(size: int) -> Image.Image:
    """The rounded square alone, size x size, transparent outside its edge."""
    art = Image.open(SOURCE).convert("RGB").crop(BOX)
    art = art.resize((size, size), Image.LANCZOS)
    big = size * SUPERSAMPLE
    radius = RADIUS * big / (BOX[2] - BOX[0])
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, big - 1, big - 1), radius=radius, fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)
    out = art.convert("RGBA")
    out.putalpha(mask)
    return out


def main():
    canvas = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    body = cut_out(BODY)
    canvas.paste(body, ((MASTER - BODY) // 2, (MASTER - BODY) // 2), body)
    canvas.save(ROOT / "brand" / "namtrix-profiler-appicon.png")
    cut_out(96).save(ROOT / "brand" / "namtrix-profiler-mark.png", optimize=True)
    cut_out(64).save(ROOT / "brand" / "namtrix-profiler-favicon.png", optimize=True)

    iconset = ROOT / "build" / "namtrix.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for old in iconset.glob("*.png"):
        old.unlink()
    for size in (16, 32, 64, 128, 256, 512):
        canvas.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        canvas.resize((size * 2, size * 2), Image.LANCZOS).save(
            iconset / f"icon_{size}x{size}@2x.png"
        )

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(ROOT / "build" / "namtrix.icns")],
        check=True,
    )
    print(f"    icon: {(ROOT / 'build' / 'namtrix.icns').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
