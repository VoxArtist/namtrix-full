"""
Build the app icon from the brand mark.

The mark is supplied at 214 px and there is no larger original, so it is
upscaled. Lanczos and a padded square canvas keep that as tidy as it can be.
"""

import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASTER = 1024
PAD = 0.14          # macOS icons breathe; a mark filling the square looks wrong


def main():
    mark = Image.open(ROOT / "brand" / "namtrix-mark-N.png").convert("RGBA")
    inner = int(MASTER * (1 - 2 * PAD))
    scale = min(inner / mark.width, inner / mark.height)
    resized = mark.resize(
        (round(mark.width * scale), round(mark.height * scale)), Image.LANCZOS
    )

    canvas = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    canvas.paste(
        resized, ((MASTER - resized.width) // 2, (MASTER - resized.height) // 2), resized
    )

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
