"""Generate placeholder icons for the Chrome extension.

Run once: python scripts/make_icons.py
Produces 16x16, 48x48, and 128x128 PNGs with "BA" on the brand background.
Replace with real art later.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

BG = (26, 26, 46, 255)         # #1a1a2e
FG = (255, 255, 255, 255)      # white
ACCENT = (79, 195, 247, 255)   # #4fc3f7

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "extension" / "icons"


def _font(size: int) -> ImageFont.ImageFont:
    """Best-effort font: try DejaVuSans first (ships with Pillow on most
    systems), fall back to the default font. Returns a font sized so the
    "BA" text fills ~60% of the icon height."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    target = int(size * 0.55)
    for path in candidates:
        if pathlib.Path(path).exists():
            try:
                return ImageFont.truetype(path, target)
            except OSError:
                continue
    return ImageFont.load_default()


def make(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), BG)
    draw = ImageDraw.Draw(img)

    # Rounded background accent dot
    pad = max(1, size // 8)
    r = max(1, size // 6)
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=r,
        fill=ACCENT,
    )

    font = _font(size)
    text = "BA"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=BG, font=font)

    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        img = make(size)
        out = OUT_DIR / f"icon{size}.png"
        img.save(out)
        print(f"wrote {out} ({size}x{size})")


if __name__ == "__main__":
    main()
