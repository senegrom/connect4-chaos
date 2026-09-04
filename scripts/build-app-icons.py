"""Rebuild Home Screen PNGs from the saved logo (requires Pillow).

Run from any directory: python3 scripts/build-app-icons.py
The game and its normal deployment do not require Pillow.
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/connect4-chaos-logo.png"
BACKGROUND = "#0a0a1a"


def main() -> None:
    with Image.open(SOURCE) as check:
        check.verify()
    with Image.open(SOURCE) as original:
        logo = original.convert("RGBA")
    # Ignore nearly transparent edge specks when measuring the artwork.
    bounds = logo.getchannel("A").point(lambda value: 255 if value > 128 else 0).getbbox()
    if bounds is None:
        raise ValueError("The source logo is empty")
    logo = logo.crop(bounds)
    (ROOT / "icons").mkdir(exist_ok=True)

    for size, maskable in [(180, False), (192, False), (512, False), (512, True)]:
        canvas = Image.new("RGB", (size, size), BACKGROUND)
        # A square inside 56% of the canvas fits wholly inside the manifest's
        # circular maskable safe area (radius 40% of the canvas width).
        edge = round(size * (0.56 if maskable else 0.9))
        scale = edge / max(logo.size)
        artwork = logo.resize(
            (round(logo.width * scale), round(logo.height * scale)),
            Image.Resampling.LANCZOS,
        )
        xy = ((size - artwork.width) // 2, (size - artwork.height) // 2)
        canvas.paste(artwork, xy, artwork)
        suffix = "-maskable" if maskable else ""
        path = ROOT / f"icons/connect4-chaos-{size}{suffix}.png"
        canvas.save(path, format="PNG", optimize=True)
        with Image.open(path) as check:
            check.verify()
        with Image.open(path) as check:
            check.load()
            assert check.size == (size, size)
            assert check.convert("RGBA").getchannel("A").getextrema() == (255, 255)
        print(f"Verified {path.relative_to(ROOT)}: {size}x{size}, opaque PNG")
        if size == 180:
            (ROOT / "apple-touch-icon.png").write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
