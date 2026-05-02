"""Normalize images to the 224×224 RGB format expected by the ResNet-50 backbone.

Each image is:
  1. Converted to RGB (strips alpha, handles palette-mode images, etc.)
  2. Letterboxed to a square with black padding, preserving aspect ratio
  3. Resized (up or down) to 224×224 with high-quality resampling
  4. Saved as JPEG (quality 95) in a mirrored output directory

The input directory is never modified.  The output directory is created next
to the input, named "<input>_normalized", or overridden with --output.

Usage (from project root)
--------------------------
    python scripts/normalize_images.py data/incoming_cleaned

    # custom size / output dir
    python scripts/normalize_images.py data/incoming_cleaned --size 224 --output data/ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

TARGET_SIZE = 224  # ResNet-50 expects 224×224

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"
}


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Fit *img* into a *size*×*size* black-padded square.

    The image is scaled so its longest side equals *size* (upscaling is
    allowed).  The shorter side is centred with black bars on each side.
    """
    img = img.convert("RGB")
    orig_w, orig_h = img.size

    scale = size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))

    resampler = Image.Resampling.LANCZOS
    resized = img.resize((new_w, new_h), resampler)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Letterbox-resize images to ResNet-50 input format (224×224 RGB JPEG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root input directory to process recursively.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TARGET_SIZE,
        metavar="PX",
        help="Output square side length in pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output root directory. "
            "Defaults to a sibling of the input named '<input>_normalized'."
        ),
    )
    args = parser.parse_args()

    input_root = args.directory.resolve()
    if not input_root.is_dir():
        parser.error(f"Not a directory: {input_root}")

    output_root: Path = (
        args.output.resolve()
        if args.output is not None
        else input_root.parent / (input_root.name + "_normalized")
    )

    images = collect_images(input_root)
    if not images:
        print(f"No image files found under {input_root}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(images)} image(s) under {input_root}")
    print(f"Target size            : {args.size}×{args.size}")
    print(f"Output directory       : {output_root}\n")

    ok = 0
    errors = 0

    for src in images:
        rel = src.relative_to(input_root)
        dst_dir = output_root / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / (src.stem + ".jpg")

        try:
            img = Image.open(src)
            out = letterbox(img, args.size)
            out.save(dst, "JPEG", quality=95)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] {rel}: {exc}")
            errors += 1

    print(
        f"\nDone.\n"
        f"  Converted : {ok}\n"
        f"  Errors    : {errors}\n"
        f"  Output    : {output_root}"
    )


if __name__ == "__main__":
    main()
