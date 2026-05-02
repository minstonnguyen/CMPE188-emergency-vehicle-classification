from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic dataset for smoke tests."
    )
    parser.add_argument("--output-dir", default="data/processed", help="Dataset output directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny splits for fast CI/smoke runs (overrides default counts).",
    )
    return parser.parse_args()


def _draw_background(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    sky = (rng.randint(170, 220), rng.randint(200, 235), rng.randint(220, 255))
    road = (rng.randint(70, 95), rng.randint(70, 95), rng.randint(70, 95))
    draw.rectangle((0, 0, size, int(size * 0.58)), fill=sky)
    draw.rectangle((0, int(size * 0.58), size, size), fill=road)


def _draw_vehicle(draw: ImageDraw.ImageDraw, size: int, label: str, rng: random.Random) -> None:
    car_w = rng.randint(int(size * 0.42), int(size * 0.55))
    car_h = rng.randint(int(size * 0.16), int(size * 0.22))
    left = rng.randint(8, size - car_w - 8)
    top = rng.randint(int(size * 0.48), int(size * 0.68))
    right = left + car_w
    bottom = top + car_h

    if label == "emergency_vehicle":
        body = (240, 240, 245)
        stripe = (20, 60, 160)
        light_left = (220, 40, 40)
        light_right = (40, 80, 220)
    else:
        body = (
            rng.randint(20, 220),
            rng.randint(20, 220),
            rng.randint(20, 220),
        )
        stripe = None
        light_left = None
        light_right = None

    draw.rounded_rectangle((left, top, right, bottom), radius=8, fill=body)
    roof = (left + int(car_w * 0.2), top - int(car_h * 0.35), right - int(car_w * 0.2), top + 2)
    draw.rounded_rectangle(roof, radius=6, fill=body)

    if stripe is not None:
        stripe_y = top + int(car_h * 0.42)
        draw.rectangle((left + 6, stripe_y, right - 6, stripe_y + 8), fill=stripe)
        bar_left = left + int(car_w * 0.4)
        bar_right = right - int(car_w * 0.4)
        bar_top = roof[1] - 4
        bar_bottom = roof[1] + 3
        draw.rectangle(
            (bar_left, bar_top, (bar_left + bar_right) // 2, bar_bottom), fill=light_left
        )
        draw.rectangle(
            (((bar_left + bar_right) // 2), bar_top, bar_right, bar_bottom), fill=light_right
        )

    wheel_r = 7
    for wheel_x in (left + int(car_w * 0.22), right - int(car_w * 0.22)):
        draw.ellipse(
            (wheel_x - wheel_r, bottom - 2, wheel_x + wheel_r, bottom + 12), fill=(20, 20, 20)
        )


def _make_image(size: int, label: str, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_background(draw, size, rng)
    _draw_vehicle(draw, size, label, rng)
    return image


def _split_counts(quick: bool) -> dict[str, int]:
    if quick:
        return {"train": 8, "val": 4, "test": 4}
    return {"train": 80, "val": 20, "test": 20}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    labels = ["emergency_vehicle", "non_emergency"]
    counts = _split_counts(args.quick)

    for split, count in counts.items():
        for label in labels:
            split_dir = output_dir / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = _make_image(args.image_size, label, rng)
                image.save(split_dir / f"{label}_{index:03d}.png")

    print(
        {
            "output_dir": str(output_dir),
            "splits": counts,
            "classes": labels,
        }
    )


if __name__ == "__main__":
    main()
