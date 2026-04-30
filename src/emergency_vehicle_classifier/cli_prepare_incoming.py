from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .dataset_split import run_split

INCOMING_ROOT = Path("data/incoming")
EXPECTED_CLASSES = ("emergency_vehicle", "non_emergency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/processed from data/incoming: two folders of images, "
            "then train/val/test split. Optionally remove synthetic smoke data."
        ),
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        default=INCOMING_ROOT,
        help="Root with emergency_vehicle/ and non_emergency/ subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Destination root for train/val/test.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction per class.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation fraction per class.")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-replace-processed",
        action="store_true",
        help="Do not delete existing train/val/test under --output before copying.",
    )
    parser.add_argument(
        "--keep-smoke",
        action="store_true",
        help="Do not delete data/smoke_processed (synthetic smoke-test images).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split plan only; do not copy or delete files.",
    )
    return parser.parse_args()


def _require_images(incoming: Path) -> None:
    from .dataset_layout import IMAGE_EXTENSIONS, list_images

    errors: list[str] = []
    for name in EXPECTED_CLASSES:
        class_dir = incoming / name
        n = len(list_images(class_dir))
        if n == 0:
            errors.append(f"{name}: no images ({sorted(IMAGE_EXTENSIONS)})")
        elif n < 2:
            errors.append(f"{name}: need at least 2 images (got {n})")
    if errors:
        raise ValueError(
            "Incoming folders need at least 2 images per class (train/val/test split).\n  - "
            + "\n  - ".join(errors)
        )


def _remove_smoke_processed(*, dry_run: bool) -> None:
    smoke = Path("data/smoke_processed")
    if not smoke.exists():
        return
    if dry_run:
        print({"dry_run_remove_smoke_processed": str(smoke)})
        return
    shutil.rmtree(smoke)
    print({"removed": str(smoke)})


def main() -> None:
    args = parse_args()
    incoming = args.incoming.resolve()
    output_root = args.output.resolve()
    incoming.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_CLASSES:
        (incoming / name).mkdir(parents=True, exist_ok=True)
    _require_images(incoming)

    if not args.keep_smoke:
        _remove_smoke_processed(dry_run=args.dry_run)

    summary = run_split(
        source=incoming,
        output_root=output_root,
        train_r=args.train,
        val_r=args.val,
        test_r=args.test,
        seed=args.seed,
        dry_run=args.dry_run,
        replace_output_splits=not args.no_replace_processed,
        allowed_class_names=EXPECTED_CLASSES,
    )
    print(summary)


if __name__ == "__main__":
    main()
