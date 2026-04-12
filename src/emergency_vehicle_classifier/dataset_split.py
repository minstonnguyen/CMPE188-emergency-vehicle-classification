from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from .dataset_layout import IMAGE_EXTENSIONS, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a labeled image folder (one subfolder per class) into "
            "train/val/test under the output directory."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory with one subfolder per class (e.g. emergency_vehicle/, non_emergency/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Destination root; will contain train/, val/, and test/ with the same class names.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Fraction of images per class for the training split.",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Fraction of images per class for validation.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Fraction of images per class for the test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts without copying files.",
    )
    return parser.parse_args()


def _validate_ratios(train_r: float, val_r: float, test_r: float) -> None:
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"train + val + test must sum to 1.0, got {total}")


def split_class_files(paths: list[Path], train_r: float, val_r: float, test_r: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    if not paths:
        return [], [], []
    val_plus_test = val_r + test_r
    train_paths, temp_paths = train_test_split(
        paths,
        test_size=val_plus_test,
        random_state=seed,
    )
    if not temp_paths:
        return train_paths, [], []
    test_fraction_of_temp = test_r / val_plus_test if val_plus_test > 0 else 0.0
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=test_fraction_of_temp,
        random_state=seed,
    )
    return list(train_paths), list(val_paths), list(test_paths)


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train, args.val, args.test)

    source = args.source.resolve()
    output_root = args.output.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    class_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories under {source}")

    plan: dict[str, dict[str, int]] = {}
    splits: dict[str, tuple[list[Path], list[Path], list[Path]]] = {}

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            raise ValueError(
                f"No images with extensions {sorted(IMAGE_EXTENSIONS)} in {class_dir}"
            )
        train_p, val_p, test_p = split_class_files(
            images, args.train, args.val, args.test, args.seed
        )
        class_name = class_dir.name
        splits[class_name] = (train_p, val_p, test_p)
        plan[class_name] = {
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
        }

    if args.dry_run:
        print({"output": str(output_root), "plan": plan})
        return

    for class_name, (train_p, val_p, test_p) in splits.items():
        for split_name, subset in (
            ("train", train_p),
            ("val", val_p),
            ("test", test_p),
        ):
            dest_dir = output_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in subset:
                shutil.copy2(src, dest_dir / src.name)

    print({"output": str(output_root), "plan": plan})


if __name__ == "__main__":
    main()
