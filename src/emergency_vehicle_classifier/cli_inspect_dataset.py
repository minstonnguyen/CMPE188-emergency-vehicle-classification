from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_layout import inspect_processed_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect train/val/test folders and report per-class image counts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing train/, val/, and test/ (ImageFolder layout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object instead of a short text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_processed_layout(args.data_dir)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"data_dir: {report['data_dir']}")
    for split_name in ("train", "val", "test"):
        counts = report["splits"].get(split_name, {})
        total = sum(counts.values())
        print(f"  {split_name}: {total} images — {counts}")
    print(f"  classes: {report['class_names']}")
    print(f"  layout_ok: {report['layout_ok']}")
    for w in report["warnings"]:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
