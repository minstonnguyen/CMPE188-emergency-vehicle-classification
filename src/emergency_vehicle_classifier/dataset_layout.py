from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_by_class(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        counts[class_dir.name] = len(list_images(class_dir))
    return counts


def inspect_processed_layout(data_dir: Path) -> dict:
    """Validate train/val/test ImageFolder layout under ``data_dir``."""
    splits = ("train", "val", "test")
    per_split: dict[str, dict[str, int]] = {}
    missing_dirs: list[str] = []
    for name in splits:
        path = data_dir / name
        if not path.is_dir():
            missing_dirs.append(name)
            per_split[name] = {}
            continue
        per_split[name] = count_by_class(path)

    class_sets = [set(per_split[s].keys()) for s in splits if per_split[s]]
    if not class_sets:
        consistent = True
        unified_classes: list[str] = []
    else:
        unified = (
            class_sets[0].intersection(*class_sets[1:]) if len(class_sets) > 1 else class_sets[0]
        )
        union = set().union(*class_sets)
        consistent = len(unified) == len(union) and len(union) > 0
        unified_classes = sorted(union)

    warnings: list[str] = []
    if missing_dirs:
        warnings.append(f"Missing split directories: {', '.join(missing_dirs)}")
    if class_sets and not consistent:
        warnings.append("Class folder names differ across train/val/test or some splits are empty.")
    for s in splits:
        for cls, n in per_split.get(s, {}).items():
            if n == 0:
                warnings.append(f"No images in {s}/{cls}/")

    return {
        "data_dir": str(data_dir),
        "splits": per_split,
        "class_names": unified_classes,
        "layout_ok": len(missing_dirs) == 0 and consistent and len(unified_classes) > 0,
        "warnings": warnings,
    }
