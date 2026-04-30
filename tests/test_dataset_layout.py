from __future__ import annotations

from pathlib import Path

from PIL import Image

from emergency_vehicle_classifier.dataset_layout import (
    IMAGE_EXTENSIONS,
    count_by_class,
    inspect_processed_layout,
    list_images,
)


def test_image_extensions_lowercase(tmp_path: Path) -> None:
    d = tmp_path / "cls"
    d.mkdir()
    for ext in IMAGE_EXTENSIONS:
        (d / f"a{ext}").write_bytes(b"\x89PNG\r\n\x1a\n" if ext == ".png" else b"x")

    paths = list_images(d)
    assert len(paths) == len(IMAGE_EXTENSIONS)


def test_list_images_empty_when_missing(tmp_path: Path) -> None:
    assert list_images(tmp_path / "nope") == []


def test_count_by_class_skips_non_directories(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    (root / "file.txt").write_text("x", encoding="utf-8")
    cls = root / "emergency_vehicle"
    cls.mkdir()
    img = cls / "x.png"
    Image.new("RGB", (4, 4), color=(200, 10, 10)).save(img)

    counts = count_by_class(root)
    assert counts == {"emergency_vehicle": 1}


def test_inspect_processed_layout_ok(tmp_path: Path) -> None:
    base = tmp_path / "processed"
    for split in ("train", "val", "test"):
        for cls in ("emergency_vehicle", "non_emergency"):
            p = base / split / cls
            p.mkdir(parents=True)
            Image.new("RGB", (4, 4), color=(10, 10, 200)).save(p / f"{split}_{cls}.png")

    report = inspect_processed_layout(base)
    assert report["layout_ok"] is True
    assert set(report["class_names"]) == {"emergency_vehicle", "non_emergency"}
    assert report["warnings"] == []
