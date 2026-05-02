"""Clean a dataset directory by extracting per-car crops via DegiRum inference.

For each image found under the input directory, the YOLOv5 car detection model
is run. Each detected car bounding box is saved as a separate JPEG crop in a
mirrored output directory (sibling of the input, named "<input>_cleaned/").
Images with no car detections are skipped entirely.

Example
-------
    # from project root
    python scripts/clean_dataset_with_degirum.py data/incoming

    # override host, confidence, and output directory
    python scripts/clean_dataset_with_degirum.py data/incoming \\
        --host @local --conf 0.4 --output data/incoming_cleaned
"""

from __future__ import annotations
import openvino
import argparse
import os
import sys
from pathlib import Path

import degirum as dg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "yolov5m_relu6_car--640x640_float_openvino_multidevice_1"
ZOO_URL = "degirum/public"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(root: Path) -> list[Path]:
    """Return a flat list of all image files under *root*."""
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _iou(a: list[float], b: list[float]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def filter_by_iou(bboxes: list[list[float]], iou_threshold: float) -> list[list[float]]:
    """Discard *both* boxes in any pair whose IoU exceeds *iou_threshold*.

    Returns only those boxes that are not involved in any high-overlap pair.
    """
    n = len(bboxes)
    discard = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _iou(bboxes[i], bboxes[j]) > iou_threshold:
                discard[i] = True
                discard[j] = True
    return [bboxes[k] for k in range(n) if not discard[k]]


def crop_and_save(
    pil_image,
    bbox: list[float],
    out_path: Path,
) -> bool:
    """Crop *pil_image* to *bbox* ([x1,y1,x2,y2]) and save as JPEG at *out_path*.

    Returns False (and does not write a file) if the crop area is degenerate.
    """
    img_w, img_h = pil_image.size
    x1 = _clamp(int(bbox[0]), 0, img_w)
    y1 = _clamp(int(bbox[1]), 0, img_h)
    x2 = _clamp(int(bbox[2]), 0, img_w)
    y2 = _clamp(int(bbox[3]), 0, img_h)

    if x2 <= x1 or y2 <= y1:
        return False  # degenerate box

    crop = pil_image.crop((x1, y1, x2, y2))
    # Convert any mode (RGBA, P, …) to RGB so JPEG can be saved cleanly
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    crop.save(out_path, "JPEG", quality=95)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-car crops from a dataset directory using DegiRum inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root input directory (e.g. data/incoming).",
    )
    parser.add_argument(
        "--host",
        default="@local",
        metavar="HOST",
        help=(
            "DegiRum inference host address. "
            "Use '@local' for local OpenVINO execution, '@cloud' for DegiRum cloud, "
            "or an IP/hostname for an AI Server on your LAN."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.65,
        metavar="THRESHOLD",
        help="Minimum detection confidence to keep a bounding box.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output root directory. "
            "Defaults to a sibling of the input named '<input>_cleaned'."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    input_root = args.directory.resolve()
    if not input_root.is_dir():
        parser.error(f"Not a directory: {input_root}")

    output_root: Path = (
        args.output.resolve()
        if args.output is not None
        else input_root.parent / (input_root.name + "_cleaned")
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    token = os.environ.get("DEGIRUM_CLOUD_TOKEN", "")

    print(f"Loading model '{MODEL_NAME}' ...")
    print(f"  zoo_url              : {ZOO_URL}")
    print(f"  inference_host       : {args.host}")
    print(f"  confidence threshold : {args.conf}")
    # Note: if loading fails, try zoo_url="degirum/public" (without leading slash)
    # — the leading-slash form is valid for local zoo directories but may need
    # adjustment depending on your DegiRum installation layout.
    model = dg.load_model(
        model_name=MODEL_NAME,
        inference_host_address=args.host,
        zoo_url=ZOO_URL,
        token=token,
        image_backend="pil",
        output_confidence_threshold=args.conf,
    )
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------
    images = collect_images(input_root)
    if not images:
        print(f"No image files found under {input_root}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(images)} image(s) under {input_root}")
    print(f"Output directory       : {output_root}\n")

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    total_images = 0
    total_crops = 0
    total_skipped = 0

    for image_path in images:
        rel = image_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = model(str(image_path))
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Inference failed for {rel}: {exc}")
            total_skipped += 1
            total_images += 1
            continue

        detections = result.results or []
        total_images += 1

        if not detections:
            total_skipped += 1
            continue

        # Collect bboxes, then remove any pair with IoU > 0.3
        raw_bboxes = [det.get("bbox") for det in detections if det.get("bbox") is not None]
        kept_bboxes = filter_by_iou(raw_bboxes, iou_threshold=0.3)
        iou_dropped = len(raw_bboxes) - len(kept_bboxes)
        if iou_dropped:
            print(f"  [INFO] {rel}: dropped {iou_dropped} box(es) due to IoU > 0.3")

        if not kept_bboxes:
            total_skipped += 1
            continue

        saved = 0
        for i, bbox in enumerate(kept_bboxes):
            out_path = out_dir / f"{image_path.stem}_car{i}.jpg"
            if crop_and_save(result.image, bbox, out_path):
                saved += 1
            else:
                print(f"  [WARN] Degenerate bbox {bbox} in {rel} (detection {i}) — skipped.")

        total_crops += saved

        status = f"{saved} crop(s)" if saved else "no valid crops"
        print(f"  {rel}  →  {status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(
        f"\nDone.\n"
        f"  Images processed : {total_images}\n"
        f"  Crops saved      : {total_crops}\n"
        f"  Images skipped   : {total_skipped} (no detections or error)\n"
        f"  Output root      : {output_root}"
    )


if __name__ == "__main__":
    main()
