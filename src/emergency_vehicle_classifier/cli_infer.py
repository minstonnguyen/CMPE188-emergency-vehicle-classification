from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on one image or every image in a directory.",
    )
    parser.add_argument("--image", default=None, help="Path to a single image file.")
    parser.add_argument(
        "--image-dir",
        default=None,
        dest="image_dir",
        help="Directory of images (non-recursive). Exactly one of --image or --image-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this path (.json or .csv). Default: print JSON to stdout.",
    )
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _list_images_in_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(entry)
    return paths


def _load_model_bundle(checkpoint_path: Path, device):
    import torch

    from .model import SmallCNN

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    image_size: int = int(checkpoint["image_size"])
    model = SmallCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


def _scores_dict(class_names: list[str], probabilities) -> dict[str, float]:
    return {
        name: round(float(score), 4)
        for name, score in zip(class_names, probabilities.tolist(), strict=True)
    }


def _run_one(
    model,
    class_names: list[str],
    image_path: Path,
    image_size: int,
    device,
) -> dict:
    import torch

    from .data import load_image

    image = load_image(image_path, image_size).to(device)
    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prediction_idx = int(torch.argmax(probabilities).item())
    return {
        "path": str(image_path.resolve()),
        "predicted_class": class_names[prediction_idx],
        "confidence": round(float(probabilities[prediction_idx].item()), 4),
        "scores": _scores_dict(class_names, probabilities),
    }


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "predicted_class", "confidence", "scores_json"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "predicted_class": row["predicted_class"],
                    "confidence": row["confidence"],
                    "scores_json": json.dumps(row.get("scores", {})),
                }
            )


def main() -> None:
    args = parse_args()
    if (args.image is None) == (args.image_dir is None):
        raise SystemExit("Specify exactly one of --image or --image-dir.")

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, class_names, image_size = _load_model_bundle(checkpoint_path, device)

    if args.image is not None:
        rows = [_run_one(model, class_names, Path(args.image), image_size, device)]
    else:
        paths = _list_images_in_dir(Path(args.image_dir))
        rows = [_run_one(model, class_names, p, image_size, device) for p in paths]

    single_payload = len(rows) == 1 and args.image is not None
    if args.output:
        out = Path(args.output)
        suffix = out.suffix.lower()
        if suffix == ".csv":
            _write_csv(rows, out)
        elif suffix == ".json":
            out.parent.mkdir(parents=True, exist_ok=True)
            to_dump: object = rows[0] if single_payload else rows
            with out.open("w", encoding="utf-8") as handle:
                json.dump(to_dump, handle, indent=2)
        else:
            raise SystemExit("Use --output with a path ending in .json or .csv")
    else:
        to_print: object = rows[0] if single_payload else rows
        print(json.dumps(to_print, indent=2))


if __name__ == "__main__":
    main()
