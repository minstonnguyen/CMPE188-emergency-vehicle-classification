"""Export a model checkpoint to ONNX.

Run from the project root:
    python scripts/export_onnx.py models/best_model.pt
    python scripts/export_onnx.py models/best_model.pt --output models/best_model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from emergency_vehicle_classifier.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a .pt checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (default: same location as checkpoint with .onnx extension).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    out_path = args.output or args.checkpoint.with_suffix(".onnx")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_name: str = str(ckpt.get("model_name", "resnet50"))
    class_names: list[str] = ckpt.get("class_names", [])
    image_size: int = int(ckpt.get("image_size", 224))

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : {model_name}")
    print(f"Classes    : {class_names}")
    print(f"Image size : {image_size}")
    print(f"Output     : {out_path}")
    print(f"Opset      : {args.opset}")

    model = build_model(model_name, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, image_size, image_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"\nExported → {out_path}")
    print(f"Input  : image  [batch, 3, {image_size}, {image_size}]  float32")
    print(f"Output : logits [batch, {len(class_names)}]  float32  (raw, apply softmax for probabilities)")
    print(f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(class_names))}")


if __name__ == "__main__":
    main()
