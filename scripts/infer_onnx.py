"""Run inference on a single image using an ONNX model via OpenVINO.

Class names are read from a companion .pt checkpoint (same stem, same folder)
if one exists, or can be supplied with --classes. Falls back to class_0, class_1, ...

Run from the project root:
    python scripts/infer_onnx.py models/best_model.onnx path/to/image.jpg
    python scripts/infer_onnx.py models/best_model.onnx image.jpg --device CPU
    python scripts/infer_onnx.py models/best_model.onnx image.jpg --classes emergency_vehicle non_emergency
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX model inference on a single image via OpenVINO."
    )
    parser.add_argument("model", type=Path, help="Path to .onnx model file.")
    parser.add_argument("image", type=Path, help="Path to input image.")
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device: GPU, CPU, AUTO, NPU, ... (default: GPU).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Class names in label-index order. Auto-detected from companion .pt if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size: replicate the image N times per forward pass (default: 1).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up passes before timing (default: 5).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of inference runs for timing (default: 10).",
    )
    return parser.parse_args()


# ImageNet normalisation constants (must match training).
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def load_class_names(model_path: Path, override: list[str] | None) -> list[str]:
    if override:
        return override
    companion = model_path.with_suffix(".pt")
    if companion.exists():
        import torch
        try:
            ckpt = torch.load(companion, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(companion, map_location="cpu")
        names = ckpt.get("class_names")
        if names:
            return list(names)
    return []


def preprocess(image_path: Path, height: int, width: int):
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img = img.resize((width, height), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(_MEAN, dtype=np.float32)
    std  = np.array(_STD,  dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)   # [3, H, W]
    arr = arr[np.newaxis, ...]     # [1, 3, H, W]
    return arr, original_size


def softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


def compile_model(core, model_path: Path, device: str):
    """Compile for the requested device, falling back to CPU if unavailable."""
    import openvino as ov

    available = core.available_devices
    if device not in available and device != "AUTO":
        print(f"[WARN] Device '{device}' not available {available}. Falling back to CPU.")
        device = "CPU"

    ov_model = core.read_model(str(model_path))
    compiled = core.compile_model(ov_model, device)
    return compiled, device


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        sys.exit(f"[ERROR] Model not found: {args.model}")
    if not args.image.exists():
        sys.exit(f"[ERROR] Image not found: {args.image}")

    try:
        import openvino as ov
    except ImportError:
        sys.exit("[ERROR] openvino is required.\n        pip install openvino")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] Pillow is required.\n        pip install Pillow")

    import numpy as np

    class_names = load_class_names(args.model, args.classes)

    # --- Load + compile ---
    core = ov.Core()
    t0 = time.perf_counter()
    compiled, actual_device = compile_model(core, args.model, args.device)
    load_time = time.perf_counter() - t0

    # Read input shape from the model (H, W may be static or dynamic).
    inp_node  = compiled.input(0)
    out_node  = compiled.output(0)
    shape = inp_node.partial_shape  # e.g. [1,3,224,224] or [?,3,?,?]
    height = shape[2].get_length() if shape[2].is_static else 224
    width  = shape[3].get_length() if shape[3].is_static else 224

    if not class_names:
        out_shape = out_node.partial_shape
        n = out_shape[1].get_length() if out_shape[1].is_static else 2
        class_names = [f"class_{i}" for i in range(n)]

    # --- Preprocess ---
    t0 = time.perf_counter()
    single, original_size = preprocess(args.image, height, width)
    batch = np.repeat(single, args.batch_size, axis=0)  # [B, 3, H, W]
    preprocess_time = time.perf_counter() - t0

    # --- Warm-up then timed runs ---
    infer_req = compiled.create_infer_request()
    print(f"Warming up ({args.warmup} passes) ...", end=" ", flush=True)
    for _ in range(args.warmup):
        infer_req.infer({inp_node: batch})
    print("done")

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        infer_req.infer({inp_node: batch})
        times.append(time.perf_counter() - t0)

    logits  = infer_req.get_output_tensor(0).data[0]  # first image in batch [num_classes]
    probs   = softmax(logits)
    top_idx = int(np.argmax(probs))

    infer_mean = float(np.mean(times)) * 1000
    infer_min  = float(np.min(times))  * 1000
    infer_max  = float(np.max(times))  * 1000

    # --- Output ---
    print()
    print(f"Image      : {args.image}  (original {original_size[0]}×{original_size[1]}, resized to {width}×{height})")
    print(f"Model      : {args.model}")
    print(f"Device     : {actual_device}")
    print(f"Batch size : {args.batch_size}")
    print(f"Warmup     : {args.warmup} passes")
    print()

    pad = max(len(n) for n in class_names)
    bar_scale = 30
    print("Confidences")
    print("-" * (pad + bar_scale + 14))
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        bar = "█" * int(prob * bar_scale)
        marker = " ◀ top" if i == top_idx else ""
        print(f"  {name:<{pad}}  {bar:<{bar_scale}}  {prob * 100:6.2f}%{marker}")

    print()
    print(f"Prediction : {class_names[top_idx]}  ({probs[top_idx] * 100:.2f}% confidence)")
    print()
    per_mean = infer_mean / args.batch_size
    per_min  = infer_min  / args.batch_size
    per_max  = infer_max  / args.batch_size

    print("Timing")
    print("-" * 52)
    print(f"  Model compile        : {load_time * 1000:7.2f} ms")
    print(f"  Preprocessing        : {preprocess_time * 1000:7.2f} ms")
    print(f"  Batch inference mean : {infer_mean:7.2f} ms  (batch={args.batch_size}, {args.runs} runs)")
    print(f"  Batch inference min  : {infer_min:7.2f} ms")
    print(f"  Batch inference max  : {infer_max:7.2f} ms")
    if args.batch_size > 1:
        print(f"  Per-image mean       : {per_mean:7.2f} ms")
        print(f"  Per-image min        : {per_min:7.2f} ms")
        print(f"  Per-image max        : {per_max:7.2f} ms")
        print(f"  Throughput (mean)    : {1000 / per_mean:7.1f} img/s")
    print()


if __name__ == "__main__":
    main()
