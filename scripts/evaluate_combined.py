"""Evaluate a model checkpoint on the combined val+test set.

Run from the project root:
    python scripts/evaluate_combined.py models/best_model.pt
    python scripts/evaluate_combined.py models/best_model.pt --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from src/ when run directly (without pip install -e .)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

from emergency_vehicle_classifier.config import load_config
from emergency_vehicle_classifier.data import _base_transform
from emergency_vehicle_classifier.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on the combined val+test set from a checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="YAML config (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    return all_labels, all_preds, avg_loss


def print_results(
    labels: list[int],
    preds: list[int],
    loss: float,
    class_names: list[str],
) -> None:
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    col_w = 26
    print()
    print("Val+Test Combined Results")
    print("-" * 46)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 46)
    print(f"{'Accuracy':<30} {acc * 100:>9.2f}%")
    print(f"{'Precision (weighted)':<30} {precision * 100:>9.2f}%")
    print(f"{'Recall (weighted)':<30} {recall * 100:>9.2f}%")
    print(f"{'F1 (weighted)':<30} {f1 * 100:>9.2f}%")
    print(f"{'Loss':<30} {loss:>10.4f}")
    print()

    print("Confusion Matrix")
    header_cells = [""] + [f"Predicted: {n}" for n in class_names]
    print("  ".join(f"{c:<{col_w}}" for c in header_cells))
    for i, row_name in enumerate(class_names):
        cells = [f"Actual: {row_name}"]
        for j, count in enumerate(cm[i]):
            mark = "✓" if i == j else "✗"
            cells.append(f"{count} {mark}")
        print("  ".join(f"{c:<{col_w}}" for c in cells))
    print()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    config = load_config(args.config)
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size

    # Load checkpoint metadata
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_name = str(ckpt.get("model_name", config.model))
    class_names: list[str] = ckpt.get("class_names", [])
    image_size: int = int(ckpt.get("image_size", config.image_size))

    device_name = config.device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : {model_name}")
    print(f"Classes    : {class_names}")
    print(f"Image size : {image_size}")
    print(f"Device     : {device}")

    model = build_model(model_name, len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    tf = _base_transform(image_size)
    val_dir = config.data_dir / "val"
    test_dir = config.data_dir / "test"

    for d in (val_dir, test_dir):
        if not d.exists():
            sys.exit(f"[ERROR] Directory not found: {d}")

    val_dataset = datasets.ImageFolder(val_dir, transform=tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=tf)
    combined = ConcatDataset([val_dataset, test_dataset])

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    total = len(combined)
    val_count = len(val_dataset)
    test_count = len(test_dataset)
    print(f"Val images : {val_count}")
    print(f"Test images: {test_count}")
    print(f"Total      : {total}")

    loss_fn = torch.nn.CrossEntropyLoss()

    labels, preds, loss = evaluate(model, loader, loss_fn, device)
    print_results(labels, preds, loss, class_names)


if __name__ == "__main__":
    main()
