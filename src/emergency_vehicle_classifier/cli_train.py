from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the emergency vehicle classifier.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()

    import torch

    from .config import load_config
    from .data import build_dataloaders
    from .engine import evaluate, train_one_epoch
    from .io_utils import save_checkpoint, save_json
    from .model import SmallCNN

    config = load_config(args.config)
    set_seed(config.seed)

    device = resolve_device(config.device)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(config)

    model = SmallCNN(num_classes=len(class_names)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = Path(config.model_dir) / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(epoch_summary)
        print(epoch_summary)

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            save_checkpoint(checkpoint_path, model, class_names, config.image_size)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    save_json(Path(config.output_dir) / "training_history.json", {"history": history})
    save_json(
        Path(config.output_dir) / "test_metrics.json",
        {
            "class_names": class_names,
            "metrics": test_metrics,
        },
    )
    print({"checkpoint": str(checkpoint_path), "test_metrics": test_metrics})


if __name__ == "__main__":
    main()
