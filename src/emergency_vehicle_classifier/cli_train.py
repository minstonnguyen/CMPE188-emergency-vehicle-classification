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
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT.pt",
        help="Load model weights from this checkpoint before training (e.g. models/best_model.pt).",
    )
    parser.add_argument(
        "--until-val-accuracy",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Keep training until validation accuracy reaches this value (e.g. 0.7). Use with --max-epochs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs when using --until-val-accuracy (default 500). Ignored otherwise.",
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
    from .model import build_model

    config = load_config(args.config)
    set_seed(config.seed)

    device = resolve_device(config.device)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(config)

    model = build_model(config.model, len(class_names)).to(device)
    checkpoint_path = Path(config.model_dir) / "best_model.pt"

    if args.resume:
        resume_path = Path(args.resume)
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        saved_name = str(ckpt.get("model_name", "small"))
        if saved_name != config.model:
            raise SystemExit(
                f"Checkpoint model_name={saved_name!r} does not match config model={config.model!r}. "
                "Use the same architecture or omit --resume."
            )
        model.load_state_dict(ckpt["model_state_dict"])

    class_weights = None
    if config.use_class_weights:
        num_classes = len(class_names)
        counts = torch.zeros(num_classes, dtype=torch.float64)
        for _, class_idx in train_loader.dataset.samples:
            counts[int(class_idx)] += 1.0
        # n_samples / (n_classes * count_k); avoids div-by-zero on empty class
        class_weights = (len(train_loader.dataset) / (num_classes * counts.clamp(min=1.0))).float()

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    epochs_without_val_loss_improve = 0
    history: list[dict[str, float]] = []

    target_val = args.until_val_accuracy
    if target_val is not None and not (0.0 < target_val <= 1.0):
        raise SystemExit("--until-val-accuracy must be between 0 and 1 (e.g. 0.7).")

    if target_val is not None:
        max_epochs = args.max_epochs if args.max_epochs is not None else 500
    else:
        max_epochs = config.epochs

    history_meta: dict = {}
    if target_val is not None:
        history_meta["until_val_accuracy"] = target_val

    target_reached = False
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(float(val_metrics["loss"]))

        val_acc = float(val_metrics["accuracy"])
        val_loss = float(val_metrics["loss"])
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        print(epoch_summary, flush=True)
        if target_val is not None:
            save_json(
                Path(config.output_dir) / "training_history.json",
                {**history_meta, "history": history},
            )

        # Best checkpoint by validation loss (lower is better) to limit overfitting to the train set.
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            epochs_without_val_loss_improve = 0
            save_checkpoint(
                checkpoint_path,
                model,
                class_names,
                config.image_size,
                model_name=config.model,
            )
        else:
            epochs_without_val_loss_improve += 1

        if (
            config.early_stopping_patience is not None
            and epochs_without_val_loss_improve >= config.early_stopping_patience
        ):
            print(
                {
                    "early_stopping": True,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "epochs_without_val_loss_improve": epochs_without_val_loss_improve,
                },
                flush=True,
            )
            break

        if target_val is not None and val_acc >= target_val:
            target_reached = True
            print(
                {
                    "target_val_accuracy_reached": True,
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                },
                flush=True,
            )
            break

    if target_val is not None and not target_reached:
        print(
            {
                "target_val_accuracy_reached": False,
                "best_val_loss": best_val_loss,
                "max_epochs": max_epochs,
                "message": (
                    f"Validation accuracy did not reach {target_val:.0%} within {max_epochs} epochs "
                    f"(best val loss: {best_val_loss:.6f}). Try more data, tuning, or a higher --max-epochs."
                ),
            },
            flush=True,
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    history_payload: dict = {
        "history": history,
        "best_val_loss": best_val_loss,
    }
    if target_val is not None:
        history_payload["until_val_accuracy"] = target_val
        history_payload["target_val_accuracy_reached"] = target_reached
    save_json(Path(config.output_dir) / "training_history.json", history_payload)
    save_json(
        Path(config.output_dir) / "test_metrics.json",
        {
            "class_names": class_names,
            "metrics": test_metrics,
        },
    )
    print({"checkpoint": str(checkpoint_path), "test_metrics": test_metrics}, flush=True)


if __name__ == "__main__":
    main()
