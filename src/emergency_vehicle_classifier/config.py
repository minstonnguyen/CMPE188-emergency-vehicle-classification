from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class TrainingConfig:
    data_dir: Path
    model_dir: Path
    output_dir: Path
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int
    num_workers: int
    device: str
    use_class_weights: bool = False
    model: str = "small"
    augment_train: bool = False
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    early_stopping_patience: int | None = None


def load_config(config_path: str | Path) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    paths = raw["paths"]
    training = raw["training"]

    return TrainingConfig(
        data_dir=Path(paths["data_dir"]),
        model_dir=Path(paths["model_dir"]),
        output_dir=Path(paths["output_dir"]),
        image_size=int(training["image_size"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(training["seed"]),
        num_workers=int(training["num_workers"]),
        device=str(training["device"]),
        use_class_weights=bool(training.get("use_class_weights", False)),
        model=str(training.get("model", "small")),
        augment_train=bool(training.get("augment_train", False)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        label_smoothing=float(training.get("label_smoothing", 0.0)),
        early_stopping_patience=(
            int(training["early_stopping_patience"])
            if training.get("early_stopping_patience") is not None
            and int(training["early_stopping_patience"]) > 0
            else None
        ),
    )
