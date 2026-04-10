from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainingConfig


def _base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset directory '{path}' to exist. "
            "See data/README.md for the required train/val/test layout."
        )


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    test_dir = config.data_dir / "test"

    _require_dir(train_dir)
    _require_dir(val_dir)
    _require_dir(test_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=_base_transform(config.image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=_base_transform(config.image_size))
    test_dataset = datasets.ImageFolder(test_dir, transform=_base_transform(config.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader, train_dataset.classes


def load_image(image_path: str | Path, image_size: int):
    transform = _base_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)
