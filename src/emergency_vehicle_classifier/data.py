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


def _train_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.extend(
            [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            ]
        )
    ops.append(transforms.ToTensor())
    if augment:
        ops.append(
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.4, 2.5)),
        )
    return transforms.Compose(ops)


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset directory '{path}' to exist. "
            "See data/README.md for the required train/val/test layout."
        )


def build_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    test_dir = config.data_dir / "test"

    _require_dir(train_dir)
    _require_dir(val_dir)
    _require_dir(test_dir)

    train_tf = _train_transform(config.image_size, config.augment_train)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
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
