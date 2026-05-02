from __future__ import annotations

import json
from pathlib import Path

import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    model_name: str = "small",
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "model_name": model_name,
        },
        path,
    )
