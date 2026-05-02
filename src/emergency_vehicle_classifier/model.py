from __future__ import annotations

import torch


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    key = (name or "small").lower().strip()
    if key == "medium":
        return MediumCNN(num_classes)
    if key == "small":
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown model name {name!r}; use 'small' or 'medium'.")


class SmallCNN(torch.nn.Module):
    """Small baseline model intended for coursework-sized experiments."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


class MediumCNN(torch.nn.Module):
    """Deeper / wider CNN for harder splits; still lightweight for CPU training."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.55),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))
