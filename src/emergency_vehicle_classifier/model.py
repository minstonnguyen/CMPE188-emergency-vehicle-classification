from __future__ import annotations

import torch
import torchvision.models as tv_models


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    key = (name or "resnet50").lower().strip()
    if key in ("resnet50", "medium", "small"):
        return ResNet50Classifier(num_classes)
    raise ValueError(f"Unknown model name {name!r}; use 'resnet50'.")


class ResNet50Classifier(torch.nn.Module):
    """ResNet-50 backbone with a simple feedforward classification head.

    The torchvision ResNet-50 is loaded with ImageNet-pretrained weights.
    Its original fully-connected layer is replaced with a two-layer MLP:

        2048  →  ReLU  →  Dropout(0.5)  →  512  →  ReLU  →  num_classes
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        # Remove the original classification head; keep everything up to the
        # global average pool so the backbone outputs a (B, 2048) feature vector.
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)       # (B, 2048, 1, 1)
        features = features.flatten(1)         # (B, 2048)
        return self.head(features)             # (B, num_classes)
