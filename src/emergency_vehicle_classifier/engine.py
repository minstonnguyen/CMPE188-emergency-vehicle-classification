from __future__ import annotations

from collections.abc import Iterable

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

    return {"loss": total_loss / max(total_examples, 1)}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, float | list[list[int]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
    }
