from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()

    import torch

    from .data import load_image
    from .model import SmallCNN

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    class_names: list[str] = checkpoint["class_names"]
    image_size: int = checkpoint["image_size"]

    model = SmallCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_image(args.image, image_size).to(device)
    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prediction_idx = int(torch.argmax(probabilities).item())

    print(
        {
            "predicted_class": class_names[prediction_idx],
            "confidence": round(float(probabilities[prediction_idx].item()), 4),
            "scores": {
                class_name: round(float(score), 4)
                for class_name, score in zip(class_names, probabilities.tolist())
            },
        }
    )


if __name__ == "__main__":
    main()
