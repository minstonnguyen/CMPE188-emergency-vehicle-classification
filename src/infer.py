from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print("Inference entrypoint placeholder")
    print(f"Project root: {project_root}")
    print("Next step: load a trained model and run predictions on images or video.")


if __name__ == "__main__":
    main()
