from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print("Training entrypoint placeholder")
    print(f"Project root: {project_root}")
    print("Next step: implement dataset loading and baseline training.")


if __name__ == "__main__":
    main()
