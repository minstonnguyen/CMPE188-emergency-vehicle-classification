"""
End-to-end smoke test: synthetic data -> train -> infer.
Uses data/smoke_processed and models/smoke (gitignored) so local data/processed is untouched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    display = " ".join(cmd)
    print("+", display, flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    run(
        [
            sys.executable,
            "src/generate_sample_dataset.py",
            "--output-dir",
            "data/smoke_processed",
            "--quick",
            "--image-size",
            "64",
        ]
    )
    run([sys.executable, "src/train.py", "--config", "configs/smoke.yaml"])
    sample = ROOT / "data/smoke_processed/test/emergency_vehicle/emergency_vehicle_000.png"
    if not sample.is_file():
        raise FileNotFoundError(f"Expected smoke image at {sample}")
    run(
        [
            sys.executable,
            "src/infer.py",
            "--image",
            str(sample),
            "--checkpoint",
            "models/smoke/best_model.pt",
            "--device",
            "cpu",
        ]
    )
    run(
        [
            sys.executable,
            "src/infer.py",
            "--image-dir",
            str(sample.parent),
            "--checkpoint",
            "models/smoke/best_model.pt",
            "--device",
            "cpu",
            "--output",
            "outputs/smoke/batch_predictions.json",
        ]
    )
    print("Smoke pipeline OK.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
