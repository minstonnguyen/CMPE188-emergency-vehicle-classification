from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_smoke_pipeline_runs() -> None:
    subprocess.run(
        [sys.executable, "scripts/smoke_pipeline.py"],
        cwd=ROOT,
        check=True,
    )
