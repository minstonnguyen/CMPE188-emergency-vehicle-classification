"""CLIP-based dataset cleaner using DegiRum image and text encoders.

For each image in the emergency-vehicle directory, CLIP cosine similarity is
computed against two text prompts.  If the image scores higher for the
non-emergency prompt it is considered mislabelled and excluded from the output.
The same logic applies in reverse for the non-emergency directory.

The two input directories are never modified.  A clean copy is written to a
mirrored output location (sibling of the common parent, suffixed "_clip_cleaned"),
preserving the original subdirectory names.

Requires
--------
    pip install transformers degirum

The CLIP tokenizer vocabulary is downloaded from HuggingFace on first use
and cached locally (~4 MB).

Usage (from project root)
--------------------------
    python scripts/clip_clean_dataset.py \\
        data/normalized/emergency_vehicle \\
        data/normalized/non_emergency

    # stricter: correct-class similarity must beat wrong-class by >= 0.05
    python scripts/clip_clean_dataset.py \\
        data/normalized/emergency_vehicle \\
        data/normalized/non_emergency \\
        --margin 0.05

    # explicit output parent dir
    python scripts/clip_clean_dataset.py \\
        data/normalized/emergency_vehicle \\
        data/normalized/non_emergency \\
        --output data/clip_cleaned
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import openvino
import numpy as np
import degirum as dg

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EMERGENCY_PROMPT = (
    "a photo of an emergency vehicle such as a police car, ambulance, or fire truck "
    "with emergency lights, sirens, or official law-enforcement markings"
)
NON_EMERGENCY_PROMPT = (
    "a photo of an ordinary civilian passenger car or non-emergency vehicle "
    "on a road, street, or parking lot"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"
}


# ---------------------------------------------------------------------------
# Tokenisation (CLIP BPE via HuggingFace transformers)
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        sys.exit(
            "[ERROR] The 'transformers' package is required.\n"
            "        Install it with:  pip install transformers"
        )
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def tokenize(tokenizer, text: str) -> np.ndarray:
    """Return (1, 77) int64 token IDs as expected by the DegiRum text encoder."""
    out = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    return out["input_ids"].astype(np.int64)  # (1, 77) int64


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_embedding(result) -> np.ndarray:
    """Extract a flat float32 embedding from a DegiRum inference result.

    Handles several result layouts produced by different model versions.
    """
    raw = result.results
    if isinstance(raw, np.ndarray):
        return raw.flatten().astype(np.float32)
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, (list, np.ndarray)):
            return np.array(first, dtype=np.float32).flatten()
        if isinstance(first, dict):
            for key in ("data", "embedding", "features", "output", "vector"):
                if key in first:
                    return np.array(first[key], dtype=np.float32).flatten()
    return np.array(raw, dtype=np.float32).flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def collect_images(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Per-directory processing
# ---------------------------------------------------------------------------

def process_directory(
    src_dir: Path,
    out_dir: Path,
    correct_emb: np.ndarray,
    wrong_emb: np.ndarray,
    image_model,
    margin: float,
    label: str,
) -> tuple[int, int, int]:
    """Copy images whose CLIP similarity favours *correct_emb* over *wrong_emb*.

    Returns (total, kept, removed).
    """
    images = collect_images(src_dir)
    total = len(images)
    kept = 0
    removed = 0

    for src in images:
        rel = src.relative_to(src_dir)
        dst = out_dir / rel.parent / (src.stem + src.suffix)
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            img_emb = get_embedding(image_model(str(src)))
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] {label}/{rel}: inference failed ({exc}) — keeping.")
            shutil.copy2(src, dst)
            kept += 1
            continue

        sim_correct = cosine_sim(img_emb, correct_emb)
        sim_wrong   = cosine_sim(img_emb, wrong_emb)

        if sim_correct >= sim_wrong + margin:
            shutil.copy2(src, dst)
            kept += 1
        else:
            removed += 1
            print(
                f"  [REMOVE] {label}/{rel}  "
                f"correct={sim_correct:.4f}  wrong={sim_wrong:.4f}"
            )

    return total, kept, removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use CLIP to remove mislabelled images from emergency-vehicle and "
            "non-emergency-vehicle dataset directories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "emergency_dir",
        type=Path,
        help="Directory containing emergency-vehicle images.",
    )
    parser.add_argument(
        "non_emergency_dir",
        type=Path,
        help="Directory containing non-emergency / civilian-vehicle images.",
    )
    parser.add_argument(
        "--host",
        default="@local",
        metavar="HOST",
        help="DegiRum inference host (@local, @cloud, or AI-server IP).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        metavar="DELTA",
        help=(
            "Cosine-similarity margin the correct prompt must beat the wrong prompt by. "
            "0.0 = keep whenever correct >= wrong."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Parent output directory. Each input dir is reproduced inside it "
            "under its original name. "
            "Defaults to a sibling of the common parent named '<parent>_clip_cleaned'."
        ),
    )
    args = parser.parse_args()

    emg_dir = args.emergency_dir.resolve()
    non_dir = args.non_emergency_dir.resolve()

    for d in (emg_dir, non_dir):
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")

    # Determine output root
    if args.output is not None:
        out_parent = args.output.resolve()
    else:
        try:
            common = Path(os.path.commonpath([emg_dir, non_dir]))
        except ValueError:
            common = emg_dir.parent
        out_parent = common.parent / (common.name + "_clip_cleaned")

    out_emg = out_parent / emg_dir.name
    out_non = out_parent / non_dir.name

    # ------------------------------------------------------------------
    # Tokenizer + text embeddings (computed once)
    # ------------------------------------------------------------------
    print("Loading CLIP tokenizer (downloads ~4 MB on first run) ...")
    tokenizer = _load_tokenizer()

    token = os.environ.get("DEGIRUM_CLOUD_TOKEN", "")

    print("Loading DegiRum CLIP text encoder ...")
    text_model = dg.load_model(
        model_name="clip_rn50_text_encoder--1x77_float_openvino_cpu_1",
        inference_host_address=args.host,
        zoo_url="degirum/public",
        token=token,
    )

    print("Encoding text prompts ...")
    emg_text_emb = get_embedding(text_model(tokenize(tokenizer, EMERGENCY_PROMPT)))
    non_text_emb = get_embedding(text_model(tokenize(tokenizer, NON_EMERGENCY_PROMPT)))
    print(f"  Emergency     : {emg_text_emb.shape}  norm={np.linalg.norm(emg_text_emb):.4f}")
    print(f"  Non-emergency : {non_text_emb.shape}  norm={np.linalg.norm(non_text_emb):.4f}")

    # ------------------------------------------------------------------
    # Image encoder
    # ------------------------------------------------------------------
    print("\nLoading DegiRum CLIP image encoder ...")
    image_model = dg.load_model(
        model_name="clip_rn50_image_encoder--224x224_float_openvino_cpu_1",
        inference_host_address=args.host,
        zoo_url="degirum/public",
        token=token,
        image_backend="pil",
    )

    print(f"\nMargin  : {args.margin}")
    print(f"Output  : {out_parent}\n")

    # ------------------------------------------------------------------
    # Process both directories
    # ------------------------------------------------------------------
    print(f"=== Emergency vehicle directory: {emg_dir} ===")
    e_total, e_kept, e_removed = process_directory(
        emg_dir, out_emg,
        correct_emb=emg_text_emb,
        wrong_emb=non_text_emb,
        image_model=image_model,
        margin=args.margin,
        label=emg_dir.name,
    )

    print(f"\n=== Non-emergency directory: {non_dir} ===")
    n_total, n_kept, n_removed = process_directory(
        non_dir, out_non,
        correct_emb=non_text_emb,
        wrong_emb=emg_text_emb,
        image_model=image_model,
        margin=args.margin,
        label=non_dir.name,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(
        f"\n{'=' * 52}\n"
        f"Done.\n"
        f"  Emergency  : {e_kept}/{e_total} kept, {e_removed} removed\n"
        f"  Non-emerg. : {n_kept}/{n_total} kept, {n_removed} removed\n"
        f"  Output     : {out_parent}"
    )


if __name__ == "__main__":
    main()
