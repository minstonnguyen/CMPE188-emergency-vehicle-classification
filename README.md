# Emergency Vehicle Classification

## Team Members

- Evan Alekseyev
- Minston Nguyen
- Sonny Au

## Problem Statement

This project builds a computer vision system that identifies emergency or law-enforcement vehicles from images. The practical goal is to help warn drivers when these vehicles are present so they can respond more carefully and appropriately. For the current milestone, the scope is image classification, with possible expansion to video-based inference later.

## Dataset / Data Source

We plan to use public image datasets and curated vehicle image collections, including:

- Roboflow emergency and police vehicle datasets: https://universe.roboflow.com/search?q=class%3Apolice+car
- Police Car Image Classification Dataset on Images.cv: https://images.cv/dataset/police-car-image-classification-dataset

The training layout in this repository assumes processed images are organized into `train`, `val`, and `test` folders under `data/processed/`, with class folders such as `emergency_vehicle` and `non_emergency`.

## Planned Model / System Approach

The current system uses a PyTorch-based convolutional neural network for binary image classification. The planned workflow is:

- collect and clean labeled vehicle images from public sources
- standardize labels across datasets
- train a CNN baseline on image folders
- validate during training and save the best checkpoint
- evaluate on a held-out test set using accuracy, precision, recall, F1, and confusion matrix
- run single-image or batch (folder) inference from a saved model, optionally writing JSON or CSV

If time allows, the project may expand into multi-class classification for `police`, `ambulance`, and `fire_truck`, or frame-by-frame inference on recorded video. A longer-term **three-class law-enforcement** direction (civilian / overt LE / covert LE) is outlined in `ARCHITECTURE_PROPOSAL.md` and summarized under **Roadmap** below.

## Current Implementation Progress

The repository already contains real initial implementation work, not just project scaffolding. Current progress includes:

- runnable PyTorch training pipeline in `src/`
- modular package for configuration, data loading, model definition, training engine, and inference
- baseline configuration in `configs/baseline.yaml`
- command-line training and inference entrypoints
- saved-output structure for checkpoints and metrics in `models/` and `outputs/`
- documentation in `docs/proposal.md` and `docs/architecture.md`

At this stage, the project can train a CNN on a folder-structured dataset, evaluate on validation and test splits, save the best model checkpoint, write metrics to JSON, run **single-image or directory batch** inference, and verify the full path with an automated **smoke test** (`scripts/smoke_pipeline.py` or `pytest`).

## Repository Structure

- `configs/` training configuration (see `baseline.yaml` for real data; `smoke.yaml` for CI/dev checks)
- `scripts/` one-shot automation (e.g. smoke pipeline)
- `tests/` quality gate (pytest)
- `data/` dataset instructions and local storage layout
- `docs/` proposal and architecture writeups
- `models/` trained checkpoints
- `notebooks/` optional experiments
- `outputs/` metrics and predictions
- `src/` runnable ML code

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Expected Dataset Layout

```text
data/processed/
  train/
    emergency_vehicle/
    non_emergency/
  val/
    emergency_vehicle/
    non_emergency/
  test/
    emergency_vehicle/
    non_emergency/
```

This structure can later be expanded to more classes if the team moves beyond the binary baseline.

### Quick path: inbox → processed

1. Copy photos into `data/incoming/emergency_vehicle/` and `data/incoming/non_emergency/` (at least two images each).
2. Run `python src/prepare_data.py` — it builds `data/processed/train|val|test/...` and removes synthetic `data/smoke_processed/` by default.
3. Train: `python src/train.py --config configs/baseline.yaml`

See `data/incoming/README.md` for options (`--dry-run`, `--keep-smoke`).

Default `configs/baseline.yaml` uses **`device: cpu`** and a moderate `batch_size` so training runs on typical laptops. If you have CUDA, set `device` to `cuda` or `auto`.

## Train

```bash
python src/train.py --config configs/baseline.yaml
```

Outputs:

- `models/best_model.pt`
- `outputs/training_history.json`
- `outputs/test_metrics.json`

## Infer

Single image (prints JSON):

```bash
python src/infer.py --image path/to/example.jpg --checkpoint models/best_model.pt
```

All images in a **flat** folder (non-recursive), written to JSON or CSV:

```bash
python src/infer.py --image-dir path/to/images --checkpoint models/best_model.pt --output outputs/preds.json
python src/infer.py --image-dir path/to/images --checkpoint models/best_model.pt --output outputs/preds.csv
```

## Smoke test (quality gate)

End-to-end check using **synthetic** data under `data/smoke_processed/`, checkpoints under `models/smoke/`, and metrics under `outputs/smoke/`. These paths are gitignored and do not replace your real `data/processed/` tree.

```bash
python scripts/smoke_pipeline.py
```

Or:

```bash
pytest tests/test_smoke_pipeline.py -v --timeout=600
```

## Roadmap: binary milestone → three-class proposal

The **current milestone** is **binary** classification (`emergency_vehicle` vs `non_emergency`) using a **CNN** (`SmallCNN`) and the folder layout above. That matches an SDLC-style ML pipeline: **config → prepare data → train → evaluate → checkpoint → infer**, with JSON metrics for reproducibility.

**Next step** toward `ARCHITECTURE_PROPOSAL.md` is mostly **problem formulation and data**, not just code:

1. **Classes** — map datasets to **civilian / overt LE / covert LE** (new folder names or a label table).
2. **Model** — keep the same pipeline shell; swap the backbone (e.g. DINOv2 + register tokens) or add a **second stage** as in the proposal’s two-model design.
3. **Config** — add a YAML next to `baseline.yaml` for the new class count and hyperparameters.

More detail lives in `ARCHITECTURE_PROPOSAL.md` and `docs/architecture.md`.
