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
- run single-image inference from a saved model

If time allows, the project may expand into multi-class classification for `police`, `ambulance`, and `fire_truck`, or frame-by-frame inference on recorded video.

## Current Implementation Progress

The repository already contains real initial implementation work, not just project scaffolding. Current progress includes:

- runnable PyTorch training pipeline in `src/`
- modular package for configuration, data loading, model definition, training engine, and inference
- baseline configuration in `configs/baseline.yaml`
- command-line training and inference entrypoints
- saved-output structure for checkpoints and metrics in `models/` and `outputs/`
- documentation in `docs/proposal.md` and `docs/architecture.md`

At this stage, the project can train a CNN on a folder-structured dataset, evaluate on validation and test splits, save the best model checkpoint, write metrics to JSON, and perform single-image inference.

## Repository Structure

- `configs/` training configuration
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

## Train

```bash
python src/train.py --config configs/baseline.yaml
```

Outputs:

- `models/best_model.pt`
- `outputs/training_history.json`
- `outputs/test_metrics.json`

## Infer

```bash
python src/infer.py --image path/to/example.jpg --checkpoint models/best_model.pt
```
