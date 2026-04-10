# Emergency Vehicle Classification

Minimal but real ML MVP for classifying emergency vehicles from images.

The code is intentionally small:
- one package for config, data loading, model, training, and inference
- one YAML config file
- one standard dataset layout
- one checkpoint plus JSON metrics output

## What This MVP Does

- Trains a CNN on image folders using PyTorch
- Validates during training and saves the best checkpoint
- Evaluates on a held-out test split
- Writes report-ready metrics to `outputs/test_metrics.json`
- Runs single-image inference from a saved checkpoint

## Expected Dataset Layout

Store images like this:

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

You can later expand this to more classes such as `police`, `ambulance`, and `fire_truck`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --config configs/baseline.yaml
```

Outputs:
- `models/best_model.pt`
- `outputs/training_history.json`
- `outputs/test_metrics.json`

Metrics include:
- accuracy
- weighted precision
- weighted recall
- weighted F1
- confusion matrix

## Infer

```bash
python src/infer.py --image path/to/example.jpg --checkpoint models/best_model.pt
```

## Repository Structure

- `configs/` training configuration
- `data/` dataset instructions and local storage layout
- `docs/` proposal and course writeups
- `models/` trained checkpoints
- `notebooks/` optional experiments
- `outputs/` metrics and predictions
- `src/` runnable ML code

## Production-Like Choices Kept Simple

- Config file instead of hard-coded paths
- Dedicated package instead of one giant script
- Checkpointing the best validation model
- Repeatable metrics written to disk
- Clear CLI entrypoints for train and inference
