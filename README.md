# Emergency Vehicle Classification

Course project starter for a computer vision model that identifies law enforcement and emergency vehicles in images, with a later extension to video.

## Project Scope

Initial milestone:
- Binary image classification: `emergency_vehicle` vs `non_emergency_vehicle`

Possible extension:
- Multi-class classification: `police`, `ambulance`, `fire_truck`, `non_emergency`
- Video inference on recorded footage

## Repository Structure

- `docs/` proposal and project writeups
- `src/` training and inference code
- `notebooks/` experiments and EDA
- `data/` dataset instructions and local storage layout
- `models/` saved checkpoints
- `outputs/` predictions and demo artifacts

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Next Steps

1. Collect positive and negative examples from public datasets.
2. Standardize labels into a single CSV manifest.
3. Train a baseline CNN.
4. Evaluate precision, recall, and confusion matrix.
5. Run inference on unseen images and demo footage.
