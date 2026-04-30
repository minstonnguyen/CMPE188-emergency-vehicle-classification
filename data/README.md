# Data Layout

Place downloaded datasets here.

- `incoming/` **drop zone**: put emergency vs non-emergency images in the two class folders, then run `python src/prepare_data.py` (see `incoming/README.md`)
- `raw/` original downloads from Roboflow, Images.cv, or other public sources
- `processed/` final train/val/test folder structure consumed by training (filled by `prepare_data.py` or `split_dataset.py`)

Expected structure for the current training pipeline:

```text
processed/
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

Notes:

- Folder names become class labels automatically.
- All images should be ordinary RGB files such as `.jpg` or `.png`.
- Do not commit large datasets to git.
