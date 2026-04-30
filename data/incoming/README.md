# Photo inbox

Put your **real** images here **before** training (at least **two** per class so we can build train/validation splits):

| Folder | Put these images here |
|--------|------------------------|
| `emergency_vehicle/` | Police, ambulance, fire truck, other emergency / obvious LE vehicles |
| `non_emergency/` | Normal civilian vehicles |

Supported file types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` (directly inside each folder, not in nested subfolders).

Then from the **project root** run:

```bash
python src/prepare_data.py
```

That script:

1. Deletes old `train/` / `val/` / `test/` under `data/processed/` and rebuilds them with a random **70% / 15% / 15%** split per class (for larger sets; very small folders use a simple train/val/(test) assignment).
2. Deletes **`data/smoke_processed/`** (synthetic smoke-test images) unless you pass `--keep-smoke`.

Dry run (see counts only):

```bash
python src/prepare_data.py --dry-run
```

**You still sort photos by hand** into the two folders above. The script does not know emergency vs civilian from one mixed pile— it only splits into train/val/test and refreshes `processed/`.

If your download used different folder names, rename them to **`emergency_vehicle`** and **`non_emergency`** before running.
