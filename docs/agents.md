# Agent onboarding (CMPE188 — Emergency Vehicle Classification)

This file is written for **AI coding agents** and human developers resuming work. Read it **first**, then drill into linked documents only when changing behavior or data semantics.

---

## Minimal context (always read before editing)

| Order | Path | Why |
|------:|------|-----|
| 1 | [README.md](../README.md) | Setup, **end-to-end commands**, dataset layout expectations, roadmap |
| 2 | [data/incoming/README.md](../data/incoming/README.md) | **Inbox rules**: class folder names, supported extensions, split behavior |
| 3 | [docs/agents.md](agents.md) | This handoff — workflow, pitfalls, where code lives |

**Optional but high value**

| Path | Why |
|------|-----|
| [docs/architecture.md](architecture.md) | Pipeline diagrams, component map (`data.py`, `engine.py`, CLI) |
| [ARCHITECTURE_PROPOSAL.md](../ARCHITECTURE_PROPOSAL.md) | Future **three-class** (civilian / overt LE / covert LE) direction |
| [docs/proposal.md](proposal.md) | Course framing, datasets, milestones |

---

## What this project does

- **Task:** Binary image classification — `emergency_vehicle` vs `non_emergency`.
- **Stack:** PyTorch, `torchvision.datasets.ImageFolder`, small CNN (`SmallCNN`), YAML config.
- **Entrypoints:** `src/prepare_data.py`, `src/train.py`, `src/infer.py` (thin wrappers calling `emergency_vehicle_classifier.cli_*`).
- **Package root for imports:** run CLIs from **repository root**, or set `PYTHONPATH=src` if invoking modules manually.

---

## Data workflow (source of truth)

1. Users place labeled images **only** in:
   - `data/incoming/emergency_vehicle/`
   - `data/incoming/non_emergency/`  
   **Flat directories** — nested subfolders are not used by the splitter. Extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
2. `python src/prepare_data.py` builds **`data/processed/{train,val,test}/<class>/`** (~70 / 15 / 15 per class by default).
3. **By default** it removes `data/smoke_processed/` (synthetic CI data). Preserve it with `--keep-smoke` if needed.
4. Training reads **`configs/baseline.yaml`** → `paths.data_dir` (normally `data/processed`).

**Agent note:** Filename suffix must match supported extensions (`Path.suffix`), or images are invisible to `dataset_layout.list_images()`.

---

## Commands (from repo root)

```bash
python src/prepare_data.py --dry-run   # counts / plan only
python src/prepare_data.py             # rebuild processed splits
python src/train.py --config configs/baseline.yaml
python src/infer.py --image path/to/img.jpg --checkpoint models/best_model.pt
python scripts/smoke_pipeline.py       # synthetic E2E; does not replace real processed data
pytest -q
```

Default `baseline.yaml` uses **`device: cpu`** and **`num_workers: 0`** for portability. Use `cuda` or `auto` when appropriate on the runner’s machine.

---

## Repository map (where to change behavior)

| Area | Location |
|------|------------|
| Training hyperparameters / paths | `configs/baseline.yaml`, `configs/smoke.yaml` |
| Config parsing | `src/emergency_vehicle_classifier/config.py` |
| Loaders, transforms | `src/emergency_vehicle_classifier/data.py` |
| Model | `src/emergency_vehicle_classifier/model.py` |
| Train / eval loops, metrics | `src/emergency_vehicle_classifier/engine.py` |
| Incoming → processed split | `src/emergency_vehicle_classifier/cli_prepare_incoming.py`, `dataset_split.py`, `dataset_layout.py` |
| Inference CLI | `src/emergency_vehicle_classifier/cli_infer.py` |
| Automated pipeline check | `scripts/smoke_pipeline.py`, `tests/test_smoke_pipeline.py` |

Artifacts usually **not** in git: **`models/`**, **`outputs/`**, and **`data/processed/`** (rebuilt from inbox). **`data/incoming/`** labeled images **are tracked** for team visibility; keep sizes reasonable for the repo host.

---

## Professional expectations for agents

- **Match existing style:** dataclasses, type hints, small focused changes; avoid drive-by refactors.
- **Run from project root** when executing scripts so relative paths (`data/...`, `configs/...`) resolve.
- **Verify with data:** after label or preprocessing changes, re-run `prepare_data.py` then `train.py`; inspect **`outputs/test_metrics.json`** — especially **`confusion_matrix`** — not headline accuracy alone.
- **Smoke vs real:** `smoke.yaml` proves wiring; **`baseline.yaml` + real `data/processed`** is what matters for the course deliverable.
- **Do not assume** CUDA; ask or read `baseline.yaml` `device` when suggesting train commands.

---

## Known pitfalls (resume here if metrics look wrong)

1. **Majority-class collapse.** With **class imbalance**, a short baseline run may predict almost only the dominant class — high accuracy can still mislead. Always read the confusion matrix on the held-out **test** set.
2. **Label quality dominates.** Wrong images in `emergency_vehicle` vs `non_emergency` undermine everything; spot-check folders before tuning hyperparameters.
3. **Imbalance vs data volume.** Roughly balanced classes, more minority-class examples, weighted loss, or balanced sampling are the usual fixes — implement in line with maintainers’ preference and existing `engine.py` structure.
4. **Inference scope.** Batch inference scans a **single directory level** (non-recursive); see `README.md` / `cli_infer.py`.

---

## Course / team context

Team (from README): Evan Alekseyev, Minston Nguyen, Sonny Au. Dataset sources cited in README and proposal include **Roboflow** and **Images.cv**; document provenance when adding external data.

---

## Changelog hints for agents

When you materially change workflow or metrics behavior, update **README.md** (user-facing commands) and, if architecture shifts, **docs/architecture.md**. Keep **`docs/agents.md`** updated only when onboarding facts, paths, or known pitfalls change.
