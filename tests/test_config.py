from __future__ import annotations

from pathlib import Path

from emergency_vehicle_classifier.config import TrainingConfig, load_config


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
paths:
  data_dir: data/foo
  model_dir: models/foo
  output_dir: outputs/foo
training:
  image_size: 32
  batch_size: 2
  epochs: 3
  learning_rate: 0.01
  seed: 7
  num_workers: 0
  device: cpu
""".strip(),
        encoding="utf-8",
    )
    c = load_config(cfg_path)
    assert isinstance(c, TrainingConfig)
    assert c.data_dir == Path("data/foo")
    assert c.model_dir == Path("models/foo")
    assert c.output_dir == Path("outputs/foo")
    assert c.image_size == 32
    assert c.batch_size == 2
    assert c.epochs == 3
    assert c.learning_rate == 0.01
    assert c.seed == 7
    assert c.num_workers == 0
    assert c.device == "cpu"
    assert c.use_class_weights is False
    assert c.model == "small"
    assert c.augment_train is False
    assert c.weight_decay == 0.0
    assert c.label_smoothing == 0.0
    assert c.early_stopping_patience is None
