from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    path: str = "/content/embedding_dataset"
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 0


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "/content/artifacts/affinity"


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ProjectConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
