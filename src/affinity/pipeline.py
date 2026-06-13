from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from affinity.features import build_embedding_features


def build_features(
    proteins: list[str],
    smiles_values: list[str],
    protein_embedding_path: str = "",
    molecule_embedding_path: str = "",
) -> tuple[np.ndarray, dict[str, object]]:
    return build_embedding_features(
        proteins,
        smiles_values,
        protein_embedding_path,
        molecule_embedding_path,
    )


def standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    scale = features.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return ((features - mean) / scale).astype(np.float32), mean, scale


def standardize_apply(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features - mean) / scale).astype(np.float32)


def save_metadata(path: str | Path, metadata: dict) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
