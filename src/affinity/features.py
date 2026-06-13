from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def save_embedding_table(
    path: str | Path,
    keys: list[str],
    embeddings: np.ndarray,
    model_id: str,
    settings: dict | None = None,
) -> None:
    if len(keys) != len(embeddings):
        raise ValueError("Keys and embeddings must have the same row count")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        keys=np.asarray(keys, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_id=np.asarray(model_id),
        settings_json=np.asarray(json.dumps(settings or {})),
    )


def load_embedding_table(path: str | Path) -> tuple[dict[str, np.ndarray], str, dict]:
    source = Path(path)
    files = sorted(source.glob("*.npz")) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f"No embedding files found at {source}")
    mapping: dict[str, np.ndarray] = {}
    model_id = ""
    settings: dict = {}
    for file in files:
        table = np.load(file, allow_pickle=False)
        current_model = str(table["model_id"].item())
        current_settings = (
            json.loads(str(table["settings_json"].item()))
            if "settings_json" in table.files
            else {}
        )
        if model_id and current_model != model_id:
            raise ValueError(f"Embedding model mismatch in {file}")
        if settings and current_settings != settings:
            raise ValueError(f"Embedding settings mismatch in {file}")
        model_id = current_model
        settings = current_settings
        mapping.update(zip(table["keys"].tolist(), table["embeddings"], strict=True))
    return mapping, model_id, settings


def build_embedding_features(
    proteins: Iterable[str],
    smiles_values: Iterable[str],
    protein_path: str | Path,
    molecule_path: str | Path,
) -> tuple[np.ndarray, dict[str, object]]:
    if not protein_path or not molecule_path:
        raise ValueError("Both ONNX embedding tables are required")
    protein_table, protein_model, protein_settings = load_embedding_table(protein_path)
    molecule_table, molecule_model, molecule_settings = load_embedding_table(molecule_path)
    try:
        protein_block = np.stack([protein_table[value] for value in proteins])
    except KeyError as error:
        raise KeyError(f"Missing protein embedding: {str(error)[:80]}") from error
    try:
        molecule_block = np.stack([molecule_table[value] for value in smiles_values])
    except KeyError as error:
        raise KeyError(f"Missing molecule embedding: {error}") from error
    return (
        np.concatenate([protein_block, molecule_block], axis=1).astype(np.float32),
        {
            "feature_mode": "onnx_embeddings",
            "protein_model": protein_model,
            "protein_extraction": protein_settings,
            "molecule_model": molecule_model,
            "molecule_extraction": molecule_settings,
        },
    )
