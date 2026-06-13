from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SMILES_TOKENS = (
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "=", "#", "(", ")", "[", "]", "+", "-", "@",
)
PROTEIN_GROUPS = {
    "hydrophobic": set("AVILMFWY"),
    "polar": set("STNQ"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "special": set("CGP"),
}


def protein_descriptors(sequence: str) -> np.ndarray:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Protein sequence is empty")
    counts = Counter(sequence)
    length = len(sequence)
    composition = [counts[aa] / length for aa in AMINO_ACIDS]
    groups = [sum(counts[aa] for aa in members) / length for members in PROTEIN_GROUPS.values()]
    length_features = [np.log1p(length) / 10.0, min(length, 4096) / 4096.0]
    return np.asarray(composition + groups + length_features, dtype=np.float32)


def smiles_descriptors(smiles: str) -> np.ndarray:
    smiles = smiles.strip()
    if not smiles:
        raise ValueError("SMILES is empty")
    length = len(smiles)
    token_frequencies = [smiles.count(token) / length for token in SMILES_TOKENS]
    structural = [
        np.log1p(length) / 10.0,
        min(length, 512) / 512.0,
        sum(char.isdigit() for char in smiles) / length,
        sum(char.islower() for char in smiles) / length,
        smiles.count(".") / length,
    ]
    return np.asarray(token_frequencies + structural, dtype=np.float32)


def descriptor_matrix(proteins: Iterable[str], smiles_values: Iterable[str]) -> np.ndarray:
    rows = [
        np.concatenate([protein_descriptors(protein), smiles_descriptors(smiles)])
        for protein, smiles in zip(proteins, smiles_values, strict=True)
    ]
    return np.stack(rows).astype(np.float32)


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
    table = np.load(path, allow_pickle=False)
    mapping = dict(zip(table["keys"].tolist(), table["embeddings"], strict=True))
    model_id = str(table["model_id"].item())
    settings = (
        json.loads(str(table["settings_json"].item()))
        if "settings_json" in table.files
        else {}
    )
    return mapping, model_id, settings


def join_optional_embeddings(
    base_features: np.ndarray,
    proteins: Iterable[str],
    smiles_values: Iterable[str],
    protein_path: str = "",
    molecule_path: str = "",
) -> tuple[np.ndarray, dict[str, object]]:
    if bool(protein_path) != bool(molecule_path):
        raise ValueError(
            "Protein and molecule embedding tables must be provided together for LLM fusion"
        )
    blocks = [] if protein_path else [base_features]
    metadata: dict[str, object] = {
        "feature_mode": "llm_embeddings" if protein_path else "descriptors"
    }
    if protein_path:
        table, model_id, settings = load_embedding_table(protein_path)
        try:
            blocks.append(np.stack([table[value] for value in proteins]).astype(np.float32))
        except KeyError as error:
            raise KeyError(f"Missing protein embedding for sequence: {str(error)[:80]}") from error
        metadata["protein_model"] = model_id
        metadata["protein_extraction"] = settings
    if molecule_path:
        table, model_id, settings = load_embedding_table(molecule_path)
        try:
            blocks.append(np.stack([table[value] for value in smiles_values]).astype(np.float32))
        except KeyError as error:
            raise KeyError(f"Missing molecule embedding for SMILES: {error}") from error
        metadata["molecule_model"] = model_id
        metadata["molecule_extraction"] = settings
    return np.concatenate(blocks, axis=1), metadata
