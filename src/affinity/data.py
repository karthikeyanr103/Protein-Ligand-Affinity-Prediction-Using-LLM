from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("protein_sequence", "compound_smiles", "label")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTUVWY")


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    unique_proteins: int
    unique_compounds: int
    unique_pairs: int
    protein_length_min: int
    protein_length_max: int
    protein_length_mean: float
    smiles_length_min: int
    smiles_length_max: int
    smiles_length_mean: float
    label_min: float
    label_max: float
    label_mean: float
    label_std: float
    duplicate_pairs: int


def load_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=nrows)
    missing_columns = set(COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    frame = frame.loc[:, COLUMNS].copy()
    if frame.isna().any().any():
        counts = frame.isna().sum()
        raise ValueError(f"Missing values found: {counts[counts > 0].to_dict()}")
    frame["protein_sequence"] = frame["protein_sequence"].astype(str).str.strip().str.upper()
    frame["compound_smiles"] = frame["compound_smiles"].astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.float32)
    invalid = frame["protein_sequence"].map(lambda sequence: bool(set(sequence) - AMINO_ACIDS))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} protein sequences contain unsupported residues")
    return frame


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    protein_lengths = frame["protein_sequence"].str.len()
    smiles_lengths = frame["compound_smiles"].str.len()
    unique_pairs = frame.drop_duplicates(["protein_sequence", "compound_smiles"]).shape[0]
    return DatasetProfile(
        rows=len(frame),
        unique_proteins=frame["protein_sequence"].nunique(),
        unique_compounds=frame["compound_smiles"].nunique(),
        unique_pairs=unique_pairs,
        protein_length_min=int(protein_lengths.min()),
        protein_length_max=int(protein_lengths.max()),
        protein_length_mean=float(protein_lengths.mean()),
        smiles_length_min=int(smiles_lengths.min()),
        smiles_length_max=int(smiles_lengths.max()),
        smiles_length_mean=float(smiles_lengths.mean()),
        label_min=float(frame["label"].min()),
        label_max=float(frame["label"].max()),
        label_mean=float(frame["label"].mean()),
        label_std=float(frame["label"].std()),
        duplicate_pairs=len(frame) - unique_pairs,
    )


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(
    frame: pd.DataFrame,
    strategy: str = "cold_protein",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be below 1")

    if strategy == "cold_protein":
        keys = frame["protein_sequence"]
    elif strategy == "cold_compound":
        keys = frame["compound_smiles"]
    elif strategy == "pair":
        keys = frame["protein_sequence"] + "|" + frame["compound_smiles"]
    elif strategy == "random":
        keys = pd.Series(np.arange(len(frame)).astype(str), index=frame.index)
    else:
        raise ValueError("strategy must be one of: cold_protein, cold_compound, pair, random")

    fractions = keys.map(lambda value: _stable_fraction(str(value), seed))
    splits = np.where(
        fractions < train_fraction,
        "train",
        np.where(fractions < train_fraction + validation_fraction, "validation", "test"),
    )
    return pd.Series(splits, index=frame.index, name="split")


def make_sample(
    source: str | Path,
    destination: str | Path,
    rows: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    frame = load_dataset(source)
    if rows >= len(frame):
        sample = frame
    else:
        bins = pd.qcut(frame["label"], q=min(10, rows), duplicates="drop")
        sample = (
            frame.assign(_bin=bins)
            .groupby("_bin", observed=True, group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=max(1, round(rows * len(group) / len(frame))),
                    random_state=seed,
                ),
                include_groups=False,
            )
            .head(rows)
            .loc[:, COLUMNS]
        )
        if len(sample) < rows:
            remainder = frame.drop(index=sample.index).sample(
                rows - len(sample),
                random_state=seed,
            )
            sample = pd.concat([sample, remainder], ignore_index=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, profile, or sample the affinity dataset"
    )
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sample_rows:
        if not args.output:
            parser.error("--output is required with --sample-rows")
        frame = make_sample(args.data, args.output, args.sample_rows, args.seed)
    else:
        frame = load_dataset(args.data)
    profile = asdict(profile_dataset(frame))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
