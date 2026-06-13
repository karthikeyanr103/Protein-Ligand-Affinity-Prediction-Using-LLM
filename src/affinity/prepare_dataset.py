from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from affinity.data import assign_splits, load_dataset
from affinity.pipeline import build_features


def prepare_dataset(
    data_path: str,
    protein_embeddings: str,
    molecule_embeddings: str,
    output_directory: str,
    split_strategy: str = "cold_protein",
    seed: int = 42,
) -> Path:
    frame = load_dataset(data_path)
    frame["split"] = assign_splits(
        frame,
        strategy=split_strategy,
        train_fraction=0.8,
        validation_fraction=0.1,
        seed=seed,
    )
    features, feature_metadata = build_features(
        frame["protein_sequence"].tolist(),
        frame["compound_smiles"].tolist(),
        protein_embeddings,
        molecule_embeddings,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        mask = frame["split"].eq(split).to_numpy()
        split_frame = frame.loc[
            mask,
            ["protein_sequence", "compound_smiles", "label"],
        ].reset_index(drop=True)
        split_frame.to_csv(output / f"{split}.csv", index=False)
        np.savez_compressed(
            output / f"{split}_features.npz",
            features=features[mask].astype(np.float32),
        )
        counts[split] = int(mask.sum())
    (output / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "split_strategy": split_strategy,
                "seed": seed,
                "counts": counts,
                "feature_dimension": int(features.shape[1]),
                "features": feature_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train, validation and test files from ONNX embeddings"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--protein-embeddings", required=True)
    parser.add_argument("--molecule-embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split-strategy",
        choices=["cold_protein", "cold_compound", "pair", "random"],
        default="cold_protein",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        prepare_dataset(
            args.data,
            args.protein_embeddings,
            args.molecule_embeddings,
            args.output,
            args.split_strategy,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
