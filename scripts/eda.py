from __future__ import annotations

import argparse
import json
from pathlib import Path

from affinity.data import load_dataset, profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset and target analysis artifacts")
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="artifacts/eda")
    parser.add_argument("--plot-sample", type=int, default=50_000)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(args.data)
    frame["protein_length"] = frame["protein_sequence"].str.len()
    frame["smiles_length"] = frame["compound_smiles"].str.len()
    profile = profile_dataset(frame)
    (output / "profile.json").write_text(
        json.dumps(profile.__dict__, indent=2),
        encoding="utf-8",
    )

    sample = frame.sample(min(args.plot_sample, len(frame)), random_state=42)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(sample["label"], bins=40, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Affinity target distribution")
    sns.histplot(sample["protein_length"], bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("Protein sequence length")
    sns.histplot(sample["smiles_length"], bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("SMILES length")
    sns.scatterplot(
        data=sample,
        x="protein_length",
        y="label",
        alpha=0.25,
        s=15,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Target vs protein length")
    figure.tight_layout()
    figure.savefig(output / "dataset_overview.png", dpi=180)
    plt.close(figure)

    label_by_protein = (
        frame.groupby("protein_sequence")["label"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )
    label_by_protein.head(100).to_csv(output / "top_proteins.csv")
    print(json.dumps(profile.__dict__, indent=2))


if __name__ == "__main__":
    main()
