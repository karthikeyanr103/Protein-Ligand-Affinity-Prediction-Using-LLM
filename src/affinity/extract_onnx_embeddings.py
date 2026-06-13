from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from affinity.data import load_dataset
from affinity.features import save_embedding_table
from affinity.onnx_embeddings import MolLLaMAOnnxEmbedder, ProLLaMAOnnxEmbedder


def extract_proteins(args, proteins: list[str]) -> None:
    destination = Path(args.protein_output)
    if destination.exists() and not args.overwrite:
        print(f"Skipping existing protein cache: {destination}")
        return
    embedder = ProLLaMAOnnxEmbedder(
        args.prollama_onnx,
        tokenizer_id=args.protein_model_id,
        max_length=args.protein_max_length,
    )
    blocks = []
    for start in range(0, len(proteins), args.protein_batch_size):
        stop = min(start + args.protein_batch_size, len(proteins))
        print(f"Proteins {start}:{stop} / {len(proteins)}")
        blocks.append(embedder.encode(proteins[start:stop]))
    save_embedding_table(
        destination,
        proteins,
        np.concatenate(blocks, axis=0),
        args.protein_model_id,
        settings={
            "runtime": "onnxruntime",
            "prompt": "[Determine superfamily] Seq=<{value}>",
            "pooling": "attention_masked_mean_last_hidden_state",
            "max_length": args.protein_max_length,
        },
    )


def extract_molecules(args, molecules: list[str]) -> None:
    output = Path(args.molecule_output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = MolLLaMAOnnxEmbedder(args.mol_llama_onnx)
    total_shards = (len(molecules) + args.molecule_shard_size - 1) // args.molecule_shard_size
    for shard_index in range(total_shards):
        start = shard_index * args.molecule_shard_size
        stop = min(start + args.molecule_shard_size, len(molecules))
        destination = output / f"molecules-{shard_index:05d}.npz"
        if destination.exists() and not args.overwrite:
            print(f"Skipping molecule shard {shard_index + 1}/{total_shards}")
            continue
        values = molecules[start:stop]
        print(f"Molecule shard {shard_index + 1}/{total_shards}: {start}:{stop}")
        save_embedding_table(
            destination,
            values,
            embedder.encode(values),
            args.molecule_model_id,
            settings={
                "runtime": "onnxruntime",
                "pooling": "mean_qformer_query_tokens",
                "conformer_seed": 42,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resumable training caches with the deployed ONNX encoders"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--prollama-onnx", required=True)
    parser.add_argument("--mol-llama-onnx", required=True)
    parser.add_argument("--protein-output", required=True)
    parser.add_argument("--molecule-output", required=True)
    parser.add_argument("--protein-model-id", default="GreatCaptainNemo/ProLLaMA")
    parser.add_argument(
        "--molecule-model-id",
        default="DongkiKim/Mol-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--protein-batch-size", type=int, default=1)
    parser.add_argument("--protein-max-length", type=int, default=1536)
    parser.add_argument("--molecule-shard-size", type=int, default=1000)
    parser.add_argument("--proteins-only", action="store_true")
    parser.add_argument("--molecules-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.proteins_only and args.molecules_only:
        parser.error("--proteins-only and --molecules-only cannot be combined")

    frame = load_dataset(args.data)
    if not args.molecules_only:
        extract_proteins(args, frame["protein_sequence"].drop_duplicates().tolist())
    if not args.proteins_only:
        extract_molecules(args, frame["compound_smiles"].drop_duplicates().tolist())


if __name__ == "__main__":
    main()
