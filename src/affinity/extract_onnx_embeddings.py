from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from affinity.data import load_dataset
from affinity.features import save_embedding_table
from affinity.onnx_embeddings import (
    MolLLaMAOnnxEmbedder,
    ProLLaMAOnnxEmbedder,
    detect_onnx_providers,
)


def extract_proteins(args, proteins: list[str], providers: list[str]) -> None:
    destination = Path(args.protein_output)
    if destination.exists() and not args.overwrite:
        print(f"Skipping existing protein cache: {destination}")
        return
    embedder = ProLLaMAOnnxEmbedder(
        args.prollama_onnx,
        tokenizer_id=args.protein_model_id,
        max_length=args.protein_max_length,
        providers=providers,
    )
    print(
        f"ProLLaMA session providers: {embedder.session.get_providers()}",
        flush=True,
    )
    blocks = []
    progress = tqdm(
        total=len(proteins),
        desc="Protein embeddings",
        unit="protein",
        dynamic_ncols=True,
    )
    for start in range(0, len(proteins), args.protein_batch_size):
        batch = proteins[start : start + args.protein_batch_size]
        blocks.append(embedder.encode(batch))
        progress.update(len(batch))
    progress.close()
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


def extract_molecules(args, molecules: list[str], providers: list[str]) -> None:
    output = Path(args.molecule_output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = MolLLaMAOnnxEmbedder(args.mol_llama_onnx, providers=providers)
    print(
        f"Mol-LLaMA session providers: {embedder.session.get_providers()}",
        flush=True,
    )
    total_shards = (len(molecules) + args.molecule_shard_size - 1) // args.molecule_shard_size
    completed = 0
    if not args.overwrite:
        for shard_index in range(total_shards):
            start = shard_index * args.molecule_shard_size
            stop = min(start + args.molecule_shard_size, len(molecules))
            destination = output / f"molecules-{shard_index:05d}.npz"
            if destination.exists():
                completed += stop - start
    progress = tqdm(
        total=len(molecules),
        initial=completed,
        desc="Molecule embeddings",
        unit="molecule",
        dynamic_ncols=True,
    )
    for shard_index in range(total_shards):
        start = shard_index * args.molecule_shard_size
        stop = min(start + args.molecule_shard_size, len(molecules))
        destination = output / f"molecules-{shard_index:05d}.npz"
        if destination.exists() and not args.overwrite:
            continue
        values = molecules[start:stop]
        progress.set_postfix_str(f"shard {shard_index + 1}/{total_shards}")
        embeddings = []
        for smiles in values:
            embeddings.append(embedder.encode([smiles])[0])
            progress.update(1)
        save_embedding_table(
            destination,
            values,
            np.asarray(embeddings, dtype=np.float32),
            args.molecule_model_id,
            settings={
                "runtime": "onnxruntime",
                "pooling": "mean_qformer_query_tokens",
                "conformer_seed": 42,
            },
        )
    progress.close()


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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ONNX Runtime device selection",
    )
    args = parser.parse_args()
    if args.proteins_only and args.molecules_only:
        parser.error("--proteins-only and --molecules-only cannot be combined")

    print("Loading dataset...", flush=True)
    frame = load_dataset(args.data)
    proteins = frame["protein_sequence"].drop_duplicates().tolist()
    molecules = frame["compound_smiles"].drop_duplicates().tolist()
    print(
        f"Unique proteins: {len(proteins):,} | "
        f"Unique molecules: {len(molecules):,}",
        flush=True,
    )
    providers = detect_onnx_providers(args.device, verbose=True)
    if not args.molecules_only:
        print("Loading ProLLaMA ONNX session...", flush=True)
        extract_proteins(args, proteins, providers)
    if not args.proteins_only:
        print("Loading Mol-LLaMA ONNX session...", flush=True)
        extract_molecules(args, molecules, providers)


if __name__ == "__main__":
    main()
