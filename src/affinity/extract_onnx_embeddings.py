from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from affinity.data import load_dataset
from affinity.features import save_embedding_table
from affinity.onnx_embeddings import (
    create_onnx_embedder,
    detect_onnx_providers,
)


def extract_proteins(args, proteins: list[str], providers: list[str]) -> None:
    destination = Path(args.protein_output)
    if destination.exists() and not args.overwrite:
        print(f"Skipping existing protein cache: {destination}")
        return
    embedder = create_onnx_embedder(
        args.protein_encoder,
        args.protein_onnx,
        model_id=args.protein_model_id,
        max_length=args.protein_max_length,
        providers=providers,
    )
    print(
        f"Protein encoder ({args.protein_encoder}) providers: "
        f"{embedder.session.get_providers()}",
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
            "encoder_type": args.protein_encoder,
            "pooling": (
                "mean_amino_acid_tokens"
                if args.protein_encoder == "esm2"
                else "attention_masked_mean_last_hidden_state"
            ),
            "max_length": args.protein_max_length,
            **(
                {"prompt": "[Determine superfamily] Seq=<{value}>"}
                if args.protein_encoder == "prollama"
                else {}
            ),
        },
    )


def extract_molecules(args, molecules: list[str], providers: list[str]) -> None:
    output = Path(args.molecule_output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = create_onnx_embedder(
        args.molecule_encoder,
        args.molecule_onnx,
        model_id=args.molecule_model_id,
        max_length=args.molecule_max_length,
        providers=providers,
    )
    print(
        f"Molecule encoder ({args.molecule_encoder}) providers: "
        f"{embedder.session.get_providers()}",
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
        for batch_start in range(0, len(values), args.molecule_batch_size):
            batch = values[batch_start : batch_start + args.molecule_batch_size]
            embeddings.extend(embedder.encode(batch))
            progress.update(len(batch))
        save_embedding_table(
            destination,
            values,
            np.asarray(embeddings, dtype=np.float32),
            args.molecule_model_id,
            settings={
                "runtime": "onnxruntime",
                "encoder_type": args.molecule_encoder,
                "pooling": (
                    "pooler_output"
                    if args.molecule_encoder == "molformer"
                    else "mean_qformer_query_tokens"
                ),
                "max_length": args.molecule_max_length,
                **(
                    {"conformer_seed": 42}
                    if args.molecule_encoder == "mol_llama"
                    else {}
                ),
            },
        )
    progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resumable training caches with the deployed ONNX encoders"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--protein-onnx")
    parser.add_argument("--molecule-onnx")
    parser.add_argument("--prollama-onnx")
    parser.add_argument("--mol-llama-onnx")
    parser.add_argument("--protein-output", required=True)
    parser.add_argument("--molecule-output", required=True)
    parser.add_argument(
        "--protein-encoder",
        choices=["esm2", "prollama"],
        default="esm2",
    )
    parser.add_argument(
        "--molecule-encoder",
        choices=["molformer", "mol_llama"],
        default="molformer",
    )
    parser.add_argument(
        "--protein-model-id",
        default="facebook/esm2_t12_35M_UR50D",
    )
    parser.add_argument(
        "--molecule-model-id",
        default="ibm-research/MoLFormer-XL-both-10pct",
    )
    parser.add_argument("--protein-batch-size", type=int, default=16)
    parser.add_argument("--molecule-batch-size", type=int, default=32)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--molecule-max-length", type=int, default=202)
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
    if args.prollama_onnx:
        args.protein_onnx = args.protein_onnx or args.prollama_onnx
        args.protein_encoder = "prollama"
        if args.protein_model_id == "facebook/esm2_t12_35M_UR50D":
            args.protein_model_id = "GreatCaptainNemo/ProLLaMA"
    if args.mol_llama_onnx:
        args.molecule_onnx = args.molecule_onnx or args.mol_llama_onnx
        args.molecule_encoder = "mol_llama"
        if args.molecule_model_id == "ibm-research/MoLFormer-XL-both-10pct":
            args.molecule_model_id = "DongkiKim/Mol-Llama-3.1-8B-Instruct"
    if not args.protein_onnx or not args.molecule_onnx:
        parser.error("--protein-onnx and --molecule-onnx are required")
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
        print(f"Loading {args.protein_encoder} protein session...", flush=True)
        extract_proteins(args, proteins, providers)
    if not args.proteins_only:
        print(f"Loading {args.molecule_encoder} molecule session...", flush=True)
        extract_molecules(args, molecules, providers)


if __name__ == "__main__":
    main()
