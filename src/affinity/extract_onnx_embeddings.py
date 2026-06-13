from __future__ import annotations

import argparse

from affinity.data import load_dataset
from affinity.features import save_embedding_table
from affinity.onnx_embeddings import MolLLaMAOnnxEmbedder, ProLLaMAOnnxEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training caches with the same ONNX encoders used for inference"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--prollama-onnx", required=True)
    parser.add_argument("--mol-llama-onnx", required=True)
    parser.add_argument("--protein-output", required=True)
    parser.add_argument("--molecule-output", required=True)
    parser.add_argument(
        "--protein-model-id",
        default="GreatCaptainNemo/ProLLaMA",
    )
    parser.add_argument(
        "--molecule-model-id",
        default="DongkiKim/Mol-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--protein-batch-size", type=int, default=1)
    parser.add_argument("--protein-max-length", type=int, default=1536)
    args = parser.parse_args()

    frame = load_dataset(args.data)
    proteins = frame["protein_sequence"].drop_duplicates().tolist()
    molecules = frame["compound_smiles"].drop_duplicates().tolist()
    protein_embedder = ProLLaMAOnnxEmbedder(
        args.prollama_onnx,
        tokenizer_id=args.protein_model_id,
        max_length=args.protein_max_length,
    )
    protein_blocks = []
    for start in range(0, len(proteins), args.protein_batch_size):
        protein_blocks.append(
            protein_embedder.encode(
                proteins[start : start + args.protein_batch_size]
            )
        )
    import numpy as np

    protein_embeddings = np.concatenate(protein_blocks, axis=0)
    save_embedding_table(
        args.protein_output,
        proteins,
        protein_embeddings,
        args.protein_model_id,
        settings={
            "runtime": "onnxruntime",
            "prompt": "[Determine superfamily] Seq=<{value}>",
            "pooling": "attention_masked_mean_last_hidden_state",
            "max_length": args.protein_max_length,
        },
    )

    molecule_embedder = MolLLaMAOnnxEmbedder(args.mol_llama_onnx)
    molecule_embeddings = molecule_embedder.encode(molecules)
    save_embedding_table(
        args.molecule_output,
        molecules,
        molecule_embeddings,
        args.molecule_model_id,
        settings={
            "runtime": "onnxruntime",
            "pooling": "mean_qformer_query_tokens",
            "conformer_seed": 42,
        },
    )


if __name__ == "__main__":
    main()

