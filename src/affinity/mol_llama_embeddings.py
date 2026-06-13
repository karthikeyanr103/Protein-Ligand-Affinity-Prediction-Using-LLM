from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from affinity.data import load_dataset
from affinity.features import save_embedding_table


class MolLLaMAEmbedder:
    def __init__(
        self,
        official_repo: str | Path,
        model_id: str = "DongkiKim/Mol-Llama-3.1-8B-Instruct",
        precision: str = "bfloat16",
        device: str = "cuda:0",
        conformer_seed: int = 42,
    ) -> None:
        repo = str(Path(official_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from transformers import AutoTokenizer

        try:
            from models.mol_llama import MolLLaMA, get_mol_graphs_from_data
        except ImportError as error:
            raise ImportError(
                "Could not import the official Mol-LLaMA code. Clone "
                "https://github.com/DongkiKim95/Mol-LLaMA and install its requirements."
            ) from error

        self.model_id = model_id
        self.get_mol_graphs_from_data = get_mol_graphs_from_data
        self.conformer_seed = conformer_seed
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = MolLLaMA.from_pretrained(
            model_id,
            vocab_size=len(tokenizer),
            torch_dtype=precision,
            enable_flash=False,
        ).to(device)
        model.eval()
        self.encoder = model.encoder
        self.device = next(self.encoder.parameters()).device
        del model

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(self, smiles_values: list[str], batch_size: int = 1) -> np.ndarray:
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        vectors: list[np.ndarray] = []
        for start in range(0, len(smiles_values), batch_size):
            batch = smiles_values[start : start + batch_size]
            molecule_data = []
            for smiles in batch:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                molecule = Chem.AddHs(molecule)
                parameters = AllChem.ETKDGv3()
                parameters.randomSeed = self.conformer_seed
                if AllChem.EmbedMolecule(molecule, parameters) != 0:
                    raise ValueError(f"Could not generate conformer: {smiles}")
                try:
                    AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
                except ValueError:
                    pass
                molecule = Chem.RemoveHs(molecule)
                molecule_data.append(
                    {
                        "smiles": smiles,
                        "atoms": [atom.GetSymbol() for atom in molecule.GetAtoms()],
                        "coordinates": np.asarray(
                            molecule.GetConformer().GetPositions(),
                            dtype=np.float32,
                        ),
                    }
                )
            graph_batch = self.get_mol_graphs_from_data(
                molecule_data,
                self.encoder.unimol_dictionary,
                self.device,
            )
            with torch.inference_mode():
                _, _, query_output = self.encoder.graph_forward(graph_batch)
            pooled = query_output.last_hidden_state.mean(dim=1)
            if len(pooled) != len(batch):
                raise ValueError(
                    "Mol-LLaMA skipped an invalid molecule. Validate or remove that SMILES first."
                )
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


def extract_official_mol_llama_embeddings(
    smiles_values: list[str],
    official_repo: str | Path,
    model_id: str = "DongkiKim/Mol-Llama-3.1-8B-Instruct",
    batch_size: int = 1,
    precision: str = "bfloat16",
    device: str = "cuda:0",
) -> np.ndarray:
    """Extract pooled Q-Former vectors through the official Mol-LLaMA molecular stack."""
    embedder = MolLLaMAEmbedder(
        official_repo=official_repo,
        model_id=model_id,
        precision=precision,
        device=device,
    )
    return embedder.encode(smiles_values, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract molecular Q-Former embeddings with the official Mol-LLaMA code"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-id",
        default="DongkiKim/Mol-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frame = load_dataset(args.data)
    smiles_values = frame["compound_smiles"].drop_duplicates().tolist()
    embeddings = extract_official_mol_llama_embeddings(
        smiles_values=smiles_values,
        official_repo=args.official_repo,
        model_id=args.model_id,
        batch_size=args.batch_size,
        precision=args.precision,
        device=args.device,
    )
    save_embedding_table(
        args.output,
        smiles_values,
        embeddings,
        args.model_id,
        settings={
            "pooling": "mean_qformer_query_tokens",
            "precision": args.precision,
            "conformer_seed": 42,
            "official_repository": "DongkiKim95/Mol-LLaMA",
        },
    )


if __name__ == "__main__":
    main()
