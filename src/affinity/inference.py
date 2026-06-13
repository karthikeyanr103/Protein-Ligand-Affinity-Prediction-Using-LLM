from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from affinity.onnx_embeddings import MolLLaMAOnnxEmbedder, ProLLaMAOnnxEmbedder
from affinity.pipeline import load_metadata, standardize_apply


class ThreeOnnxAffinityPredictor:
    """ProLLaMA ONNX + Mol-LLaMA ONNX + affinity-head ONNX inference."""

    def __init__(
        self,
        artifact_directory: str | Path,
        prollama_directory: str | Path,
        mol_llama_directory: str | Path,
    ) -> None:
        import onnxruntime as ort

        artifact = Path(artifact_directory)
        metadata = load_metadata(artifact / "metadata.json")
        feature_metadata = metadata.get("features", {})
        if feature_metadata.get("feature_mode") != "llm_embeddings":
            raise ValueError("The affinity artifact was not trained with LLM embeddings")

        protein_settings = feature_metadata.get("protein_extraction", {})
        molecule_settings = feature_metadata.get("molecule_extraction", {})
        expected_prompt = "[Determine superfamily] Seq=<{value}>"
        if protein_settings.get("prompt", expected_prompt) != expected_prompt:
            raise ValueError("The artifact uses an unsupported ProLLaMA prompt")
        if protein_settings.get(
            "pooling", "attention_masked_mean_last_hidden_state"
        ) != "attention_masked_mean_last_hidden_state":
            raise ValueError("The artifact uses unsupported ProLLaMA pooling")
        if molecule_settings.get(
            "pooling", "mean_qformer_query_tokens"
        ) != "mean_qformer_query_tokens":
            raise ValueError("The artifact uses unsupported Mol-LLaMA pooling")

        self.protein_embedder = ProLLaMAOnnxEmbedder(
            model_directory=prollama_directory,
            tokenizer_id=feature_metadata["protein_model"],
            max_length=int(protein_settings.get("max_length", 1536)),
        )
        self.molecule_embedder = MolLLaMAOnnxEmbedder(mol_llama_directory)
        exported_molecule_id = self.molecule_embedder.metadata.get("model_id")
        if exported_molecule_id and exported_molecule_id != feature_metadata["molecule_model"]:
            raise ValueError(
                "Mol-LLaMA ONNX model ID does not match the training embedding cache"
            )
        self.session = ort.InferenceSession(
            str(artifact / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        normalization = np.load(artifact / "normalization.npz")
        self.mean = normalization["mean"]
        self.scale = normalization["scale"]

    def predict(self, proteins: list[str], smiles_values: list[str]) -> np.ndarray:
        if len(proteins) != len(smiles_values):
            raise ValueError("Protein and molecule input counts must match")
        protein_embeddings = self.protein_embedder.encode(proteins)
        molecule_embeddings = self.molecule_embedder.encode(smiles_values)
        features = np.concatenate([protein_embeddings, molecule_embeddings], axis=1)
        if features.shape[1] != len(self.mean):
            raise ValueError(
                f"ONNX embedding dimension {features.shape[1]} does not match "
                f"the affinity model dimension {len(self.mean)}"
            )
        features = standardize_apply(features, self.mean, self.scale)
        return self.session.run(["affinity"], {"features": features})[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict affinity through three CPU ONNX models"
    )
    parser.add_argument("--protein", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--artifacts", default="artifacts/llm_fusion")
    parser.add_argument("--prollama-onnx", required=True)
    parser.add_argument("--mol-llama-onnx", required=True)
    args = parser.parse_args()
    predictor = ThreeOnnxAffinityPredictor(
        artifact_directory=args.artifacts,
        prollama_directory=args.prollama_onnx,
        mol_llama_directory=args.mol_llama_onnx,
    )
    prediction = float(predictor.predict([args.protein], [args.smiles])[0])
    print(f"{prediction:.6f}")


if __name__ == "__main__":
    main()
