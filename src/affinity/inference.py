from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from affinity.onnx_embeddings import create_onnx_embedder, detect_onnx_providers
from affinity.pipeline import load_metadata, standardize_apply


class ThreeOnnxAffinityPredictor:
    """Protein ONNX + molecule ONNX + affinity-head ONNX inference."""

    def __init__(
        self,
        artifact_directory: str | Path,
        protein_directory: str | Path | None = None,
        molecule_directory: str | Path | None = None,
        device: str = "auto",
        *,
        prollama_directory: str | Path | None = None,
        mol_llama_directory: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        protein_directory = protein_directory or prollama_directory
        molecule_directory = molecule_directory or mol_llama_directory
        if protein_directory is None or molecule_directory is None:
            raise ValueError("Both protein and molecule ONNX directories are required")
        artifact = Path(artifact_directory)
        metadata = load_metadata(artifact / "metadata.json")
        feature_metadata = metadata.get("features", {})
        if feature_metadata.get("feature_mode") != "onnx_embeddings":
            raise ValueError("The affinity artifact was not trained with ONNX embeddings")

        protein_settings = feature_metadata.get("protein_extraction", {})
        molecule_settings = feature_metadata.get("molecule_extraction", {})
        protein_type = protein_settings.get("encoder_type")
        if not protein_type:
            protein_type = (
                "prollama"
                if "prollama" in feature_metadata["protein_model"].lower()
                else "esm2"
            )
        molecule_type = molecule_settings.get("encoder_type")
        if not molecule_type:
            molecule_type = (
                "mol_llama"
                if "mol-llama" in feature_metadata["molecule_model"].lower()
                else "molformer"
            )
        providers = detect_onnx_providers(device, verbose=True)
        self.protein_embedder = create_onnx_embedder(
            protein_type,
            model_directory=protein_directory,
            model_id=feature_metadata["protein_model"],
            max_length=int(protein_settings.get("max_length", 1536)),
            providers=providers,
        )
        exported_protein_id = self.protein_embedder.metadata.get("model_id")
        if exported_protein_id and exported_protein_id != feature_metadata["protein_model"]:
            raise ValueError(
                "Protein ONNX model ID does not match the training embedding cache"
            )
        self.molecule_embedder = create_onnx_embedder(
            molecule_type,
            model_directory=molecule_directory,
            model_id=feature_metadata["molecule_model"],
            max_length=int(molecule_settings.get("max_length", 202)),
            providers=providers,
        )
        exported_molecule_id = self.molecule_embedder.metadata.get("model_id")
        if exported_molecule_id and exported_molecule_id != feature_metadata["molecule_model"]:
            raise ValueError(
                "Molecule ONNX model ID does not match the training embedding cache"
            )
        self.session = ort.InferenceSession(
            str(artifact / "model.onnx"),
            providers=providers,
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
        description="Predict affinity through three ONNX models"
    )
    parser.add_argument("--protein", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--artifacts", default="/content/artifacts/affinity")
    parser.add_argument(
        "--protein-onnx",
        "--prollama-onnx",
        dest="protein_onnx",
        required=True,
    )
    parser.add_argument(
        "--molecule-onnx",
        "--mol-llama-onnx",
        dest="molecule_onnx",
        required=True,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    predictor = ThreeOnnxAffinityPredictor(
        artifact_directory=args.artifacts,
        protein_directory=args.protein_onnx,
        molecule_directory=args.molecule_onnx,
        device=args.device,
    )
    prediction = float(predictor.predict([args.protein], [args.smiles])[0])
    print(f"{prediction:.6f}")


if __name__ == "__main__":
    main()
