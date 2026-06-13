import numpy as np
import pytest

from affinity.features import (
    descriptor_matrix,
    join_optional_embeddings,
    protein_descriptors,
    smiles_descriptors,
)


def test_feature_shapes_and_finite_values():
    protein = protein_descriptors("ACDEFGHIKLMNPQRSTVWY")
    molecule = smiles_descriptors("CC(=O)O")
    combined = descriptor_matrix(["ACDE"], ["CCO"])
    assert protein.shape == (27,)
    assert molecule.shape == (23,)
    assert combined.shape == (1, 50)
    assert np.isfinite(combined).all()


def test_llm_fusion_requires_both_embedding_tables(tmp_path):
    with pytest.raises(ValueError, match="provided together"):
        join_optional_embeddings(
            np.zeros((1, 50), dtype=np.float32),
            ["ACDE"],
            ["CCO"],
            protein_path=str(tmp_path / "protein.npz"),
        )


def test_llm_fusion_excludes_descriptors(tmp_path):
    protein_path = tmp_path / "protein.npz"
    molecule_path = tmp_path / "molecule.npz"
    np.savez_compressed(
        protein_path,
        keys=np.asarray(["ACDE"]),
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        model_id=np.asarray("protein-model"),
    )
    np.savez_compressed(
        molecule_path,
        keys=np.asarray(["CCO"]),
        embeddings=np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32),
        model_id=np.asarray("molecule-model"),
    )
    features, metadata = join_optional_embeddings(
        np.zeros((1, 50), dtype=np.float32),
        ["ACDE"],
        ["CCO"],
        str(protein_path),
        str(molecule_path),
    )
    assert features.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert metadata["feature_mode"] == "llm_embeddings"
