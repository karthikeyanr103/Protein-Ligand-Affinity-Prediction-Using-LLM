import numpy as np
import pytest

from affinity.features import build_embedding_features, save_embedding_table


def test_embedding_features_use_both_onnx_tables(tmp_path):
    protein_path = tmp_path / "proteins.npz"
    molecule_directory = tmp_path / "molecules"
    save_embedding_table(
        protein_path,
        ["ACDE"],
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        "protein-model",
        {"runtime": "onnxruntime"},
    )
    save_embedding_table(
        molecule_directory / "molecules-00000.npz",
        ["CCO"],
        np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32),
        "molecule-model",
        {"runtime": "onnxruntime"},
    )
    features, metadata = build_embedding_features(
        ["ACDE"],
        ["CCO"],
        protein_path,
        molecule_directory,
    )
    assert features.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert metadata["feature_mode"] == "onnx_embeddings"


def test_embedding_features_require_both_tables(tmp_path):
    with pytest.raises(ValueError, match="Both ONNX embedding tables"):
        build_embedding_features(["ACDE"], ["CCO"], tmp_path / "protein.npz", "")
