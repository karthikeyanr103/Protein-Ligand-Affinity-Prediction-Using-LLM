from __future__ import annotations

import html
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import py3Dmol
from huggingface_hub import snapshot_download
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski

from affinity.inference import ThreeOnnxAffinityPredictor

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_protein(sequence: str) -> str:
    sequence = "".join(sequence.upper().split())
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise gr.Error("Protein must contain only the 20 standard amino-acid letters.")
    return sequence


def validate_smiles(smiles: str) -> str:
    smiles = smiles.strip()
    if Chem.MolFromSmiles(smiles) is None:
        raise gr.Error("The SMILES string is invalid.")
    return smiles


def _resolve_model(
    local_variable: str,
    repo_variable: str,
    allow_patterns: list[str],
) -> Path:
    local_path = os.getenv(local_variable, "")
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Local model path does not exist: {path}")
        return path
    repo_id = os.getenv(repo_variable, "")
    if not repo_id:
        raise RuntimeError(f"Set {local_variable} or {repo_variable}")
    return Path(
        snapshot_download(
            repo_id,
            token=os.getenv("HF_TOKEN"),
            allow_patterns=allow_patterns,
        )
    )


@lru_cache(maxsize=1)
def get_predictor() -> ThreeOnnxAffinityPredictor:
    pro_variant = os.getenv("PROLLAMA_ONNX_VARIANT", "int8").lower()
    mol_variant = os.getenv("MOL_LLAMA_ONNX_VARIANT", "int8").lower()
    pro_pattern = "*int8*" if pro_variant == "int8" else "prollama_encoder.onnx*"
    mol_pattern = "*int8*" if mol_variant == "int8" else "mol_llama_encoder.onnx*"
    return ThreeOnnxAffinityPredictor(
        artifact_directory=_resolve_model(
            "AFFINITY_MODEL_PATH",
            "AFFINITY_MODEL_REPO",
            ["model.onnx", "normalization.npz", "metadata.json"],
        ),
        prollama_directory=_resolve_model(
            "PROLLAMA_ONNX_PATH",
            "PROLLAMA_ONNX_REPO",
            [
                pro_pattern,
                "tokenizer*",
                "special_tokens_map.json",
                "config.json",
                "export_metadata.json",
            ],
        ),
        mol_llama_directory=_resolve_model(
            "MOL_LLAMA_ONNX_PATH",
            "MOL_LLAMA_ONNX_REPO",
            [mol_pattern, "unimol_dictionary.json", "export_metadata.json"],
        ),
    )


def predict(sequence: str, smiles: str):
    sequence = validate_protein(sequence)
    smiles = validate_smiles(smiles)
    try:
        prediction = float(get_predictor().predict([sequence], [smiles])[0])
    except Exception as error:
        raise gr.Error(f"Inference failed: {error}") from error
    return prediction, molecule_summary(smiles), protein_summary(sequence)


def molecule_summary(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(validate_smiles(smiles))
    return {
        "molecular_weight": round(Descriptors.MolWt(molecule), 3),
        "log_p": round(Descriptors.MolLogP(molecule), 3),
        "h_bond_donors": Lipinski.NumHDonors(molecule),
        "h_bond_acceptors": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
    }


def protein_summary(sequence: str) -> dict:
    sequence = validate_protein(sequence)
    return {
        "length": len(sequence),
        "acidic_fraction": round(sum(sequence.count(aa) for aa in "DE") / len(sequence), 4),
        "basic_fraction": round(sum(sequence.count(aa) for aa in "KRH") / len(sequence), 4),
        "hydrophobic_fraction": round(
            sum(sequence.count(aa) for aa in "AVILMFWY") / len(sequence), 4
        ),
    }


def molecule_2d(smiles: str):
    return Draw.MolToImage(Chem.MolFromSmiles(validate_smiles(smiles)), size=(700, 450))


def molecule_3d(smiles: str) -> str:
    molecule = Chem.AddHs(Chem.MolFromSmiles(validate_smiles(smiles)))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 42
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise gr.Error("RDKit could not generate a conformer for this molecule.")
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    viewer = py3Dmol.view(width=800, height=500)
    viewer.addModel(Chem.MolToMolBlock(molecule), "mol")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    viewer.zoomTo()
    return viewer._make_html()


def protein_3d(pdb_file) -> str:
    if pdb_file is None:
        raise gr.Error("Upload a PDB file. A sequence alone has no 3D coordinates.")
    pdb_text = Path(pdb_file).read_text(encoding="utf-8", errors="replace")
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(pdb_text, "pdb")
    viewer.setStyle({"cartoon": {"color": "spectrum"}})
    viewer.zoomTo()
    return viewer._make_html()


EXAMPLE_PROTEIN = "MAVMKNYLLPILVLFLAYYYYSTNEEFRPEMLQGKKVIVTGASKGIGREMAYHLSKMGAHVVLTARSEEGLQK"
EXAMPLE_SMILES = "C1CC1(C2=CC=C(C=C2)F)C(=O)N3CC4CC4(C3)C5=CNC6=C5C=CC=N6"

with gr.Blocks(title="Protein-Compound Affinity Explorer") as demo:
    gr.Markdown(
        "# Protein-Compound Affinity Explorer\n"
        "Each prediction runs ProLLaMA ONNX, Mol-LLaMA ONNX, and the affinity ONNX head. "
        "The first request downloads and initializes all three CPU models."
    )
    with gr.Tab("Predict"):
        protein = gr.Textbox(label="Protein sequence", lines=6, value=EXAMPLE_PROTEIN)
        smiles = gr.Textbox(label="Compound SMILES", value=EXAMPLE_SMILES)
        run = gr.Button("Run ONNX inference", variant="primary")
        score = gr.Number(label="Predicted affinity label")
        molecule_data = gr.JSON(label="Molecule descriptors")
        protein_data = gr.JSON(label="Protein descriptors")
        run.click(predict, [protein, smiles], [score, molecule_data, protein_data])
    with gr.Tab("Molecule 2D/3D"):
        molecule_input = gr.Textbox(label="SMILES", value=EXAMPLE_SMILES)
        with gr.Row():
            image = gr.Image(label="2D structure")
            view_3d = gr.HTML(label="3D conformer")
        gr.Button("Render molecule").click(
            lambda value: (molecule_2d(value), molecule_3d(value)),
            molecule_input,
            [image, view_3d],
        )
    with gr.Tab("Protein 3D"):
        pdb = gr.File(label="PDB file", file_types=[".pdb"], type="filepath")
        protein_view = gr.HTML()
        gr.Button("Render protein").click(protein_3d, pdb, protein_view)
    gr.Markdown(
        html.escape(
            "Research demonstration only. Predictions are not medical or drug-development advice."
        )
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
