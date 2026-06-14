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
RESIDUE_COLORS = {
    "D": "#ef4444",
    "E": "#ef4444",
    "K": "#3b82f6",
    "R": "#3b82f6",
    "H": "#3b82f6",
    "A": "#22c55e",
    "V": "#22c55e",
    "I": "#22c55e",
    "L": "#22c55e",
    "M": "#22c55e",
    "F": "#22c55e",
    "W": "#22c55e",
    "Y": "#22c55e",
}


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
    local_variables: str | tuple[str, ...],
    repo_variables: str | tuple[str, ...],
    allow_patterns: list[str] | None,
) -> Path:
    local_variables = (
        (local_variables,) if isinstance(local_variables, str) else local_variables
    )
    repo_variables = (repo_variables,) if isinstance(repo_variables, str) else repo_variables
    local_path = next((os.getenv(name, "") for name in local_variables if os.getenv(name)), "")
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Local model path does not exist: {path}")
        return path
    repo_id = next((os.getenv(name, "") for name in repo_variables if os.getenv(name)), "")
    if not repo_id:
        raise RuntimeError(
            f"Set one of {local_variables} or one of {repo_variables}"
        )
    return Path(
        snapshot_download(
            repo_id,
            token=os.getenv("HF_TOKEN"),
            allow_patterns=allow_patterns,
        )
    )


@lru_cache(maxsize=1)
def get_predictor() -> ThreeOnnxAffinityPredictor:
    return ThreeOnnxAffinityPredictor(
        artifact_directory=_resolve_model(
            "AFFINITY_MODEL_PATH",
            "AFFINITY_MODEL_REPO",
            ["model.onnx", "normalization.npz", "metadata.json"],
        ),
        protein_directory=_resolve_model(
            ("PROTEIN_ONNX_PATH", "PROLLAMA_ONNX_PATH"),
            ("PROTEIN_ONNX_REPO", "PROLLAMA_ONNX_REPO"),
            None,
        ),
        molecule_directory=_resolve_model(
            ("MOLECULE_ONNX_PATH", "MOL_LLAMA_ONNX_PATH"),
            ("MOLECULE_ONNX_REPO", "MOL_LLAMA_ONNX_REPO"),
            None,
        ),
        device=os.getenv("ONNX_DEVICE", "auto"),
    )


def predict(sequence: str, smiles: str):
    sequence = validate_protein(sequence)
    smiles = validate_smiles(smiles)
    try:
        prediction = float(get_predictor().predict([sequence], [smiles])[0])
    except Exception as error:
        raise gr.Error(f"Inference failed: {error}") from error
    try:
        conformer = molecule_3d(smiles)
    except Exception:
        conformer = (
            "<div class='render-note'>A 3D conformer could not be generated for "
            "this compound. The affinity prediction is still valid.</div>"
        )
    return (
        prediction_card(prediction),
        molecule_2d(smiles),
        conformer,
        molecule_summary(smiles),
        protein_render(sequence),
        protein_summary(sequence),
    )


def prediction_card(prediction: float) -> str:
    return (
        "<div class='score-card'>"
        "<div class='score-label'>Predicted affinity score</div>"
        f"<div class='score-value'>{prediction:.4f}</div>"
        "<div class='score-note'>Model output on the dataset target scale</div>"
        "</div>"
    )


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


def protein_render(sequence: str) -> str:
    sequence = validate_protein(sequence)
    residues = "".join(
        "<span class='residue' "
        f"style='background:{RESIDUE_COLORS.get(residue, '#64748b')}' "
        f"title='Position {index}: {residue}'>{html.escape(residue)}</span>"
        for index, residue in enumerate(sequence, start=1)
    )
    return (
        "<div class='protein-card'>"
        f"<div class='sequence-header'>{len(sequence):,} amino acids</div>"
        f"<div class='sequence-map'>{residues}</div>"
        "<div class='sequence-legend'>"
        "<span><i style='background:#ef4444'></i>Acidic</span>"
        "<span><i style='background:#3b82f6'></i>Basic</span>"
        "<span><i style='background:#22c55e'></i>Hydrophobic</span>"
        "<span><i style='background:#64748b'></i>Other</span>"
        "</div></div>"
    )


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

CSS = """
.gradio-container {max-width: 1220px !important; margin: auto !important;}
.hero {text-align:center; padding:1.2rem 0 .4rem;}
.hero h1 {margin-bottom:.35rem;}
.hero p {color:#64748b; margin:0 auto; max-width:760px;}
.input-card, .result-card {
    border:1px solid #e2e8f0; border-radius:16px; padding:18px;
    background:var(--block-background-fill);
}
.score-card {
    color:white; text-align:center; border-radius:16px; padding:24px;
    background:linear-gradient(135deg,#4338ca,#2563eb);
    box-shadow:0 12px 28px rgba(37,99,235,.2);
}
.score-label {font-size:15px; opacity:.9;}
.score-value {font-size:48px; line-height:1.15; font-weight:750; margin:6px 0;}
.score-note {font-size:12px; opacity:.75;}
.protein-card {border:1px solid #e2e8f0; border-radius:12px; padding:14px;}
.sequence-header {font-weight:650; margin-bottom:10px;}
.sequence-map {display:flex; flex-wrap:wrap; gap:3px; max-height:280px; overflow:auto;}
.residue {
    color:white; width:24px; height:24px; line-height:24px; text-align:center;
    border-radius:4px; font:600 12px ui-monospace,monospace;
}
.sequence-legend {display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; font-size:12px;}
.sequence-legend span {display:flex; align-items:center; gap:5px;}
.sequence-legend i {display:inline-block; width:10px; height:10px; border-radius:3px;}
.render-note {padding:24px; text-align:center; color:#64748b;}
"""

with gr.Blocks(title="Protein-Compound Affinity Explorer", css=CSS) as demo:
    gr.Markdown(
        "<div class='hero'><h1>Protein-Compound Affinity Explorer</h1>"
        "<p>Paste a protein sequence and compound SMILES to predict their affinity "
        "and inspect both inputs. Inference runs entirely through ONNX.</p></div>"
    )
    with gr.Group(elem_classes="input-card"):
        protein = gr.Textbox(
            label="Protein sequence",
            placeholder="Paste the amino-acid sequence here...",
            lines=7,
            value=EXAMPLE_PROTEIN,
        )
        smiles = gr.Textbox(
            label="Compound SMILES",
            placeholder="Paste the compound SMILES here...",
            value=EXAMPLE_SMILES,
        )
        run = gr.Button("Predict affinity and render inputs", variant="primary", size="lg")

    with gr.Column(visible=False) as results:
        score = gr.HTML()
        gr.Markdown("## Compound")
        with gr.Row():
            image = gr.Image(label="2D structure", height=420)
            view_3d = gr.HTML(label="Generated 3D conformer")
        molecule_data = gr.JSON(label="Molecule descriptors")

        gr.Markdown("## Protein")
        protein_view = gr.HTML()
        protein_data = gr.JSON(label="Protein descriptors")

    run.click(
        predict,
        [protein, smiles],
        [score, image, view_3d, molecule_data, protein_view, protein_data],
    ).then(lambda: gr.update(visible=True), outputs=results)

    with gr.Accordion("Optional protein 3D viewer", open=False):
        gr.Markdown(
            "A protein sequence does not contain 3D coordinates. Upload a PDB file "
            "to inspect an experimentally determined or predicted structure."
        )
        pdb = gr.File(label="PDB file", file_types=[".pdb"], type="filepath")
        pdb_view = gr.HTML()
        gr.Button("Render PDB structure").click(protein_3d, pdb, pdb_view)
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
