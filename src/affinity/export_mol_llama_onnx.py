from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class MolLLaMAEncoderWrapper(nn.Module):
    """Flatten the official Mol-LLaMA graph-forward API into ONNX tensor inputs."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def _molecule_stm(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.encoder.graph_encoder["moleculestm"]
        gnn = model.molecule_node_model
        if gnn.JK != "last":
            raise ValueError("The ONNX wrapper currently requires MoleculeSTM JK='last'")
        hidden = gnn.atom_encoder(node_features)
        for layer_index in range(gnn.num_layer):
            convolution = gnn.gnns[layer_index]
            encoded_edges = convolution.bond_encoder(edge_features)
            source, target = edge_index[0], edge_index[1]
            messages = F.relu(hidden[source] + encoded_edges)
            node_ids = torch.arange(
                hidden.shape[0],
                dtype=target.dtype,
                device=target.device,
            )
            incidence = node_ids.unsqueeze(1).eq(target.unsqueeze(0))
            aggregated = incidence.to(messages.dtype).matmul(messages)
            hidden = convolution.mlp(
                (1 + convolution.eps) * hidden + aggregated
            )
            hidden = gnn.batch_norms[layer_index](hidden)
            if layer_index != gnn.num_layer - 1:
                hidden = F.relu(hidden)
        graph_representation = hidden.mean(dim=0, keepdim=True)
        graph_embedding = torch.cat(
            [graph_representation.unsqueeze(1), hidden.unsqueeze(0)],
            dim=1,
        )
        graph_mask = torch.ones(
            graph_embedding.shape[:2],
            dtype=torch.bool,
            device=graph_embedding.device,
        )
        return graph_embedding, graph_mask

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_edge_type: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        molecule_nodes, molecule_mask = self._molecule_stm(
            node_features,
            edge_index,
            edge_features,
        )
        unimol_nodes, unimol_mask = self.encoder.graph_encoder["unimol"](
            src_tokens,
            src_distance,
            src_edge_type,
        )
        graph_nodes = {
            "moleculestm": self.encoder.ln_graph["moleculestm"](molecule_nodes),
            "unimol": self.encoder.ln_graph["unimol"](unimol_nodes),
        }
        graph_masks = {
            "moleculestm": molecule_mask,
            "unimol": unimol_mask,
        }
        blended_nodes, blended_mask, _ = self.encoder.blending_module(
            graph_nodes,
            graph_masks,
        )
        query_tokens = self.encoder.query_tokens.expand(
            blended_nodes.shape[0],
            -1,
            -1,
        )
        query_output = self.encoder.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=blended_nodes,
            encoder_attention_mask=blended_mask,
            use_cache=False,
            return_dict=True,
        )
        return query_output.last_hidden_state.mean(dim=1)


def _save_dictionary(dictionary, destination: Path) -> None:
    payload = {
        "symbols": list(dictionary.symbols),
        "bos_index": int(dictionary.bos()),
        "pad_index": int(dictionary.pad()),
        "eos_index": int(dictionary.eos()),
        "unk_index": int(dictionary.unk()),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_mol_llama_onnx(
    official_repo: str | Path,
    output_directory: str | Path,
    model_id: str,
    example_smiles: str,
    opset: int,
    quantize: bool,
) -> Path:
    repo = str(Path(official_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from transformers import AutoTokenizer

    from models.mol_llama import MolLLaMA, get_mol_graphs

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MolLLaMA.from_pretrained(
        model_id,
        vocab_size=len(tokenizer),
        torch_dtype="float32",
        enable_flash=False,
    ).cpu()
    model.eval()
    encoder = model.encoder.eval()
    graph_batch = get_mol_graphs(
        [example_smiles],
        encoder.unimol_dictionary,
        torch.device("cpu"),
    )
    molecule_graph = graph_batch["moleculestm"]
    inputs = (
        graph_batch["unimol"]["src_tokens"],
        graph_batch["unimol"]["src_distance"],
        graph_batch["unimol"]["src_edge_type"],
        molecule_graph.x,
        molecule_graph.edge_index,
        molecule_graph.edge_attr,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "mol_llama_encoder.onnx"
    _save_dictionary(encoder.unimol_dictionary, output / "unimol_dictionary.json")
    wrapper = MolLLaMAEncoderWrapper(encoder).eval()
    torch.onnx.export(
        wrapper,
        inputs,
        str(destination),
        input_names=[
            "src_tokens",
            "src_distance",
            "src_edge_type",
            "node_features",
            "edge_index",
            "edge_features",
        ],
        output_names=["molecule_embedding"],
        dynamic_axes={
            "src_tokens": {0: "batch", 1: "unimol_atoms"},
            "src_distance": {0: "batch", 1: "unimol_atoms", 2: "unimol_atoms"},
            "src_edge_type": {0: "batch", 1: "unimol_atoms", 2: "unimol_atoms"},
            "node_features": {0: "graph_nodes"},
            "edge_index": {1: "graph_edges"},
            "edge_features": {0: "graph_edges"},
            "molecule_embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnxruntime as ort

    input_names = [
        "src_tokens",
        "src_distance",
        "src_edge_type",
        "node_features",
        "edge_index",
        "edge_features",
    ]
    with torch.inference_mode():
        reference = wrapper(*inputs).cpu().numpy()
        _, _, official_query = encoder.graph_forward(graph_batch)
        official_reference = (
            official_query.last_hidden_state.mean(dim=1).cpu().numpy()
        )
    wrapper_max_absolute_error = float(
        np.max(np.abs(reference - official_reference))
    )
    if not np.allclose(reference, official_reference, rtol=1e-4, atol=1e-5):
        raise ValueError(
            "ONNX-friendly MoleculeSTM rewrite does not match the official encoder; "
            f"max error={wrapper_max_absolute_error}"
        )
    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, inputs, strict=True)
    }
    exported = session.run(["molecule_embedding"], feeds)[0]
    max_absolute_error = float(np.max(np.abs(reference - exported)))
    if not np.allclose(reference, exported, rtol=1e-3, atol=1e-4):
        raise ValueError(
            f"Mol-LLaMA ONNX parity check failed; max error={max_absolute_error}"
        )
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = output / "mol_llama_encoder_int8.onnx"
        quantize_dynamic(
            destination,
            quantized,
            weight_type=QuantType.QInt8,
        )
        destination = quantized
    (output / "export_metadata.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "output": "mean_qformer_query_tokens",
                "conformer_seed": 42,
                "example_smiles": example_smiles,
                "opset": opset,
                "fp32_max_absolute_error": max_absolute_error,
                "wrapper_max_absolute_error": wrapper_max_absolute_error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the official Mol-LLaMA molecular encoder to ONNX"
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-id",
        default="DongkiKim/Mol-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--example-smiles", default="CC(=O)O")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    print(
        export_mol_llama_onnx(
            args.official_repo,
            args.output,
            args.model_id,
            args.example_smiles,
            args.opset,
            args.quantize,
        )
    )


if __name__ == "__main__":
    main()
