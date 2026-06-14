from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Esm2EmbeddingWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        content_mask = attention_mask * (1 - special_tokens_mask)
        mask = content_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class MolFormerEmbeddingWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.pooler_output


def _export_encoder(
    model_id: str,
    output_directory: str | Path,
    encoder_type: str,
    example: str,
    max_length: int,
    opset: int = 18,
    quantize: bool = False,
) -> Path:
    import onnxruntime as ort
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output)

    if encoder_type == "esm2":
        loaded = AutoModelForMaskedLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        loaded.config.save_pretrained(output)
        model = getattr(loaded, "base_model", loaded).eval()
        wrapper = Esm2EmbeddingWrapper(model).eval()
        tokens = tokenizer(
            [example],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        inputs = (
            tokens["input_ids"],
            tokens["attention_mask"],
            tokens["special_tokens_mask"],
        )
        input_names = ["input_ids", "attention_mask", "special_tokens_mask"]
        filename = "esm2_encoder.onnx"
        pooling = "mean_amino_acid_tokens"
    elif encoder_type == "molformer":
        model = AutoModel.from_pretrained(
            model_id,
            deterministic_eval=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).eval()
        model.config.save_pretrained(output)
        wrapper = MolFormerEmbeddingWrapper(model).eval()
        tokens = tokenizer(
            [example],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = (tokens["input_ids"], tokens["attention_mask"])
        input_names = ["input_ids", "attention_mask"]
        filename = "molformer_encoder.onnx"
        pooling = "pooler_output"
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    destination = output / filename
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["embedding"] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        inputs,
        str(destination),
        input_names=input_names,
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    with torch.inference_mode():
        reference = wrapper(*inputs).cpu().numpy().astype(np.float32)
    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        name: tensor.cpu().numpy().astype(np.int64)
        for name, tensor in zip(input_names, inputs, strict=True)
    }
    exported = session.run(["embedding"], feeds)[0].astype(np.float32)
    max_error = float(np.max(np.abs(reference - exported)))
    if not np.allclose(reference, exported, rtol=1e-3, atol=1e-4):
        raise ValueError(f"{encoder_type} ONNX parity check failed; max error={max_error}")

    final_destination = destination
    quantized_error = None
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        final_destination = output / filename.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            destination,
            final_destination,
            per_channel=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        quantized_session = ort.InferenceSession(
            str(final_destination),
            providers=["CPUExecutionProvider"],
        )
        quantized_output = quantized_session.run(["embedding"], feeds)[0].astype(
            np.float32
        )
        quantized_error = float(np.max(np.abs(reference - quantized_output)))
        if not np.allclose(reference, quantized_output, rtol=0.2, atol=0.2):
            final_destination.unlink(missing_ok=True)
            raise ValueError(
                f"Quantized {encoder_type} parity check failed; "
                f"max error={quantized_error}"
            )

    (output / "export_metadata.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "encoder_type": encoder_type,
                "output": "embedding",
                "pooling": pooling,
                "max_length": max_length,
                "opset": opset,
                "fp32_max_absolute_error": max_error,
                "int8_max_absolute_error": quantized_error,
                "quantized": quantize,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_destination


def export_esm2_onnx(
    model_id: str,
    output_directory: str | Path,
    max_length: int = 1024,
    opset: int = 18,
    quantize: bool = False,
) -> Path:
    return _export_encoder(
        model_id,
        output_directory,
        "esm2",
        "ACDEFGHIKLMNPQRSTVWY",
        max_length,
        opset,
        quantize,
    )


def export_molformer_onnx(
    model_id: str,
    output_directory: str | Path,
    max_length: int = 202,
    opset: int = 18,
    quantize: bool = False,
) -> Path:
    return _export_encoder(
        model_id,
        output_directory,
        "molformer",
        "CC(=O)Oc1ccccc1C(=O)O",
        max_length,
        opset,
        quantize,
    )


def esm2_main() -> None:
    parser = argparse.ArgumentParser(description="Export pooled ESM-2 embeddings to ONNX")
    parser.add_argument("--model-id", default="nvidia/esm2_t12_35M_UR50D")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    print(
        export_esm2_onnx(
            args.model_id,
            args.output,
            args.max_length,
            args.opset,
            args.quantize,
        )
    )


def molformer_main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pooled MoLFormer embeddings to ONNX"
    )
    parser.add_argument(
        "--model-id",
        default="ibm-research/MoLFormer-XL-both-10pct",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=202)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    print(
        export_molformer_onnx(
            args.model_id,
            args.output,
            args.max_length,
            args.opset,
            args.quantize,
        )
    )
