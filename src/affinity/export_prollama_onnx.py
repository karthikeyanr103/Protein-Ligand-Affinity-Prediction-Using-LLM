from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ProLLaMAEncoderWrapper(nn.Module):
    """Expose the decoder's final hidden state without the causal-LM head."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        return output[0]


def _find_decoder(model: nn.Module) -> nn.Module:
    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if candidate.__class__.__name__ == "LlamaModel":
        return candidate
    if hasattr(candidate, "model") and candidate.model.__class__.__name__ == "LlamaModel":
        return candidate.model
    raise TypeError(
        "Could not locate the bare LlamaModel decoder inside the ProLLaMA checkpoint"
    )


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def export_prollama_feature_onnx(
    model_id: str,
    output_directory: str | Path,
    sequence_length: int = 32,
    opset: int = 18,
    dtype: str = "float32",
    quantize: bool = False,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.save_pretrained(output)
    model.eval()
    decoder = _find_decoder(model).eval()
    wrapper = ProLLaMAEncoderWrapper(decoder).eval()

    prompt = "[Determine superfamily] Seq=<ACDEFGHIKLMNPQRSTVWY>"
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    position_ids = (
        attention_mask.long().cumsum(-1) - 1
    ).masked_fill(attention_mask.eq(0), 0)
    inputs = (input_ids, attention_mask, position_ids)

    destination = output / "prollama_encoder.onnx"
    export_options = {
        "input_names": ["input_ids", "attention_mask", "position_ids"],
        "output_names": ["last_hidden_state"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "position_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        export_options["external_data"] = True
    torch.onnx.export(
        wrapper,
        inputs,
        str(destination),
        **export_options,
    )

    import onnxruntime as ort

    with torch.inference_mode():
        reference = wrapper(*inputs).float().cpu().numpy()
    del wrapper
    del decoder
    del model
    gc.collect()
    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }
    exported = session.run(["last_hidden_state"], feeds)[0].astype(np.float32)
    max_absolute_error = float(np.max(np.abs(reference - exported)))
    tolerance = 1e-4 if dtype == "float32" else 2e-2
    if not np.allclose(reference, exported, rtol=tolerance, atol=tolerance):
        raise ValueError(
            f"ProLLaMA ONNX parity check failed; max error={max_absolute_error}"
        )

    final_destination = destination
    quantized_max_absolute_error = None
    if quantize:
        if dtype != "float32":
            raise ValueError("INT8 dynamic quantization requires a float32 ONNX source")
        del session
        del exported
        gc.collect()

        from onnxruntime.quantization import QuantType, quantize_dynamic

        final_destination = output / "prollama_encoder_int8.onnx"
        quantize_dynamic(
            destination,
            final_destination,
            weight_type=QuantType.QInt8,
            use_external_data_format=True,
        )
        quantized_session = ort.InferenceSession(
            str(final_destination),
            providers=["CPUExecutionProvider"],
        )
        quantized_output = quantized_session.run(
            ["last_hidden_state"],
            feeds,
        )[0].astype(np.float32)
        quantized_max_absolute_error = float(
            np.max(np.abs(reference - quantized_output))
        )
        if not np.allclose(reference, quantized_output, rtol=0.15, atol=0.15):
            raise ValueError(
                "Quantized ProLLaMA ONNX parity check failed; "
                f"max error={quantized_max_absolute_error}"
            )

    (output / "export_metadata.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "output": "last_hidden_state",
                "dtype": dtype,
                "opset": opset,
                "example_sequence_length": int(input_ids.shape[1]),
                "fp_max_absolute_error": max_absolute_error,
                "int8_max_absolute_error": quantized_max_absolute_error,
                "quantized": quantize,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually export ProLLaMA hidden states to ONNX"
    )
    parser.add_argument("--model-id", default="GreatCaptainNemo/ProLLaMA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    print(
        export_prollama_feature_onnx(
            model_id=args.model_id,
            output_directory=args.output,
            sequence_length=args.sequence_length,
            opset=args.opset,
            dtype=args.dtype,
            quantize=args.quantize,
        )
    )


if __name__ == "__main__":
    main()
