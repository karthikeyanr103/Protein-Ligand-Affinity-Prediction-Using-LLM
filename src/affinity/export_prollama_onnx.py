from __future__ import annotations

import argparse
import gc
import inspect
import json
import shutil
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


def _low_memory_session(model_path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _remove_rejected_quantization(directory: Path) -> None:
    quantized_directory = directory / "int8"
    if quantized_directory.exists():
        shutil.rmtree(quantized_directory)

    legacy_graph = directory / "prollama_encoder_int8.onnx"
    if not legacy_graph.exists():
        return

    try:
        import onnx

        model = onnx.load(str(legacy_graph), load_external_data=False)
        fp_graph = directory / "prollama_encoder.onnx"
        fp_locations = set()
        if fp_graph.exists():
            fp_model = onnx.load(str(fp_graph), load_external_data=False)
            fp_locations = {
                item.value
                for tensor in fp_model.graph.initializer
                for item in tensor.external_data
                if item.key == "location"
            }
        locations = {
            item.value
            for tensor in model.graph.initializer
            for item in tensor.external_data
            if item.key == "location"
        }
        for location in locations - fp_locations:
            external_file = legacy_graph.parent / location
            if external_file.is_file():
                external_file.unlink()
    finally:
        legacy_graph.unlink(missing_ok=True)


def export_prollama_feature_onnx(
    model_id: str,
    output_directory: str | Path,
    sequence_length: int = 32,
    opset: int = 18,
    dtype: str = "float32",
    quantize: bool = False,
    skip_parity_check: bool = False,
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

    reference = None
    if not skip_parity_check:
        with torch.inference_mode():
            reference = wrapper(*inputs).float().cpu().numpy()
    del wrapper
    del decoder
    del model
    gc.collect()
    feeds = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }
    max_absolute_error = None
    session = None
    if not skip_parity_check:
        session = _low_memory_session(destination)
        exported = session.run(["last_hidden_state"], feeds)[0].astype(np.float32)
        max_absolute_error = float(np.max(np.abs(reference - exported)))
        tolerance = 1e-4 if dtype == "float32" else 2e-2
        if not np.allclose(reference, exported, rtol=tolerance, atol=tolerance):
            raise ValueError(
                f"ProLLaMA ONNX parity check failed; max error={max_absolute_error}"
            )

    final_destination = destination
    quantized_max_absolute_error = None
    quantization_status = "not_requested"
    if quantize:
        if skip_parity_check:
            raise ValueError(
                "Quantization and --skip-parity-check cannot be combined. "
                "Quantize in a fresh high-memory process after validating FP32."
            )
        if dtype != "float32":
            raise ValueError("INT8 dynamic quantization requires a float32 ONNX source")
        del session
        del exported
        gc.collect()

        from onnxruntime.quantization import QuantType, quantize_dynamic

        _remove_rejected_quantization(output)
        quantized_directory = output / "int8"
        quantized_directory.mkdir(parents=True, exist_ok=True)
        quantized_destination = quantized_directory / "prollama_encoder_int8.onnx"
        quantize_dynamic(
            destination,
            quantized_destination,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            use_external_data_format=True,
        )
        quantized_session = _low_memory_session(quantized_destination)
        quantized_output = quantized_session.run(
            ["last_hidden_state"],
            feeds,
        )[0].astype(np.float32)
        quantized_max_absolute_error = float(
            np.max(np.abs(reference - quantized_output))
        )
        if not np.allclose(reference, quantized_output, rtol=0.15, atol=0.15):
            del quantized_session
            del quantized_output
            gc.collect()
            _remove_rejected_quantization(output)
            quantization_status = "rejected_parity"
            print(
                "WARNING: INT8 ProLLaMA was rejected and removed because its "
                f"parity error was {quantized_max_absolute_error}. "
                "The valid FP32 ONNX model will be used instead.",
                flush=True,
            )
        else:
            final_destination = quantized_destination
            quantization_status = "accepted"

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
                "quantized": quantization_status == "accepted",
                "quantization_status": quantization_status,
                "quantization_configuration": {
                    "weight_type": "QInt8",
                    "per_channel": True,
                    "reduce_range": True,
                    "op_types": ["MatMul"],
                }
                if quantize
                else None,
                "parity_check_skipped": skip_parity_check,
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
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help=(
            "Do not initialize ONNX Runtime after export. Use this on memory-limited "
            "hosts, then validate in a fresh process."
        ),
    )
    args = parser.parse_args()
    print(
        export_prollama_feature_onnx(
            model_id=args.model_id,
            output_directory=args.output,
            sequence_length=args.sequence_length,
            opset=args.opset,
            dtype=args.dtype,
            quantize=args.quantize,
            skip_parity_check=args.skip_parity_check,
        )
    )


if __name__ == "__main__":
    main()
