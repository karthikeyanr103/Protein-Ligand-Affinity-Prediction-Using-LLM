from __future__ import annotations

import argparse
from pathlib import Path


def export_prollama_feature_onnx(model_id: str, output_directory: str | Path) -> Path:
    """Export the base ProLLaMA transformer with a last_hidden_state output."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_id,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ProLLaMA as ONNX feature extraction"
    )
    parser.add_argument("--model-id", default="GreatCaptainNemo/ProLLaMA")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_prollama_feature_onnx(args.model_id, args.output))


if __name__ == "__main__":
    main()

