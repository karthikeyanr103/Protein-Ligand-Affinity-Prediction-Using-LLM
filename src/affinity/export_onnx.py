from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from affinity.model import AffinityRegressor
from affinity.pipeline import load_metadata


def export_onnx(artifact_directory: str, output_path: str = "") -> Path:
    artifact = Path(artifact_directory)
    metadata = load_metadata(artifact / "metadata.json")
    model = AffinityRegressor(
        metadata["input_dim"],
        metadata["hidden_dims"],
        metadata["dropout"],
    )
    model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    destination = Path(output_path) if output_path else artifact / "model.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, metadata["input_dim"], dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(destination),
        input_names=["features"],
        output_names=["affinity"],
        dynamic_axes={"features": {0: "batch"}, "affinity": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    with torch.inference_mode():
        reference = model(example).numpy()
    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )
    exported = session.run(["affinity"], {"features": example.numpy()})[0]
    if not np.allclose(reference, exported, rtol=1e-4, atol=1e-5):
        error = float(np.max(np.abs(reference - exported)))
        raise ValueError(f"Affinity ONNX parity check failed; max error={error}")
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the fusion regressor to ONNX")
    parser.add_argument("--artifacts", default="/content/artifacts/affinity")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    export_onnx(args.artifacts, args.output)


if __name__ == "__main__":
    main()
