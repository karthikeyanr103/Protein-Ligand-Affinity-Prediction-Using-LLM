from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local ONNX Gradio application")
    parser.add_argument("--prollama", required=True, help="ProLLaMA ONNX directory")
    parser.add_argument("--mol-llama", required=True, help="Mol-LLaMA ONNX directory")
    parser.add_argument("--affinity", required=True, help="Affinity ONNX directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.environ["PROLLAMA_ONNX_PATH"] = args.prollama
    os.environ["MOL_LLAMA_ONNX_PATH"] = args.mol_llama
    os.environ["AFFINITY_MODEL_PATH"] = args.affinity

    from space.app import demo

    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
