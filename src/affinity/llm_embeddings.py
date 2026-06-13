from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np

from affinity.data import load_dataset
from affinity.features import save_embedding_table


def _mean_pool(hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class ProLLaMAEmbedder:
    def __init__(
        self,
        model_id: str = "GreatCaptainNemo/ProLLaMA",
        max_length: int = 1536,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            if load_in_4bit
            else None
        )
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            trust_remote_code=True,
        )
        self.model.eval()

    def encode(self, sequences: Iterable[str], batch_size: int = 1) -> np.ndarray:
        import torch

        values = list(sequences)
        results: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            prompts = [
                f"[Determine superfamily] Seq=<{value}>"
                for value in values[start : start + batch_size]
            ]
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model(**tokens, output_hidden_states=True, use_cache=False)
            pooled = _mean_pool(output.hidden_states[-1], tokens["attention_mask"])
            results.append(pooled.float().cpu().numpy())
        return np.concatenate(results).astype(np.float32)


def extract_causal_lm_embeddings(
    values: Iterable[str],
    model_id: str,
    prefix: str,
    batch_size: int = 1,
    max_length: int = 1024,
    load_in_4bit: bool = True,
) -> np.ndarray:
    embedder = ProLLaMAEmbedder(
        model_id,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
    )
    if prefix != "[Determine superfamily] Seq=<{value}>":
        raise ValueError("ProLLaMA extraction requires its protein understanding prompt")
    return embedder.encode(values, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen ProLLaMA or compatible causal-LM embeddings"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--column", choices=["protein_sequence"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    frame = load_dataset(args.data)
    values = frame[args.column].drop_duplicates().tolist()
    prefix = "[Determine superfamily] Seq=<{value}>"
    embeddings = extract_causal_lm_embeddings(
        values,
        args.model_id,
        prefix,
        args.batch_size,
        args.max_length,
        not args.no_4bit,
    )
    save_embedding_table(
        args.output,
        values,
        embeddings,
        args.model_id,
        settings={
            "prompt": "[Determine superfamily] Seq=<{value}>",
            "pooling": "attention_masked_mean_last_hidden_state",
            "max_length": args.max_length,
        },
    )


if __name__ == "__main__":
    main()
