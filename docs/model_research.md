# Model research

## Mol-LLaMA

- **Weights:** [DongkiKim/Mol-Llama-3.1-8B-Instruct](https://huggingface.co/DongkiKim/Mol-Llama-3.1-8B-Instruct)
- **Paper:** [Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model](https://arxiv.org/abs/2502.13449)
- **Code:** [DongkiKim95/Mol-LLaMA](https://github.com/DongkiKim95/Mol-LLaMA)
- **Backbone:** Meta Llama 3.1 8B Instruct.
- **Molecular stack:** MoleculeSTM 2D encoder, Uni-Mol 3D encoder, cross-attention
  blending module, and a SciBERT-based Q-Former.
- **Fine-tuning:** LoRA adapters are used for instruction tuning.
- **Training data:** Mol-LLaMA-Instruct, organized around detailed structural descriptions,
  structure-to-feature relationships, and comprehensive molecule conversations.

The Hugging Face repository contains Mol-LLaMA-specific weights/projectors rather than a small,
standalone 0.2B predictor. Reproducing the complete molecular pathway requires the official code,
the gated Llama backbone, graph inputs, and 3D conformers.

## ProLLaMA

- **Weights:** [GreatCaptainNemo/ProLLaMA](https://huggingface.co/GreatCaptainNemo/ProLLaMA)
- **Paper:** [ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing](https://arxiv.org/abs/2402.16445)
- **Code:** [PKU-YuanGroup/ProLLaMA](https://github.com/PKU-YuanGroup/ProLLaMA)
- **Backbone:** Llama-2-7B.
- **Training approach:** Protein Vocabulary Pruning followed by continual pretraining and
  instruction tuning.
- **Training data:** The paper reports a multi-task instruction corpus of about 13 million
  examples with protein superfamily information.
- **Original tasks:** Unconditional generation, controllable generation, and protein superfamily
  understanding. Affinity regression is a new downstream task in this repository.

## How they are used here

The Kaggle workflow treats both models as frozen domain feature extractors:

1. Deduplicate the 2,665 protein sequences and extract one ProLLaMA representation per sequence.
2. Deduplicate the compounds before molecular extraction.
3. Cache both embedding tables as compressed NPZ files.
4. Join embeddings to labeled pairs and train a compact fusion regressor.
5. Export only the regression head to ONNX.
6. Export both embedding paths to ONNX and run three ONNX graphs during inference.

This avoids repeatedly running two multi-billion-parameter models for every duplicated pair. It
keeps training practical while preserving model consistency for new inputs. Mol-LLaMA deployment
exports only its molecular stack because its Llama decoder is not used to create the Q-Former
embedding.

## Practical limitation

The linked community converter may export ProLLaMA as causal text generation. That graph is usable
only if it exposes `last_hidden_state`; logits cannot replace the training embedding. Mol-LLaMA is
exported manually with explicit 2D graph and 3D Uni-Mol tensor inputs. Even quantized, ProLLaMA 7B
is large, so free-Space RAM and latency remain operational risks rather than GPU requirements.
