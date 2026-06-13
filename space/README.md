---
title: Protein Compound Affinity Explorer
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Protein-Compound Affinity Explorer

CPU inference through three ONNX graphs: ProLLaMA feature extraction, Mol-LLaMA molecular encoding,
and affinity regression. Configure `PROLLAMA_ONNX_REPO`, `MOL_LLAMA_ONNX_REPO`,
`AFFINITY_MODEL_REPO`, and optionally `HF_TOKEN`.
