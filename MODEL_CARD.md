---
license: apache-2.0
library_name: onnx
pipeline_tag: tabular-regression
tags:
  - protein
  - molecule
  - onnx
  - drug-discovery
---

# Protein-Compound Affinity ONNX Model

This model is the regression head of a three-model pipeline.

Inputs are embeddings produced by:

- `facebook/esm2_t12_35M_UR50D`
- `ibm-research/MoLFormer-XL-both-10pct`

The regression head must be used with the exact encoder exports and preprocessing settings recorded
in `metadata.json`.

Artifacts trained with the optional legacy profile may instead reference
`GreatCaptainNemo/ProLLaMA` and `DongkiKim/Mol-Llama-3.1-8B-Instruct`.

The primary evaluation split is cold-protein. Final metrics are stored in `metadata.json`.

This is a research project. It is not validated for clinical or medicinal chemistry decisions.
