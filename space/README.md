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

CPU inference through three ONNX graphs: protein encoding, molecule encoding, and affinity
regression. The default pair is ESM-2 plus MoLFormer. Configure `PROTEIN_ONNX_REPO`,
`MOLECULE_ONNX_REPO`, `AFFINITY_MODEL_REPO`, and optionally `HF_TOKEN`.

The first prediction downloads and initializes all three ONNX repositories. Subsequent predictions
reuse the cached sessions.
