---
license: apache-2.0
library_name: onnx
pipeline_tag: tabular-regression
tags:
  - protein
  - molecule
  - drug-discovery
  - onnx
---

# Protein-Compound Affinity Fusion Regressor

This repository trains a compact regression head over cached ProLLaMA and Mol-LLaMA embeddings.
Inference runs ProLLaMA ONNX and Mol-LLaMA ONNX before invoking the affinity ONNX head.

## Intended use

Portfolio, educational, and research prototyping. It is not validated for medicinal chemistry
decisions, clinical use, or safety-critical screening.

## Inputs and output

- Protein sequence containing the 20 standard amino-acid symbols.
- Compound represented as a valid SMILES string.
- One numeric affinity prediction on the scale used by the competition `label`.

## Evaluation

The primary split is cold-protein to prevent the same protein from appearing in train and test.
Final metrics are written into `metadata.json` by the training command.

## Limitations

The competition label's exact physical definition and units must be confirmed from the Kaggle
competition documentation. A quantized 7B ProLLaMA ONNX graph remains large and CPU inference can
be slow or exceed free-host memory. Generated molecule conformers are plausible geometries, not
experimentally determined binding poses. Protein 3D rendering requires an uploaded PDB structure.
