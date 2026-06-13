# Experiment protocol

## Objective

Predict the numeric `label` from a protein amino-acid sequence and a compound SMILES string.
Report MAE, RMSE, R-squared, and Pearson correlation.

## Data quality and leakage

The supplied training CSV contains many repeated proteins and compounds. A random row split lets
the same protein appear in training and testing, which can substantially overstate generalization.
The primary result therefore uses `cold_protein`: every protein belongs to exactly one split.

Run two secondary experiments:

- `cold_compound` to measure generalization to unseen molecules.
- `pair` to compare against a less strict unique-pair split.

Exact duplicate protein-compound pairs should be investigated before final training. If duplicate
pairs have conflicting labels, retain them only with a documented aggregation policy such as the
median label.

## Training

1. Fit preprocessing statistics on the training split only.
2. Cache frozen domain embeddings by unique protein/compound identifier.
3. Train the fusion MLP with AdamW and MSE loss.
4. Select the checkpoint with the lowest validation RMSE.
5. Stop after the configured number of stale validation epochs.

## Validation

Use validation data for checkpoint selection and hyperparameter decisions only. Track RMSE as the
primary optimization metric and MAE, R-squared, and Pearson correlation as supporting metrics.

## Testing

Evaluate the selected checkpoint exactly once on the held-out test split. Save split assignments,
normalization parameters, model metadata, and the final metrics with the model artifact.

## Inference

Validate amino-acid and SMILES inputs, run the ProLLaMA and Mol-LLaMA ONNX encoders in the exact
training order, apply saved normalization, and execute the third ONNX fusion graph. The exported
models must preserve the same IDs, prompt, pooling, truncation, molecular graph construction, and
Uni-Mol dictionary used to build the training caches.
