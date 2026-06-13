# Model release bundle

After the final Kaggle run, place these generated files in this directory:

- `model.onnx`
- `normalization.npz`
- `metadata.json`

Commit them to the release branch or `main`. The `Publish ONNX Model` GitHub Actions workflow
validates and uploads this directory to the Hugging Face model repository configured by
`HF_MODEL_ID`.

Do not publish a model trained only on `data/sample_train.csv` as a final scientific result.

