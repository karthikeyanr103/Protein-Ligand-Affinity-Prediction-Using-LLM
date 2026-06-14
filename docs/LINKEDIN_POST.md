# LinkedIn Launch Post

## Recommended post

🧬 I built and deployed an end-to-end protein-compound affinity prediction project.

The application takes a protein amino-acid sequence and a compound SMILES string, extracts
representations with pretrained scientific language models, and predicts an affinity score.

### What I used

🔹 ESM-2 for protein sequence embeddings  
🔹 MoLFormer for molecular SMILES embeddings  
🔹 A trained MLP regression head for affinity prediction  
🔹 ONNX Runtime for all production inference  
🔹 Gradio and Docker on Hugging Face Spaces  
🔹 Colab/Kaggle notebooks for export, embedding extraction, training, validation, and testing

The deployed application runs on CPU without AWS or a paid GPU endpoint. It also includes:

✅ protein and SMILES validation  
✅ molecule descriptors and 2D rendering  
✅ generated molecular conformer projections  
✅ protein sequence analysis and optional PDB visualization  
✅ one-click example predictions  
✅ cold-protein train/validation/test splitting  
✅ MAE, RMSE, R², and Pearson evaluation

One of the most useful parts of this project was learning how to handle large ONNX exports,
external tensor files, numerical parity checks, memory constraints, and container-native
dependencies.

🚀 Live demo:  
https://huggingface.co/spaces/IAmKarthik/protein-compound-affinity

💻 GitHub repository:  
https://github.com/karthikeyanr103/Protein-Ligand-Affinity-Prediction-Using-LLM

📊 Dataset:  
https://www.kaggle.com/competitions/protein-compound-affinity

This is a research and portfolio demonstration, not a replacement for docking, experimental
binding assays, or clinical validation.

#MachineLearning #Bioinformatics #Cheminformatics #DrugDiscovery #ProteinLanguageModel
#MolecularMachineLearning #ONNX #HuggingFace #Python #DeepLearning #MLOps

## How to publish it

1. Open the live Hugging Face Space and take one clean screenshot showing an example prediction.
2. Create a LinkedIn post and upload the screenshot before pasting the text above.
3. Keep the first three lines visible before LinkedIn's **see more** break.
4. Add the live demo, GitHub repository, and dataset links as plain URLs.
5. Tag Hugging Face, ONNX, Gradio, Kaggle, and the model authors only when relevant.
6. Publish when you can respond to technical questions during the next few hours.
7. Add the project to the **Featured** section of your LinkedIn profile using the live demo URL.

## Suggested screenshot caption

> CPU-only ONNX inference combining ESM-2 protein embeddings, MoLFormer molecular embeddings, and
> a trained affinity regression head.

## Short version

🧬 New portfolio project: Protein-Compound Affinity Prediction with ESM-2, MoLFormer, and ONNX.

I built the complete workflow from encoder export and embedding extraction through training,
evaluation, ONNX conversion, and CPU deployment on Hugging Face Spaces.

🚀 Demo: https://huggingface.co/spaces/IAmKarthik/protein-compound-affinity  
💻 Code: https://github.com/karthikeyanr103/Protein-Ligand-Affinity-Prediction-Using-LLM

#Bioinformatics #Cheminformatics #DrugDiscovery #ONNX #HuggingFace #MachineLearning
