from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _safe_index(values: list, value) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values) - 1


def _atom_feature_vector(atom: Chem.Atom) -> list[int]:
    chirality = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "misc",
    ]
    return [
        _safe_index(list(range(1, 119)) + ["misc"], atom.GetAtomicNum()),
        _safe_index(chirality, atom.GetChiralTag()),
        _safe_index(list(range(0, 11)) + ["misc"], atom.GetTotalDegree()),
        _safe_index(list(range(-5, 6)) + ["misc"], atom.GetFormalCharge()),
        _safe_index(list(range(0, 9)) + ["misc"], atom.GetTotalNumHs()),
        _safe_index(
            list(range(0, 5)) + ["misc"],
            atom.GetNumRadicalElectrons(),
        ),
        _safe_index(hybridization, atom.GetHybridization()),
        [False, True].index(atom.GetIsAromatic()),
        [False, True].index(atom.IsInRing()),
    ]


def _bond_feature_vector(bond: Chem.Bond) -> list[int]:
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        "misc",
    ]
    stereo = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
    return [
        _safe_index(bond_types, bond.GetBondType()),
        _safe_index(stereo, bond.GetStereo()),
        [False, True].index(bond.GetIsConjugated()),
    ]


def _find_onnx_model(directory: str | Path, preferred: str = "") -> Path:
    directory = Path(directory)
    if preferred and (directory / preferred).exists():
        return directory / preferred
    candidates = sorted(directory.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX graph found under {directory}")
    return candidates[0]


class ProLLaMAOnnxEmbedder:
    def __init__(
        self,
        model_directory: str | Path,
        tokenizer_id: str = "GreatCaptainNemo/ProLLaMA",
        max_length: int = 1536,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        directory = Path(model_directory)
        self.max_length = max_length
        tokenizer_source = (
            directory if (directory / "tokenizer_config.json").exists() else tokenizer_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        metadata_path = directory / "export_metadata.json"
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.session = None
        self.hidden_output = ""
        model_paths = sorted(directory.rglob("*.onnx"))
        model_paths.sort(key=lambda path: "int8" not in path.name.lower())
        for model_path in model_paths:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            output_names = {output.name for output in session.get_outputs()}
            hidden_output = next(
                (
                    name
                    for name in (
                        "last_hidden_state",
                        "hidden_states",
                        "sentence_embedding",
                    )
                    if name in output_names
                ),
                "",
            )
            if hidden_output:
                self.session = session
                self.hidden_output = hidden_output
                break
        if not self.hidden_output:
            raise ValueError(
                "The ProLLaMA ONNX graph exposes no hidden-state output. Export it as "
                "feature-extraction; a text-generation graph that only returns logits cannot "
                "reproduce the training embedding."
            )

    def encode(self, sequences: list[str]) -> np.ndarray:
        prompts = [f"[Determine superfamily] Seq=<{sequence}>" for sequence in sequences]
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feeds: dict[str, np.ndarray] = {}
        for input_info in self.session.get_inputs():
            if input_info.name in tokens:
                feeds[input_info.name] = tokens[input_info.name].astype(np.int64)
            elif input_info.name == "position_ids":
                batch, length = tokens["input_ids"].shape
                feeds[input_info.name] = np.broadcast_to(
                    np.arange(length, dtype=np.int64)[None, :],
                    (batch, length),
                ).copy()
        hidden = self.session.run([self.hidden_output], feeds)[0]
        if hidden.ndim == 2:
            return hidden.astype(np.float32)
        mask = tokens["attention_mask"][..., None].astype(np.float32)
        return ((hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)).astype(
            np.float32
        )


class MolLLaMAOnnxPreprocessor:
    def __init__(self, dictionary_path: str | Path, conformer_seed: int = 42) -> None:
        payload = json.loads(Path(dictionary_path).read_text(encoding="utf-8"))
        self.symbol_to_index = {
            symbol: index for index, symbol in enumerate(payload["symbols"])
        }
        self.bos = int(payload["bos_index"])
        self.eos = int(payload["eos_index"])
        self.pad = int(payload["pad_index"])
        self.unk = int(payload["unk_index"])
        self.dictionary_size = len(payload["symbols"])
        self.conformer_seed = conformer_seed

    def _conformer(self, smiles: str) -> tuple[Chem.Mol, np.ndarray]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid SMILES")
        heavy_atoms = molecule.GetNumAtoms()
        molecule = Chem.AddHs(molecule)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = self.conformer_seed
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            raise ValueError("RDKit could not generate a 3D conformer")
        try:
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
        except ValueError:
            pass
        molecule = Chem.RemoveHs(molecule)
        if molecule.GetNumAtoms() != heavy_atoms:
            raise ValueError("Hydrogen removal changed the heavy-atom graph")
        coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)
        return molecule, coordinates

    @staticmethod
    def _graph_features(molecule: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_features = np.asarray(
            [_atom_feature_vector(atom) for atom in molecule.GetAtoms()],
            dtype=np.int64,
        )
        edges: list[tuple[int, int]] = []
        edge_features: list[list[int]] = []
        for bond in molecule.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = _bond_feature_vector(bond)
            edges.extend([(begin, end), (end, begin)])
            edge_features.extend([feature, feature])
        edge_index = (
            np.asarray(edges, dtype=np.int64).T
            if edges
            else np.empty((2, 0), dtype=np.int64)
        )
        edge_attributes = (
            np.asarray(edge_features, dtype=np.int64)
            if edge_features
            else np.empty((0, 3), dtype=np.int64)
        )
        return node_features, edge_index, edge_attributes

    def prepare(self, smiles: str) -> dict[str, np.ndarray]:
        molecule, coordinates = self._conformer(smiles)
        atom_ids = [
            self.symbol_to_index.get(atom.GetSymbol(), self.unk)
            for atom in molecule.GetAtoms()
        ]
        tokens = np.asarray([self.bos, *atom_ids, self.eos], dtype=np.int64)
        centered = coordinates - coordinates.mean(axis=0)
        padded_coordinates = np.concatenate(
            [np.zeros((1, 3), np.float32), centered, np.zeros((1, 3), np.float32)]
        )
        distances = np.linalg.norm(
            padded_coordinates[:, None, :] - padded_coordinates[None, :, :],
            axis=-1,
        ).astype(np.float32)
        edge_types = (
            tokens[:, None] * self.dictionary_size + tokens[None, :]
        ).astype(np.int64)
        padded_length = ((len(tokens) + 7) // 8) * 8
        src_tokens = np.full((1, padded_length), self.pad, dtype=np.int64)
        src_distance = np.zeros((1, padded_length, padded_length), dtype=np.float32)
        src_edge_type = np.zeros((1, padded_length, padded_length), dtype=np.int64)
        src_tokens[0, : len(tokens)] = tokens
        src_distance[0, : len(tokens), : len(tokens)] = distances
        src_edge_type[0, : len(tokens), : len(tokens)] = edge_types
        node_features, edge_index, edge_features = self._graph_features(molecule)
        return {
            "src_tokens": src_tokens,
            "src_distance": src_distance,
            "src_edge_type": src_edge_type,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }


class MolLLaMAOnnxEmbedder:
    def __init__(self, model_directory: str | Path) -> None:
        import onnxruntime as ort

        directory = Path(model_directory)
        metadata_path = directory / "export_metadata.json"
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        self.preprocessor = MolLLaMAOnnxPreprocessor(
            directory / "unimol_dictionary.json",
            conformer_seed=int(self.metadata.get("conformer_seed", 42)),
        )
        preferred = (
            "mol_llama_encoder_int8.onnx"
            if (directory / "mol_llama_encoder_int8.onnx").exists()
            else "mol_llama_encoder.onnx"
        )
        self.session = ort.InferenceSession(
            str(_find_onnx_model(directory, preferred)),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, smiles_values: list[str]) -> np.ndarray:
        outputs = []
        for smiles in smiles_values:
            feeds = self.preprocessor.prepare(smiles)
            expected = {item.name for item in self.session.get_inputs()}
            feeds = {name: value for name, value in feeds.items() if name in expected}
            outputs.append(
                self.session.run(["molecule_embedding"], feeds)[0][0]
            )
        return np.asarray(outputs, dtype=np.float32)
