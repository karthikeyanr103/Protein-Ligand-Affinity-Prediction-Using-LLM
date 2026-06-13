import pandas as pd

from affinity.data import assign_splits, profile_dataset


def test_cold_protein_split_has_no_protein_overlap():
    frame = pd.DataFrame(
        {
            "protein_sequence": ["ACDE"] * 3 + ["FGHI"] * 3 + ["KLMN"] * 3,
            "compound_smiles": [f"C{i}" for i in range(9)],
            "label": [float(i) for i in range(9)],
        }
    )
    splits = assign_splits(frame, "cold_protein", 0.6, 0.2, seed=7)
    frame["split"] = splits
    memberships = frame.groupby("protein_sequence")["split"].nunique()
    assert memberships.max() == 1


def test_profile_reports_lengths_and_duplicates():
    frame = pd.DataFrame(
        {
            "protein_sequence": ["ACDE", "ACDE", "FGHIK"],
            "compound_smiles": ["CCO", "CCO", "N"],
            "label": [5.0, 5.1, 7.0],
        }
    )
    profile = profile_dataset(frame)
    assert profile.rows == 3
    assert profile.protein_length_min == 4
    assert profile.protein_length_max == 5
    assert profile.duplicate_pairs == 1

