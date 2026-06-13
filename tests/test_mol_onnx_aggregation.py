import torch


def test_incidence_aggregation_matches_index_add_with_duplicate_targets():
    messages = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    targets = torch.tensor([1, 1, 0, 1])
    expected = torch.zeros(3, 2).index_add(0, targets, messages)

    node_ids = torch.arange(3, dtype=targets.dtype)
    incidence = node_ids.unsqueeze(1).eq(targets.unsqueeze(0))
    actual = incidence.to(messages.dtype).matmul(messages)

    torch.testing.assert_close(actual, expected)
