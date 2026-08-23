from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.modeling import (
    governed_candidate_is_supported,
    mean_pairwise_ari,
    select_cluster_count,
)


def test_pairwise_ari_is_permutation_invariant() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    assert mean_pairwise_ari([labels, permuted]) == 1.0


def test_pairwise_ari_requires_two_label_sets() -> None:
    with pytest.raises(ValueError, match="at least two"):
        mean_pairwise_ari([np.array([0, 1])])


def test_selection_excludes_ineligible_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "cluster_count": [4, 6],
            "selection_score": [0.9, 0.8],
            "eligible": [False, True],
        }
    )
    assert select_cluster_count(candidates) == 6


def test_selection_prefers_smaller_k_on_exact_tie() -> None:
    candidates = pd.DataFrame(
        {
            "cluster_count": [5, 4],
            "selection_score": [0.8, 0.8],
            "eligible": [True, True],
        }
    )
    assert select_cluster_count(candidates) == 4


def test_selection_retains_supported_governed_count_within_tolerance() -> None:
    candidates = pd.DataFrame(
        {
            "cluster_count": [3, 6, 7],
            "selection_score": [0.79, 0.77, 0.80],
            "eligible": [True, True, True],
        }
    )
    assert select_cluster_count(candidates, governed_cluster_count=6, selection_tolerance=0.03) == 6
    assert governed_candidate_is_supported(candidates, 6, 0.03) is True
    assert governed_candidate_is_supported(candidates, 6, 0.01) is False
