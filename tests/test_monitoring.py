from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.modeling import QuantileClipper
from customer_segmentation.monitoring import build_migration_matrix, segment_share_psi


def test_quantile_clipper_reports_affected_customers() -> None:
    values = pd.DataFrame({"x": [0.0, 1.0, 2.0, 100.0], "y": [1.0, 1.0, 1.0, 1.0]})
    clipper = QuantileClipper(lower_quantile=0.0, upper_quantile=0.75).fit(values)
    transformed = clipper.transform(values)
    assert transformed[-1, 0] < 100
    assert clipper.clipped_customer_share(values) == 0.25


def test_segment_share_psi_detects_population_shift() -> None:
    reference = {"A": 0.5, "B": 0.5}
    assert np.isclose(segment_share_psi(reference, pd.Series(["A", "B"])), 0)
    assert segment_share_psi(reference, pd.Series(["A"] * 9 + ["B"])) > 0.5


def test_migration_matrix_reconciles_overlapping_customers() -> None:
    previous = pd.DataFrame({"customer_id": ["1", "2", "3"], "segment_name": ["A", "A", "B"]})
    current = pd.DataFrame({"customer_id": ["1", "2", "3"], "segment_name": ["A", "B", "B"]})
    matrix, metrics = build_migration_matrix(current, previous)
    assert set(matrix["segment_name_previous"]) == {"A", "B"}
    assert metrics == {"overlapping_customers": 3, "migration_rate": pytest.approx(1 / 3)}


def test_migration_matrix_rejects_nonoverlapping_customers() -> None:
    current = pd.DataFrame({"customer_id": ["1"], "segment_name": ["A"]})
    previous = pd.DataFrame({"customer_id": ["2"], "segment_name": ["B"]})
    with pytest.raises(ValueError, match="no overlapping"):
        build_migration_matrix(current, previous)
