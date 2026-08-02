from __future__ import annotations

import numpy as np
import pandas as pd

from customer_segmentation.features import RFM_COLUMNS, prepare_features
from customer_segmentation.schema import FEATURE_COLUMNS
from customer_segmentation.synthetic import generate_customers


def test_prepare_features_uses_only_declared_columns() -> None:
    frame, _ = generate_customers(700, 7)
    transformed = prepare_features(frame)
    assert tuple(transformed.columns) == FEATURE_COLUMNS


def test_log_transform_is_applied_without_mutation() -> None:
    frame, _ = generate_customers(700, 7)
    original = frame.copy()
    transformed = prepare_features(frame, RFM_COLUMNS)
    assert np.isclose(transformed.iloc[0]["revenue_12m"], np.log1p(frame.iloc[0]["revenue_12m"]))
    pd.testing.assert_frame_equal(frame, original)
