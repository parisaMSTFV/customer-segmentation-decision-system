"""Leakage-safe customer feature transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_segmentation.schema import FEATURE_COLUMNS

LOG_COLUMNS = (
    "recency_days",
    "orders_12m",
    "revenue_12m",
    "sessions_90d",
    "category_breadth_12m",
)

RFM_COLUMNS = ("recency_days", "orders_12m", "revenue_12m")


def prepare_features(
    frame: pd.DataFrame, columns: tuple[str, ...] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Select model features and log-transform skewed nonnegative columns."""
    transformed = frame.loc[:, columns].astype(float).copy()
    for column in set(columns).intersection(LOG_COLUMNS):
        transformed[column] = np.log1p(transformed[column])
    return transformed
