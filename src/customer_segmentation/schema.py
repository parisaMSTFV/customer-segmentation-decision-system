"""Input schema checks for synthetic customer features."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "recency_days",
    "orders_12m",
    "revenue_12m",
    "margin_rate",
    "sessions_90d",
    "conversion_rate_90d",
    "discount_order_share",
    "category_breadth_12m",
    "return_rate",
    "satisfaction_score",
)


class DataValidationError(ValueError):
    """Raised when customer features violate the public schema."""


def validate_customer_features(frame: pd.DataFrame) -> int:
    """Validate identifiers, feature presence, nulls, and bounded rates."""
    required = {"customer_id", *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    if frame["customer_id"].duplicated().any():
        raise DataValidationError("customer_id must be unique")
    if frame[list(required)].isna().any().any():
        raise DataValidationError("required columns must not contain null values")
    numeric_features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any() or not np.isfinite(numeric_features).all().all():
        raise DataValidationError("features must contain finite numeric values")
    forbidden = {"synthetic_persona", "segment", "label"}.intersection(frame.columns)
    if forbidden:
        raise DataValidationError(
            f"truth or label columns must not appear in model input: {sorted(forbidden)}"
        )
    nonnegative = [
        "recency_days",
        "orders_12m",
        "revenue_12m",
        "sessions_90d",
        "category_breadth_12m",
    ]
    if (numeric_features[nonnegative] < 0).any().any():
        raise DataValidationError("count and monetary features must be nonnegative")
    for column in ("margin_rate", "conversion_rate_90d", "discount_order_share", "return_rate"):
        if not numeric_features[column].between(0, 1).all():
            raise DataValidationError(f"{column} must be between zero and one")
    if not numeric_features["satisfaction_score"].between(1, 5).all():
        raise DataValidationError("satisfaction_score must be between one and five")
    return 8
