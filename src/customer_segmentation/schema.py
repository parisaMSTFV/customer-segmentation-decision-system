"""Input schema checks for synthetic customer features."""

from __future__ import annotations

import numpy as np
import pandas as pd

SNAPSHOT_COLUMN = "snapshot_date"

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
    """Validate identifiers, temporal scope, feature types, bounds, and consistency."""
    required = {"customer_id", SNAPSHOT_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    if frame["customer_id"].duplicated().any():
        raise DataValidationError("customer_id must be unique")
    if frame[list(required)].isna().any().any():
        raise DataValidationError("required columns must not contain null values")
    identifiers = frame["customer_id"].astype("string")
    if identifiers.str.strip().eq("").any():
        raise DataValidationError("customer_id must not be blank")
    snapshot_values = frame[SNAPSHOT_COLUMN].astype("string")
    if snapshot_values.nunique(dropna=False) != 1:
        raise DataValidationError("all rows must use one snapshot_date")
    parsed_snapshot = pd.to_datetime(snapshot_values, format="%Y-%m-%d", errors="coerce")
    if parsed_snapshot.isna().any():
        raise DataValidationError("snapshot_date must use ISO YYYY-MM-DD format")
    numeric_features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any() or not np.isfinite(numeric_features).all().all():
        raise DataValidationError("features must contain finite numeric values")
    forbidden = {"synthetic_persona", "segment", "label"}.intersection(frame.columns)
    if forbidden:
        raise DataValidationError(
            f"truth or label columns must not appear in model input: {sorted(forbidden)}"
        )
    integer_nonnegative = [
        "recency_days",
        "orders_12m",
        "sessions_90d",
        "category_breadth_12m",
    ]
    if (numeric_features[integer_nonnegative] < 0).any().any():
        raise DataValidationError("count features must be nonnegative")
    if not np.equal(numeric_features[integer_nonnegative] % 1, 0).all().all():
        raise DataValidationError("count features must contain whole numbers")
    if (numeric_features["revenue_12m"] < 0).any():
        raise DataValidationError("revenue_12m must be nonnegative")
    if not numeric_features["margin_rate"].between(-1, 1).all():
        raise DataValidationError("margin_rate must be between minus one and one")
    for column in ("conversion_rate_90d", "discount_order_share", "return_rate"):
        if not numeric_features[column].between(0, 1).all():
            raise DataValidationError(f"{column} must be between zero and one")
    if not numeric_features["satisfaction_score"].between(1, 5).all():
        raise DataValidationError("satisfaction_score must be between one and five")
    zero_sessions = numeric_features["sessions_90d"].eq(0)
    if numeric_features.loc[zero_sessions, "conversion_rate_90d"].ne(0).any():
        raise DataValidationError("customers with zero sessions must have zero conversion rate")
    return 11
