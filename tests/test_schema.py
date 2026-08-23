from __future__ import annotations

import pytest

from customer_segmentation.schema import DataValidationError, validate_customer_features
from customer_segmentation.synthetic import generate_customers


def test_schema_accepts_generated_features() -> None:
    features, _ = generate_customers(700, 7)
    assert validate_customer_features(features) == 11


def test_schema_rejects_truth_leakage() -> None:
    features, truth = generate_customers(700, 7)
    leaked = features.merge(truth, on="customer_id")
    with pytest.raises(DataValidationError, match="must not appear"):
        validate_customer_features(leaked)


def test_schema_rejects_duplicate_customer() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[1, "customer_id"] = features.loc[0, "customer_id"]
    with pytest.raises(DataValidationError, match="unique"):
        validate_customer_features(features)


def test_schema_rejects_invalid_rate() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "return_rate"] = 1.2
    with pytest.raises(DataValidationError, match="between zero and one"):
        validate_customer_features(features)


def test_schema_rejects_nonfinite_feature() -> None:
    features, _truth = generate_customers(700, 7)
    features.loc[0, "revenue_12m"] = float("inf")
    with pytest.raises(DataValidationError, match="finite numeric"):
        validate_customer_features(features)


def test_schema_accepts_negative_margin() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "margin_rate"] = -0.4
    assert validate_customer_features(features) == 11


def test_schema_rejects_blank_identifier_and_fractional_count() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "customer_id"] = "  "
    with pytest.raises(DataValidationError, match="must not be blank"):
        validate_customer_features(features)
    features.loc[0, "customer_id"] = "valid-id"
    features["orders_12m"] = features["orders_12m"].astype(float)
    features.loc[0, "orders_12m"] = 1.5
    with pytest.raises(DataValidationError, match="whole numbers"):
        validate_customer_features(features)


def test_schema_rejects_mixed_or_invalid_snapshot_dates() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "snapshot_date"] = "2026-02-01"
    with pytest.raises(DataValidationError, match="one snapshot_date"):
        validate_customer_features(features)
    features["snapshot_date"] = "01/02/2026"
    with pytest.raises(DataValidationError, match="ISO"):
        validate_customer_features(features)


def test_schema_rejects_conversion_without_sessions() -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "sessions_90d"] = 0
    features.loc[0, "conversion_rate_90d"] = 0.2
    with pytest.raises(DataValidationError, match="zero conversion"):
        validate_customer_features(features)
