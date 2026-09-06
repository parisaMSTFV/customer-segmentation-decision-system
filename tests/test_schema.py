from __future__ import annotations

import pandas as pd
import pytest

from customer_segmentation.schema import (
    DataValidationError,
    load_customer_snapshot,
    pseudonymize_customer_ids,
    validate_customer_features,
)
from customer_segmentation.synthetic import generate_customers


def test_schema_accepts_generated_features() -> None:
    features, _ = generate_customers(700, 7)
    assert validate_customer_features(features) == 15


def test_schema_rejects_truth_leakage() -> None:
    features, truth = generate_customers(700, 7)
    leaked = features.merge(truth, on="customer_id")
    with pytest.raises(DataValidationError, match="unexpected input columns"):
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
    assert validate_customer_features(features) == 15


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


@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        ("=1+1", "spreadsheet-formula"),
        ("customer\n1", "control character"),
        (" customer-1", "surrounding whitespace"),
        ("x" * 129, "128 characters"),
    ],
)
def test_schema_rejects_unsafe_identifiers(identifier: str, message: str) -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "customer_id"] = identifier
    with pytest.raises(DataValidationError, match=message):
        validate_customer_features(features)


def test_snapshot_loader_preserves_leading_zeroes_and_rejects_extra_columns(tmp_path) -> None:
    features, _ = generate_customers(700, 7)
    features.loc[0, "customer_id"] = "001"
    path = tmp_path / "features.csv"
    features.to_csv(path, index=False)
    loaded, checks, _snapshot = load_customer_snapshot(path)
    assert loaded.loc[0, "customer_id"] == "001"
    assert checks == 15

    features["email"] = "synthetic@example.invalid"
    features.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="unexpected input columns"):
        load_customer_snapshot(path)


def test_pseudonyms_are_stable_and_do_not_expose_raw_ids() -> None:
    frame = pd.DataFrame({"customer_id": ["001", "مشتری-۲"]})
    first = pseudonymize_customer_ids(frame, "test-only-salt-12345")
    second = pseudonymize_customer_ids(frame, "test-only-salt-12345")
    pd.testing.assert_frame_equal(first, second)
    assert first["customer_id"].str.startswith("cus_").all()
    assert not set(first["customer_id"]).intersection(frame["customer_id"])
    with pytest.raises(DataValidationError, match="at least 16"):
        pseudonymize_customer_ids(frame, "short")
