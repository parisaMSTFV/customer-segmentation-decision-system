from __future__ import annotations

import pytest

from customer_segmentation.schema import DataValidationError, validate_customer_features
from customer_segmentation.synthetic import generate_customers


def test_schema_accepts_generated_features() -> None:
    features, _ = generate_customers(700, 7)
    assert validate_customer_features(features) == 8


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
