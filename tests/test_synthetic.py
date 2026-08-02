from __future__ import annotations

import pandas as pd

from customer_segmentation.synthetic import PERSONAS, generate_customers


def test_generation_is_deterministic() -> None:
    left_features, left_truth = generate_customers(900, 42)
    right_features, right_truth = generate_customers(900, 42)
    pd.testing.assert_frame_equal(left_features, right_features)
    pd.testing.assert_frame_equal(left_truth, right_truth)


def test_truth_is_separate_and_aligned() -> None:
    features, truth = generate_customers(900, 42)
    assert "synthetic_persona" not in features.columns
    assert features["customer_id"].tolist() == truth["customer_id"].tolist()
    assert set(truth["synthetic_persona"]) == {persona.name for persona in PERSONAS}


def test_generated_rates_respect_bounds() -> None:
    features, _ = generate_customers(900, 42)
    for column in ("margin_rate", "conversion_rate_90d", "discount_order_share", "return_rate"):
        assert features[column].between(0, 1).all()
    assert features["satisfaction_score"].between(1, 5).all()
