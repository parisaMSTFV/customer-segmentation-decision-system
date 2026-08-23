from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from customer_segmentation.external import (
    fit_external_segmentation,
    run_external_segmentation,
    score_external_segmentation,
)
from customer_segmentation.synthetic import generate_customers


def test_external_segmentation_writes_guarded_fit_schema(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    input_path = tmp_path / "customer_features.csv"
    output_dir = tmp_path / "output"
    observations.to_csv(input_path, index=False)

    metrics = run_external_segmentation(input_path, output_dir)
    assignments = pd.read_csv(output_dir / "customer_segments.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert assignments.columns.tolist() == [
        "customer_id",
        "snapshot_date",
        "cluster_id",
        "segment_name",
        "segment_definition_id",
        "assignment_status",
    ]
    assert len(assignments) == 900
    assert assignments["segment_name"].nunique() == 6
    assert assignments["segment_definition_id"].nunique() == 1
    assert metrics == metadata
    assert metadata["quality_gate_passed"] is True
    assert metadata["campaign_impact"] == "Not evaluated"
    assert (output_dir / "model" / "model_manifest.json").is_file()
    assert (output_dir / "segment_profiles.csv").is_file()


def test_fit_definition_is_row_order_invariant_and_score_is_frozen(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    first_input = tmp_path / "first.csv"
    reordered_input = tmp_path / "reordered.csv"
    observations.to_csv(first_input, index=False)
    observations.sample(frac=1, random_state=99).to_csv(reordered_input, index=False)

    first = fit_external_segmentation(first_input, tmp_path / "model-a", tmp_path / "fit-a")
    reordered = fit_external_segmentation(reordered_input, tmp_path / "model-b")
    scored = score_external_segmentation(
        reordered_input,
        tmp_path / "model-a",
        tmp_path / "score",
        tmp_path / "fit-a" / "customer_segments.csv",
    )

    assert first["segment_definition_id"] == reordered["segment_definition_id"]
    assert first["segment_definition_id"] == scored["segment_definition_id"]
    assert scored["monitoring_gate_passed"] is True
    assert scored["migration"] == {"overlapping_customers": 900, "migration_rate": 0.0}


def test_external_fit_rejects_too_few_or_degenerate_customers(tmp_path) -> None:
    observations, _truth = generate_customers(700, 17)
    too_small = tmp_path / "too_small.csv"
    observations.head(599).to_csv(too_small, index=False)
    with pytest.raises(ValueError, match="at least 600 customers"):
        fit_external_segmentation(too_small, tmp_path / "small-model")

    repeated = pd.concat(
        [observations.head(1)] * 595
        + [observations.iloc[index : index + 1] for index in range(1, 6)],
        ignore_index=True,
    )
    repeated["customer_id"] = [f"DEGENERATE-{index:04d}" for index in range(len(repeated))]
    degenerate = tmp_path / "degenerate.csv"
    repeated.to_csv(degenerate, index=False)
    with pytest.warns(ConvergenceWarning):
        with pytest.raises(ValueError, match="governed cluster count|quality guardrails"):
            fit_external_segmentation(degenerate, tmp_path / "degenerate-model")


def test_external_score_rejects_snapshot_before_training(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    training = tmp_path / "training.csv"
    observations.to_csv(training, index=False)
    fit_external_segmentation(training, tmp_path / "model")

    shifted = observations.copy()
    shifted["snapshot_date"] = "2026-02-01"
    shifted_values = {
        "recency_days": 1,
        "orders_12m": 40,
        "revenue_12m": 5000,
        "margin_rate": 0.4,
        "sessions_90d": 100,
        "conversion_rate_90d": 0.4,
        "discount_order_share": 0.05,
        "category_breadth_12m": 15,
        "return_rate": 0.01,
        "satisfaction_score": 5.0,
    }
    for column, value in shifted_values.items():
        shifted[column] = value
    shifted_path = tmp_path / "shifted.csv"
    shifted.to_csv(shifted_path, index=False)
    held = score_external_segmentation(shifted_path, tmp_path / "model", tmp_path / "held")
    assert held["monitoring_gate_passed"] is False
    assert held["assignment_status"] == "review_required"

    observations["snapshot_date"] = "2025-12-31"
    earlier = tmp_path / "earlier.csv"
    observations.to_csv(earlier, index=False)
    with pytest.raises(ValueError, match="cannot precede"):
        score_external_segmentation(earlier, tmp_path / "model", tmp_path / "score")
