from __future__ import annotations

import json
from importlib import resources

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from customer_segmentation.external import (
    fit_external_segmentation,
    run_external_segmentation,
    score_external_segmentation,
)
from customer_segmentation.schema import FEATURE_COLUMNS
from customer_segmentation.synthetic import generate_customers

TEST_SALT = "test-only-salt-12345"


def test_external_segmentation_writes_guarded_fit_schema(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    input_path = tmp_path / "customer_features.csv"
    output_dir = tmp_path / "output"
    observations.to_csv(input_path, index=False)

    metrics = run_external_segmentation(
        input_path,
        output_dir,
        identifier_salt=TEST_SALT,
    )
    assignments = pd.read_csv(output_dir / "customer_segments.csv", dtype="string")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    manifest_text = (output_dir / "model" / "model_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert assignments.columns.tolist() == [
        "customer_id",
        "snapshot_date",
        "cluster_id",
        "segment_name",
        "segment_definition_id",
        "assignment_status",
        "identifier_policy",
    ]
    assert len(assignments) == 900
    assert assignments["segment_name"].nunique() == 6
    assert assignments["segment_definition_id"].nunique() == 1
    assert assignments["identifier_policy"].eq("pseudonymized").all()
    assert assignments["customer_id"].str.startswith("cus_").all()
    assert not set(assignments["customer_id"]).intersection(observations["customer_id"])
    assert metrics == metadata
    assert metadata["quality_gate_passed"] is True
    assert metadata["campaign_impact"] == "Not evaluated"
    assert "source_file" not in manifest
    assert TEST_SALT not in manifest_text
    assert (output_dir / "model" / "model_manifest.json").is_file()
    assert (output_dir / "segment_profiles.csv").is_file()


def test_fit_definition_is_row_order_invariant_and_score_is_frozen(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    first_input = tmp_path / "first.csv"
    reordered_input = tmp_path / "reordered.csv"
    observations.to_csv(first_input, index=False)
    observations.sample(frac=1, random_state=99).to_csv(reordered_input, index=False)

    first = fit_external_segmentation(
        first_input,
        tmp_path / "model-a",
        tmp_path / "fit-a",
        identifier_salt=TEST_SALT,
    )
    reordered = fit_external_segmentation(reordered_input, tmp_path / "model-b")
    scored = score_external_segmentation(
        reordered_input,
        tmp_path / "model-a",
        tmp_path / "score",
        tmp_path / "fit-a" / "customer_segments.csv",
        identifier_salt=TEST_SALT,
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


def test_external_fit_rejects_unstructured_population(tmp_path) -> None:
    observations, _truth = generate_customers(700, 17)
    for index, column in enumerate(FEATURE_COLUMNS):
        observations[column] = (
            observations[column]
            .sample(
                frac=1,
                random_state=100 + index,
            )
            .to_numpy()
        )
    observations.loc[
        observations["sessions_90d"].eq(0),
        "conversion_rate_90d",
    ] = 0
    path = tmp_path / "unstructured.csv"
    observations.to_csv(path, index=False)
    with pytest.raises(ValueError, match="governed cluster count"):
        fit_external_segmentation(path, tmp_path / "unstructured-model")


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
    held = score_external_segmentation(
        shifted_path,
        tmp_path / "model",
        tmp_path / "held",
        identifier_salt=TEST_SALT,
    )
    assert held["monitoring_gate_passed"] is False
    assert held["maximum_feature_psi"] > 0.25
    assert held["assignment_status"] == "review_required"

    observations["snapshot_date"] = "2025-12-31"
    earlier = tmp_path / "earlier.csv"
    observations.to_csv(earlier, index=False)
    with pytest.raises(ValueError, match="cannot precede"):
        score_external_segmentation(
            earlier,
            tmp_path / "model",
            tmp_path / "score",
            identifier_salt=TEST_SALT,
        )


def test_external_exports_require_an_explicit_identifier_policy(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    input_path = tmp_path / "features.csv"
    observations.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="salt of at least 16"):
        run_external_segmentation(input_path, tmp_path / "output")

    raw = run_external_segmentation(
        input_path,
        tmp_path / "raw-output",
        identifier_policy="raw",
    )
    assignments = pd.read_csv(tmp_path / "raw-output" / "customer_segments.csv", dtype="string")
    assert raw["identifier_policy"] == "raw"
    assert set(assignments["customer_id"]) == set(observations["customer_id"])


def test_previous_assignments_must_match_definition_policy_and_date(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    training = tmp_path / "training.csv"
    observations.to_csv(training, index=False)
    fit_external_segmentation(
        training,
        tmp_path / "model",
        tmp_path / "fit",
        identifier_salt=TEST_SALT,
    )
    previous_path = tmp_path / "fit" / "customer_segments.csv"
    previous = pd.read_csv(previous_path, dtype="string")

    previous["segment_definition_id"] = "seg-v3-wrong"
    wrong_definition = tmp_path / "wrong-definition.csv"
    previous.to_csv(wrong_definition, index=False)
    with pytest.raises(ValueError, match="different segment definition"):
        score_external_segmentation(
            training,
            tmp_path / "model",
            tmp_path / "score-definition",
            wrong_definition,
            identifier_salt=TEST_SALT,
        )

    previous = pd.read_csv(previous_path, dtype="string")
    previous["identifier_policy"] = "raw"
    wrong_policy = tmp_path / "wrong-policy.csv"
    previous.to_csv(wrong_policy, index=False)
    with pytest.raises(ValueError, match="different identifier policy"):
        score_external_segmentation(
            training,
            tmp_path / "model",
            tmp_path / "score-policy",
            wrong_policy,
            identifier_salt=TEST_SALT,
        )

    previous = pd.read_csv(previous_path, dtype="string")
    previous["snapshot_date"] = "2027-01-01"
    later = tmp_path / "later.csv"
    previous.to_csv(later, index=False)
    with pytest.raises(ValueError, match="later snapshot_date"):
        score_external_segmentation(
            training,
            tmp_path / "model",
            tmp_path / "score-date",
            later,
            identifier_salt=TEST_SALT,
        )


def test_scoring_policy_must_match_fitted_definition(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    training = tmp_path / "training.csv"
    observations.to_csv(training, index=False)
    fit_external_segmentation(training, tmp_path / "model")

    config = json.loads(
        resources.files("customer_segmentation")
        .joinpath("resources/analysis.json")
        .read_text(encoding="utf-8")
    )
    config["maximum_feature_psi"] = 10.0
    changed_config = tmp_path / "changed-config.json"
    changed_config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="governance policy does not match"):
        score_external_segmentation(
            training,
            tmp_path / "model",
            tmp_path / "score",
            config_path=changed_config,
            identifier_salt=TEST_SALT,
        )
