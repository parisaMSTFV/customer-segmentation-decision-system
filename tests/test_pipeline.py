from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from customer_segmentation.pipeline import run_pipeline


def test_pipeline_writes_required_artifacts(
    pipeline_output: tuple[Path, dict[str, object]],
) -> None:
    output_root, metrics = pipeline_output
    required = [
        "data/synthetic/customer_features.csv",
        "data/synthetic/evaluator_truth.csv",
        "data/processed/customer_segments.csv",
        "reports/metrics.json",
        "reports/rfm_model_selection.csv",
        "reports/figures/evaluation_summary.png",
        "reports/figures/model_selection.png",
        "reports/figures/segment_map.png",
        "reports/figures/segment_profiles.png",
        "reports/figures/decision_playbook.png",
        "reports/segment_decision_brief.html",
    ]
    assert all((output_root / relative).is_file() for relative in required)
    assert metrics["selection"]["scope"].startswith("Development features only")
    assert metrics["selection"]["selected_cluster_count"] == 6
    assert metrics["selection"]["rfm_selected_cluster_count"] != 6


def test_split_is_disjoint_and_truth_is_not_in_assignments(
    pipeline_output: tuple[Path, dict[str, object]],
) -> None:
    output_root, _ = pipeline_output
    assignments = pd.read_csv(output_root / "data/processed/customer_segments.csv")
    truth = pd.read_csv(output_root / "data/synthetic/evaluator_truth.csv")
    assert "synthetic_persona" not in assignments.columns
    assert set(assignments["customer_id"]) == set(truth["customer_id"])
    assert set(assignments["split"]) == {"development", "holdout"}


def test_pipeline_is_deterministic(
    pipeline_output: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    _, first_metrics = pipeline_output
    second_metrics = run_pipeline(tmp_path)
    assert first_metrics["artifact_fingerprint"] == second_metrics["artifact_fingerprint"]
    assert first_metrics == second_metrics
    loaded = json.loads((tmp_path / "reports/metrics.json").read_text(encoding="utf-8"))
    assert loaded == second_metrics


def test_report_does_not_claim_campaign_impact(
    pipeline_output: tuple[Path, dict[str, object]],
) -> None:
    _, metrics = pipeline_output
    assert metrics["decision_layer"]["claimed_campaign_impact"] == "Not evaluated"
    assert metrics["evaluation_boundary"].startswith("Held-out synthetic")
