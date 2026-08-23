"""End-to-end reproducible segmentation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from customer_segmentation.config import DEFAULT_CONFIG_PATH, load_config
from customer_segmentation.evaluation import evaluate_holdout
from customer_segmentation.features import RFM_COLUMNS, prepare_features
from customer_segmentation.labeling import (
    assign_business_names,
    build_decision_playbook,
    create_cluster_profiles,
    name_profiles,
)
from customer_segmentation.modeling import (
    bootstrap_stability,
    build_model,
    evaluate_candidates,
    select_cluster_count,
)
from customer_segmentation.reporting import (
    plot_decision_playbook,
    plot_evaluation_summary,
    plot_model_selection,
    plot_segment_map,
    plot_segment_profiles,
    write_decision_brief,
)
from customer_segmentation.schema import SNAPSHOT_COLUMN, validate_customer_features
from customer_segmentation.synthetic import generate_customers


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")


def _artifact_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def run_pipeline(output_root: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Regenerate data, models, evaluations, figures, and the decision brief."""
    config = load_config(config_path)
    observations, truth = generate_customers(config.customer_count, config.seed)
    checks_passed = validate_customer_features(observations)

    development_indices, holdout_indices = train_test_split(
        range(len(observations)),
        train_size=config.development_share,
        random_state=config.seed,
        shuffle=True,
    )
    development = observations.iloc[development_indices].reset_index(drop=True)
    holdout = observations.iloc[holdout_indices].reset_index(drop=True)
    development_index_set = set(development_indices)
    split_assignments = pd.DataFrame(
        {
            "customer_id": observations["customer_id"],
            "split": [
                "development" if index in development_index_set else "holdout"
                for index in range(len(observations))
            ],
        }
    )
    truth_by_customer = truth.set_index("customer_id")["synthetic_persona"]
    holdout_truth = truth_by_customer.loc[holdout["customer_id"]].reset_index(drop=True)

    development_features = prepare_features(development)
    holdout_features = prepare_features(holdout)
    candidate_metrics = evaluate_candidates(development_features, config)
    selected_cluster_count = select_cluster_count(
        candidate_metrics,
        config.governed_cluster_count,
        config.selection_tolerance,
    )
    candidate_metrics["selected"] = candidate_metrics["cluster_count"] == selected_cluster_count

    enhanced_model = build_model(selected_cluster_count, config.seed)
    enhanced_model.fit(development_features)
    development_rfm_features = prepare_features(development, RFM_COLUMNS)
    rfm_candidate_metrics = evaluate_candidates(development_rfm_features, config)
    rfm_selected_cluster_count = select_cluster_count(
        rfm_candidate_metrics,
        selection_tolerance=0.0,
    )
    rfm_candidate_metrics["selected"] = (
        rfm_candidate_metrics["cluster_count"] == rfm_selected_cluster_count
    )
    fixed_k_baseline_model = build_model(selected_cluster_count, config.seed)
    fixed_k_baseline_model.fit(development_rfm_features)
    self_selected_baseline_model = build_model(rfm_selected_cluster_count, config.seed)
    self_selected_baseline_model.fit(development_rfm_features)
    holdout_metrics, holdout_labels = evaluate_holdout(
        holdout_features,
        holdout_truth,
        enhanced_model,
        fixed_k_baseline_model,
        self_selected_baseline_model,
        config.seed + 1,
    )
    bootstrap_mean, bootstrap_minimum = bootstrap_stability(
        development_features,
        selected_cluster_count,
        config.bootstrap_runs,
        config.seed,
    )
    profiles = create_cluster_profiles(holdout, pd.Series(holdout_labels))
    name_map = assign_business_names(profiles, require_confident_match=True)
    named_profiles = name_profiles(profiles, name_map)
    playbook = build_decision_playbook(named_profiles)

    all_features = prepare_features(observations)
    all_labels = enhanced_model.predict(all_features)
    segments = observations[["customer_id", SNAPSHOT_COLUMN]].copy()
    segments["cluster_id"] = all_labels
    segments["segment_name"] = segments["cluster_id"].map(name_map)
    segments = segments.merge(split_assignments, on="customer_id", validate="one_to_one")

    data_paths = [
        output_root / "data" / "synthetic" / "customer_features.csv",
        output_root / "data" / "synthetic" / "evaluator_truth.csv",
        output_root / "data" / "processed" / "split_assignments.csv",
        output_root / "data" / "processed" / "customer_segments.csv",
        output_root / "reports" / "model_selection.csv",
        output_root / "reports" / "rfm_model_selection.csv",
        output_root / "reports" / "segment_profiles.csv",
        output_root / "reports" / "decision_playbook.csv",
    ]
    for frame, path in zip(
        [
            observations,
            truth,
            split_assignments,
            segments,
            candidate_metrics,
            rfm_candidate_metrics,
            named_profiles,
            playbook,
        ],
        data_paths,
        strict=True,
    ):
        _write_csv(frame, path)

    metrics: dict[str, object] = {
        "data": {
            "customers": len(observations),
            "development_customers": len(development),
            "holdout_customers": len(holdout),
            "synthetic_personas": int(truth["synthetic_persona"].nunique()),
            "schema_checks_passed": checks_passed,
        },
        "selection": {
            "selected_cluster_count": selected_cluster_count,
            "governed_cluster_count": config.governed_cluster_count,
            "rfm_selected_cluster_count": rfm_selected_cluster_count,
            "selection_tolerance": config.selection_tolerance,
            "selection_rationale": (
                "Governed count retained only when eligible and within tolerance of the best score"
            ),
            "scope": "Development features only; evaluator truth excluded",
            "minimum_cluster_share_guardrail": config.minimum_cluster_share,
        },
        "holdout_evaluation": holdout_metrics,
        "stability": {
            "bootstrap_runs": config.bootstrap_runs,
            "mean_pairwise_ari": bootstrap_mean,
            "minimum_pairwise_ari": bootstrap_minimum,
        },
        "decision_layer": {
            "segments_with_action_hypothesis": len(playbook),
            "segments_with_guardrail": int(playbook["guardrail"].notna().sum()),
            "claimed_campaign_impact": "Not evaluated",
        },
        "evaluation_boundary": "Held-out synthetic customers; no causal or production claims",
        "artifact_fingerprint": _artifact_fingerprint(data_paths),
    }
    metrics_path = output_root / "reports" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    flat_plot_metrics = {
        **holdout_metrics,
        "bootstrap_mean_pairwise_ari": bootstrap_mean,
        "selected_cluster_count": float(selected_cluster_count),
    }
    figure_root = output_root / "reports" / "figures"
    plot_model_selection(
        candidate_metrics, selected_cluster_count, figure_root / "model_selection.png"
    )
    plot_evaluation_summary(flat_plot_metrics, figure_root / "evaluation_summary.png")
    scaled_holdout = enhanced_model[:-1].transform(holdout_features)
    plot_segment_map(scaled_holdout, holdout_labels, name_map, figure_root / "segment_map.png")
    plot_segment_profiles(named_profiles, figure_root / "segment_profiles.png")
    plot_decision_playbook(playbook, figure_root / "decision_playbook.png")
    write_decision_brief(
        flat_plot_metrics,
        named_profiles,
        playbook,
        output_root / "reports" / "segment_decision_brief.html",
    )
    return metrics
