"""Governed fit and score workflows for external customer snapshots."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from customer_segmentation.artifact import (
    build_definition_payload,
    canonical_training_sha256,
    file_sha256,
    load_model_artifact,
    save_model_artifact,
)
from customer_segmentation.config import AnalysisConfig, load_config
from customer_segmentation.features import prepare_features
from customer_segmentation.labeling import (
    assign_business_names,
    build_decision_playbook,
    create_cluster_profiles,
    name_profiles,
)
from customer_segmentation.modeling import (
    build_model,
    evaluate_candidates,
    governed_candidate_is_supported,
    transform_for_clustering,
)
from customer_segmentation.monitoring import (
    build_feature_psi_reference,
    build_migration_matrix,
    feature_psi,
    maximum_centroid_shift,
    segment_share_psi,
)
from customer_segmentation.reporting import plot_decision_playbook, plot_segment_profiles
from customer_segmentation.schema import (
    FEATURE_COLUMNS,
    SNAPSHOT_COLUMN,
    load_customer_snapshot,
    load_previous_assignments,
    pseudonymize_customer_ids,
    validate_identifier_policy,
)

INPUT_CONTRACT_VERSION = "3.0"
OUTPUT_SCHEMA_VERSION = "3.0"


def _governance_policy(config: AnalysisConfig) -> dict[str, object]:
    """Return the fit and scoring policy bound to a segment definition."""
    return {
        "governed_cluster_count": config.governed_cluster_count,
        "minimum_cluster_share": config.minimum_cluster_share,
        "minimum_cluster_customers": config.minimum_cluster_customers,
        "selection_tolerance": config.selection_tolerance,
        "minimum_silhouette": config.minimum_silhouette,
        "minimum_resample_stability": config.minimum_resample_stability,
        "maximum_clipped_customer_share": config.maximum_clipped_customer_share,
        "maximum_feature_psi": config.maximum_feature_psi,
        "maximum_segment_share_psi": config.maximum_segment_share_psi,
        "maximum_centroid_shift": config.maximum_centroid_shift,
        "selection_weights": {
            "silhouette": config.silhouette_weight,
            "stability": config.stability_weight,
        },
    }


def _read_snapshot(input_path: str | Path) -> tuple[Path, pd.DataFrame, int, str]:
    source = Path(input_path).resolve()
    observations, checks_passed, snapshot_date = load_customer_snapshot(source)
    return source, observations, checks_passed, snapshot_date


def _fit_quality(
    model: object,
    features: pd.DataFrame,
    labels: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, float | int | bool]:
    counts = pd.Series(labels).value_counts()
    shares = counts / counts.sum()
    transformed = transform_for_clustering(model, features)
    clipped_share = model.named_steps["clip"].clipped_customer_share(features)
    every_segment_present = len(counts) == config.governed_cluster_count
    minimum_customers = int(counts.min()) if every_segment_present else 0
    minimum_share = float(shares.min()) if every_segment_present else 0.0
    intrinsic_silhouette = (
        float(silhouette_score(transformed, labels)) if 1 < len(counts) < len(labels) else -1.0
    )
    metrics: dict[str, float | int | bool] = {
        "every_segment_present": every_segment_present,
        "minimum_segment_customers": minimum_customers,
        "minimum_segment_share": minimum_share,
        "clipped_customer_share": clipped_share,
        "silhouette": intrinsic_silhouette,
    }
    metrics["quality_gate_passed"] = bool(
        metrics["minimum_segment_customers"] >= config.minimum_cluster_customers
        and metrics["minimum_segment_share"] >= config.minimum_cluster_share
        and clipped_share <= config.maximum_clipped_customer_share
        and intrinsic_silhouette >= config.minimum_silhouette
    )
    return metrics


def _write_outputs(
    output_dir: Path,
    observations: pd.DataFrame,
    labels: np.ndarray,
    name_map: dict[int, str],
    segment_definition_id: str,
    assignment_status: str,
    metadata: dict[str, object],
    migration: pd.DataFrame | None = None,
    *,
    identifier_policy: str,
    identifier_salt: str | None,
) -> pd.DataFrame:
    profiles = create_cluster_profiles(observations, pd.Series(labels))
    named_profiles = name_profiles(profiles, name_map)
    activation_status = "hold" if assignment_status == "review_required" else "hypothesis_only"
    playbook = build_decision_playbook(named_profiles, activation_status=activation_status)

    assignments = observations[["customer_id", SNAPSHOT_COLUMN]].copy()
    assignments["cluster_id"] = labels
    assignments["segment_name"] = assignments["cluster_id"].map(name_map)
    assignments["segment_definition_id"] = segment_definition_id
    assignments["assignment_status"] = assignment_status
    assignments["identifier_policy"] = identifier_policy
    if identifier_policy == "pseudonymized":
        assignments = pseudonymize_customer_ids(assignments, identifier_salt or "")

    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "customer_segments.csv", index=False)
    named_profiles.to_csv(output_dir / "segment_profiles.csv", index=False, float_format="%.6f")
    playbook.to_csv(output_dir / "decision_playbook.csv", index=False)
    if migration is not None:
        migration.to_csv(output_dir / "segment_migration.csv", index=False, float_format="%.6f")
    plot_segment_profiles(
        named_profiles,
        figures / "segment_profiles.png",
        title="External snapshot segment profiles (column z-scores)",
    )
    plot_decision_playbook(playbook, figures / "decision_playbook.png")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assignments


def fit_external_segmentation(
    input_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path | None = None,
    config_path: Path | None = None,
    *,
    identifier_policy: str = "pseudonymized",
    identifier_salt: str | None = None,
) -> dict[str, object]:
    """Fit a governed taxonomy, fail closed on weak structure, and save its artifact."""
    config = load_config(config_path)
    if output_dir is not None:
        validate_identifier_policy(identifier_policy, identifier_salt)
    source, observations, checks_passed, snapshot_date = _read_snapshot(input_path)
    if len(observations) < config.minimum_external_customers:
        raise ValueError(
            f"External fitting requires at least {config.minimum_external_customers} customers"
        )

    features = prepare_features(observations)
    candidate_metrics = evaluate_candidates(features, config)
    if not governed_candidate_is_supported(
        candidate_metrics,
        config.governed_cluster_count,
        config.selection_tolerance,
    ):
        raise ValueError(
            "The governed cluster count is not eligible or is outside the selection tolerance"
        )
    model = build_model(config.governed_cluster_count, config.seed)
    labels = model.fit_predict(features)
    if len(np.unique(labels)) != config.governed_cluster_count:
        raise ValueError("The fitted model did not produce every governed segment")
    quality = _fit_quality(model, features, labels, config)
    if not quality["quality_gate_passed"]:
        raise ValueError(f"External fit failed quality guardrails: {quality}")

    profiles = create_cluster_profiles(observations, pd.Series(labels))
    name_map = assign_business_names(profiles, require_confident_match=True)
    training_sha = canonical_training_sha256(observations)
    payload = build_definition_payload(
        model,
        name_map,
        snapshot_date,
        training_sha,
        INPUT_CONTRACT_VERSION,
        _governance_policy(config),
    )
    training_shares = pd.Series(labels).map(name_map).value_counts(normalize=True).to_dict()
    manifest = save_model_artifact(
        Path(model_dir).resolve(),
        model,
        name_map,
        payload,
        {
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "features": list(FEATURE_COLUMNS),
            "training_snapshot_date": snapshot_date,
            "training_customers": len(observations),
            "training_data_sha256": training_sha,
            "source_file_sha256": file_sha256(source),
            "training_segment_shares": training_shares,
            "training_feature_psi_reference": build_feature_psi_reference(features),
            "fit_quality": quality,
        },
    )
    candidate_records = json.loads(candidate_metrics.to_json(orient="records"))
    metadata: dict[str, object] = {
        "run_type": "fit",
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
        "customers": len(observations),
        "schema_checks_passed": checks_passed,
        "selected_cluster_count": config.governed_cluster_count,
        "selection_tolerance": config.selection_tolerance,
        "candidate_metrics": candidate_records,
        **quality,
        "segment_definition_id": manifest["segment_definition_id"],
        "assignment_status": "fit_snapshot_descriptive_only",
        "identifier_policy": identifier_policy if output_dir is not None else "not_exported",
        "campaign_impact": "Not evaluated",
    }
    if output_dir is not None:
        target = Path(output_dir).resolve()
        _write_outputs(
            target,
            observations,
            labels,
            name_map,
            str(manifest["segment_definition_id"]),
            "fit_snapshot_descriptive_only",
            metadata,
            identifier_policy=identifier_policy,
            identifier_salt=identifier_salt,
        )
        candidate_metrics.to_csv(target / "model_selection.csv", index=False, float_format="%.6f")
    return metadata


def score_external_segmentation(
    input_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    previous_assignments: str | Path | None = None,
    config_path: Path | None = None,
    *,
    identifier_policy: str = "pseudonymized",
    identifier_salt: str | None = None,
) -> dict[str, object]:
    """Score a snapshot with a frozen definition and gate activation on monitoring checks."""
    config = load_config(config_path)
    validate_identifier_policy(identifier_policy, identifier_salt)
    source, observations, checks_passed, snapshot_date = _read_snapshot(input_path)
    if len(observations) < config.minimum_external_customers:
        raise ValueError(
            f"External scoring requires at least {config.minimum_external_customers} customers"
        )
    model, name_map, manifest = load_model_artifact(Path(model_dir).resolve())
    if manifest.get("input_contract_version") != INPUT_CONTRACT_VERSION:
        raise ValueError("Model and scoring input contract versions do not match")
    if manifest["definition_payload"].get("governance") != _governance_policy(config):
        raise ValueError("Scoring governance policy does not match the fitted segment definition")
    training_snapshot_date = date.fromisoformat(str(manifest["training_snapshot_date"]))
    if date.fromisoformat(snapshot_date) < training_snapshot_date:
        raise ValueError("Scoring snapshot_date cannot precede the training snapshot")

    features = prepare_features(observations)
    labels = model.predict(features)
    quality = _fit_quality(model, features, labels, config)
    segment_names = pd.Series(labels).map(name_map)
    psi = segment_share_psi(manifest["training_segment_shares"], segment_names)
    feature_psi_scores = feature_psi(manifest["training_feature_psi_reference"], features)
    maximum_feature_psi = max(feature_psi_scores.values())
    centroid_shift = maximum_centroid_shift(model, features, labels)
    monitoring_passed = bool(
        quality["quality_gate_passed"]
        and maximum_feature_psi <= config.maximum_feature_psi
        and psi <= config.maximum_segment_share_psi
        and centroid_shift <= config.maximum_centroid_shift
    )
    assignment_status = "descriptive_only" if monitoring_passed else "review_required"

    current_preview = observations[["customer_id"]].copy()
    current_preview["segment_name"] = segment_names
    if identifier_policy == "pseudonymized":
        current_preview = pseudonymize_customer_ids(current_preview, identifier_salt or "")
    migration = None
    migration_metrics: dict[str, float | int] | None = None
    previous_snapshot_date: str | None = None
    if previous_assignments is not None:
        previous = load_previous_assignments(Path(previous_assignments).resolve())
        previous_snapshot_date = str(previous[SNAPSHOT_COLUMN].iloc[0])
        if date.fromisoformat(previous_snapshot_date) > date.fromisoformat(snapshot_date):
            raise ValueError("Previous assignments cannot use a later snapshot_date")
        previous_definition = str(previous["segment_definition_id"].iloc[0])
        if previous_definition != manifest["segment_definition_id"]:
            raise ValueError("Previous assignments use a different segment definition")
        previous_policy = str(previous["identifier_policy"].iloc[0])
        if previous_policy != identifier_policy:
            raise ValueError("Previous assignments use a different identifier policy")
        migration, migration_metrics = build_migration_matrix(current_preview, previous)

    metadata: dict[str, object] = {
        "run_type": "score",
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "input_sha256": file_sha256(source),
        "snapshot_date": snapshot_date,
        "training_snapshot_date": manifest["training_snapshot_date"],
        "customers": len(observations),
        "features": list(FEATURE_COLUMNS),
        "schema_checks_passed": checks_passed,
        **quality,
        "segment_share_psi": psi,
        "feature_psi": feature_psi_scores,
        "maximum_feature_psi": maximum_feature_psi,
        "maximum_centroid_shift": centroid_shift,
        "monitoring_gate_passed": monitoring_passed,
        "migration": migration_metrics,
        "previous_snapshot_date": previous_snapshot_date,
        "segment_definition_id": manifest["segment_definition_id"],
        "assignment_status": assignment_status,
        "identifier_policy": identifier_policy,
        "campaign_impact": "Not evaluated",
    }
    _write_outputs(
        Path(output_dir).resolve(),
        observations,
        labels,
        name_map,
        str(manifest["segment_definition_id"]),
        assignment_status,
        metadata,
        migration,
        identifier_policy=identifier_policy,
        identifier_salt=identifier_salt,
    )
    return metadata


def run_external_segmentation(
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    config_path: Path | None = None,
    *,
    identifier_policy: str = "pseudonymized",
    identifier_salt: str | None = None,
) -> dict[str, object]:
    """Run a guarded one-off fit while retaining the original command contract."""
    if seed != 42:
        raise ValueError(
            "Use the governed configuration seed; custom segment seeds are unsupported"
        )
    target = Path(output_dir).resolve()
    return fit_external_segmentation(
        input_path,
        target / "model",
        target,
        config_path,
        identifier_policy=identifier_policy,
        identifier_salt=identifier_salt,
    )
