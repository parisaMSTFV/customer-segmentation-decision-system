"""Fit and assign the governed six-segment definition to an external feature table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from customer_segmentation.features import prepare_features
from customer_segmentation.labeling import (
    assign_business_names,
    build_decision_playbook,
    create_cluster_profiles,
    name_profiles,
)
from customer_segmentation.modeling import build_model
from customer_segmentation.reporting import plot_decision_playbook, plot_segment_profiles
from customer_segmentation.schema import FEATURE_COLUMNS, validate_customer_features

INPUT_CONTRACT_VERSION = "1.0"
OUTPUT_SCHEMA_VERSION = "1.0"
SEGMENT_COUNT = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_version(model: object, name_map: dict[int, str]) -> str:
    scale = model.named_steps["scale"]
    cluster = model.named_steps["cluster"]
    digest = hashlib.sha256()
    for values in (scale.mean_, scale.scale_, cluster.cluster_centers_):
        digest.update(np.asarray(values, dtype="<f8").round(10).tobytes())
    digest.update(json.dumps(name_map, sort_keys=True).encode("utf-8"))
    return f"seg-v1-{digest.hexdigest()[:12]}"


def run_external_segmentation(
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, object]:
    """Validate a customer feature CSV, fit six clusters, and write stable outputs."""
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    observations = pd.read_csv(source, dtype={"customer_id": "string"})
    checks_passed = validate_customer_features(observations)
    if len(observations) < SEGMENT_COUNT * 10:
        raise ValueError("External segmentation requires at least 60 customers")

    features = prepare_features(observations)
    model = build_model(SEGMENT_COUNT, seed)
    labels = model.fit_predict(features)
    if len(np.unique(labels)) != SEGMENT_COUNT:
        raise ValueError("The fitted model did not produce all six governed segments")

    profiles = create_cluster_profiles(observations, pd.Series(labels))
    name_map = assign_business_names(profiles)
    named_profiles = name_profiles(profiles, name_map)
    playbook = build_decision_playbook(named_profiles)
    model_version = _model_version(model, name_map)

    assignments = observations[["customer_id"]].copy()
    assignments["cluster_id"] = labels
    assignments["segment_name"] = assignments["cluster_id"].map(name_map)
    assignments["segment_definition_id"] = model_version
    assignments["assignment_status"] = "descriptive_only"

    target = Path(output_dir).resolve()
    figures = target / "figures"
    target.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(target / "customer_segments.csv", index=False)
    named_profiles.to_csv(target / "segment_profiles.csv", index=False, float_format="%.6f")
    playbook.to_csv(target / "decision_playbook.csv", index=False)
    plot_segment_profiles(
        named_profiles,
        figures / "segment_profiles.png",
        title="External-input segment profiles (column z-scores)",
    )
    plot_decision_playbook(playbook, figures / "decision_playbook.png")

    cluster_shares = assignments["segment_name"].value_counts(normalize=True)
    scaled_features = model.named_steps["scale"].transform(features)
    metrics: dict[str, object] = {
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "customers": len(observations),
        "features": list(FEATURE_COLUMNS),
        "schema_checks_passed": checks_passed,
        "segment_count": SEGMENT_COUNT,
        "minimum_segment_share": float(cluster_shares.min()),
        "silhouette": float(
            silhouette_score(
                scaled_features,
                labels,
                sample_size=min(10_000, len(observations)),
                random_state=seed,
            )
        ),
        "segment_definition_id": model_version,
        "assignment_status": "descriptive_only",
        "campaign_impact": "Not evaluated",
    }
    (target / "run_metadata.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
