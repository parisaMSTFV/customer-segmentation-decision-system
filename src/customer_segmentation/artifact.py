"""Versioned, checksum-verified segmentation model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from customer_segmentation import __version__
from customer_segmentation.schema import FEATURE_COLUMNS, SNAPSHOT_COLUMN

ARTIFACT_SCHEMA_VERSION = "2.0"
MODEL_FILENAME = "segmentation_model.joblib"
MANIFEST_FILENAME = "model_manifest.json"


def file_sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_training_sha256(frame: pd.DataFrame) -> str:
    """Hash a normalized row-order-invariant representation of training data."""
    columns = ["customer_id", SNAPSHOT_COLUMN, *FEATURE_COLUMNS]
    normalized = frame.loc[:, columns].sort_values("customer_id", kind="stable")
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rounded(values: object) -> list[object]:
    return np.asarray(values, dtype=float).round(10).tolist()


def build_definition_payload(
    model: Pipeline,
    name_map: dict[int, str],
    snapshot_date: str,
    training_data_sha256: str,
    input_contract_version: str,
    governance: dict[str, object],
) -> dict[str, object]:
    """Build a semantic payload independent of arbitrary numeric cluster labels."""
    clip = model.named_steps["clip"]
    scale = model.named_steps["scale"]
    cluster = model.named_steps["cluster"]
    centers_by_name = {
        name: _rounded(cluster.cluster_centers_[cluster_id])
        for cluster_id, name in sorted(name_map.items(), key=lambda item: item[1])
    }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "input_contract_version": input_contract_version,
        "features": list(FEATURE_COLUMNS),
        "snapshot_date": snapshot_date,
        "training_data_sha256": training_data_sha256,
        "governance": governance,
        "preprocessing": {
            "log_features": [
                "recency_days",
                "orders_12m",
                "revenue_12m",
                "sessions_90d",
                "category_breadth_12m",
            ],
            "clip_lower_quantile": clip.lower_quantile,
            "clip_upper_quantile": clip.upper_quantile,
            "clip_lower_bounds": _rounded(clip.lower_bounds_),
            "clip_upper_bounds": _rounded(clip.upper_bounds_),
            "scale_mean": _rounded(scale.mean_),
            "scale_standard_deviation": _rounded(scale.scale_),
        },
        "model": {
            "algorithm": "KMeans",
            "cluster_count": int(cluster.n_clusters),
            "n_init": int(cluster.n_init),
            "centers_by_segment_name": centers_by_name,
        },
    }


def definition_id(payload: dict[str, object]) -> str:
    """Create a compact identifier from a canonical semantic definition."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"seg-v3-{hashlib.sha256(encoded).hexdigest()[:12]}"


def save_model_artifact(
    model_dir: Path,
    model: Pipeline,
    name_map: dict[int, str],
    definition_payload: dict[str, object],
    manifest_fields: dict[str, object],
) -> dict[str, object]:
    """Persist the fitted model and a checksum-verified JSON manifest."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    manifest = {
        **manifest_fields,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "segment_definition_id": definition_id(definition_payload),
        "definition_payload": definition_payload,
        "name_map": {str(key): value for key, value in sorted(name_map.items())},
        "model_file": MODEL_FILENAME,
        "model_file_sha256": file_sha256(model_path),
        "runtime_versions": {
            "customer_segmentation": __version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (model_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_model_artifact(model_dir: Path) -> tuple[Pipeline, dict[int, str], dict[str, object]]:
    """Verify and load a trusted local segmentation artifact."""
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported segmentation artifact schema")
    if manifest.get("model_file") != MODEL_FILENAME:
        raise ValueError("Segmentation manifest contains an unsupported model path")
    payload = manifest.get("definition_payload")
    if not isinstance(payload, dict) or definition_id(payload) != manifest.get(
        "segment_definition_id"
    ):
        raise ValueError("Segmentation definition integrity check failed")
    versions = manifest.get("runtime_versions")
    expected_versions = {
        "customer_segmentation": __version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if versions != expected_versions:
        raise ValueError(
            "Segmentation artifact runtime versions do not match the scoring environment"
        )
    model_path = model_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    if file_sha256(model_path) != manifest["model_file_sha256"]:
        raise ValueError("Segmentation model integrity check failed")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or not {"clip", "scale", "cluster"}.issubset(
        model.named_steps
    ):
        raise ValueError("Segmentation model has an unsupported pipeline structure")
    name_map = {int(key): str(value) for key, value in manifest["name_map"].items()}
    cluster_count = int(model.named_steps["cluster"].n_clusters)
    if set(name_map) != set(range(cluster_count)) or len(set(name_map.values())) != cluster_count:
        raise ValueError("Segmentation name map does not match the fitted clusters")
    return model, name_map, manifest
