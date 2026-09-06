"""Population drift, centroid shift, and segment migration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from customer_segmentation.modeling import transform_for_clustering

PSI_EPSILON = 1e-6


def segment_share_psi(reference: dict[str, float], current: pd.Series) -> float:
    """Calculate population stability index over governed segment shares."""
    current_shares = current.value_counts(normalize=True).to_dict()
    names = sorted(set(reference).union(current_shares))
    score = 0.0
    for name in names:
        expected = max(float(reference.get(name, 0.0)), PSI_EPSILON)
        observed = max(float(current_shares.get(name, 0.0)), PSI_EPSILON)
        score += (observed - expected) * np.log(observed / expected)
    return float(score)


def build_feature_psi_reference(
    features: pd.DataFrame,
    bins: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """Persist training quantile bins and expected shares for feature-level PSI."""
    if bins < 2:
        raise ValueError("feature PSI requires at least two bins")
    reference: dict[str, dict[str, list[float]]] = {}
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, quantiles))
        indices = np.searchsorted(edges, values, side="right")
        shares = np.bincount(indices, minlength=len(edges) + 1) / len(values)
        reference[column] = {
            "edges": edges.astype(float).tolist(),
            "expected_shares": shares.astype(float).tolist(),
        }
    return reference


def feature_psi(
    reference: dict[str, dict[str, list[float]]],
    current: pd.DataFrame,
) -> dict[str, float]:
    """Calculate PSI for every feature using frozen training bins."""
    if set(reference) != set(current.columns):
        raise ValueError("feature PSI reference does not match scoring columns")
    scores: dict[str, float] = {}
    for column in current.columns:
        edges = np.asarray(reference[column]["edges"], dtype=float)
        expected = np.asarray(reference[column]["expected_shares"], dtype=float)
        values = current[column].to_numpy(dtype=float)
        indices = np.searchsorted(edges, values, side="right")
        observed = np.bincount(indices, minlength=len(edges) + 1) / len(values)
        if len(expected) != len(observed):
            raise ValueError(f"feature PSI reference is invalid for {column}")
        expected = np.maximum(expected, PSI_EPSILON)
        observed = np.maximum(observed, PSI_EPSILON)
        scores[column] = float(np.sum((observed - expected) * np.log(observed / expected)))
    return scores


def maximum_centroid_shift(
    model: Pipeline,
    features: pd.DataFrame,
    labels: np.ndarray,
) -> float:
    """Measure the largest standardized distance from current to fitted centroids."""
    transformed = transform_for_clustering(model, features)
    fitted_centers = model.named_steps["cluster"].cluster_centers_
    shifts: list[float] = []
    for cluster_id in range(len(fitted_centers)):
        members = transformed[labels == cluster_id]
        if len(members) == 0:
            return 1e12
        shifts.append(float(np.linalg.norm(members.mean(axis=0) - fitted_centers[cluster_id])))
    return max(shifts)


def build_migration_matrix(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build a row-normalized migration matrix for overlapping customers."""
    required = {"customer_id", "segment_name"}
    for label, frame in (("current", current), ("previous", previous)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} assignments are missing columns: {missing}")
        if frame["customer_id"].duplicated().any():
            raise ValueError(f"{label} assignments contain duplicate customer_id values")
    joined = previous[list(required)].merge(
        current[list(required)],
        on="customer_id",
        suffixes=("_previous", "_current"),
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("previous and current assignments have no overlapping customers")
    counts = pd.crosstab(joined["segment_name_previous"], joined["segment_name_current"])
    matrix = counts.div(counts.sum(axis=1), axis=0).reset_index()
    unchanged = joined["segment_name_previous"].eq(joined["segment_name_current"])
    metrics: dict[str, float | int] = {
        "overlapping_customers": len(joined),
        "migration_rate": float(1 - unchanged.mean()),
    }
    return matrix, metrics
