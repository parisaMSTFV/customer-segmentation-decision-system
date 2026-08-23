"""Population drift, centroid shift, and segment migration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from customer_segmentation.modeling import transform_for_clustering


def segment_share_psi(reference: dict[str, float], current: pd.Series) -> float:
    """Calculate population stability index over governed segment shares."""
    epsilon = 1e-6
    current_shares = current.value_counts(normalize=True).to_dict()
    names = sorted(set(reference).union(current_shares))
    score = 0.0
    for name in names:
        expected = max(float(reference.get(name, 0.0)), epsilon)
        observed = max(float(current_shares.get(name, 0.0)), epsilon)
        score += (observed - expected) * np.log(observed / expected)
    return float(score)


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
