"""Development-only model selection and stability analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from customer_segmentation.config import AnalysisConfig


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Clip each feature to training quantiles before scaling."""

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, values: pd.DataFrame | np.ndarray, y: object = None) -> QuantileClipper:
        """Learn per-feature clipping bounds."""
        array = np.asarray(values, dtype=float)
        self.lower_bounds_ = np.quantile(array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(array, self.upper_quantile, axis=0)
        return self

    def transform(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply the learned clipping bounds."""
        array = np.asarray(values, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)

    def clipped_customer_share(self, values: pd.DataFrame | np.ndarray) -> float:
        """Return the share of rows affected by at least one clipping bound."""
        array = np.asarray(values, dtype=float)
        clipped = (array < self.lower_bounds_) | (array > self.upper_bounds_)
        return float(clipped.any(axis=1).mean())


def build_model(cluster_count: int, seed: int) -> Pipeline:
    """Create a scaling and K-means pipeline with a controlled seed."""
    return Pipeline(
        [
            ("clip", QuantileClipper()),
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(n_clusters=cluster_count, n_init=20, random_state=seed),
            ),
        ]
    )


def transform_for_clustering(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Apply every fitted preprocessing step without predicting clusters."""
    return np.asarray(model[:-1].transform(features), dtype=float)


def mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    """Measure agreement across repeated label vectors."""
    if len(label_sets) < 2:
        raise ValueError("at least two label vectors are required")
    scores = [adjusted_rand_score(left, right) for left, right in combinations(label_sets, 2)]
    return float(np.mean(scores))


def evaluate_candidates(features: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Evaluate candidates across bootstrap resamples of development features."""
    records: list[dict[str, float | int | bool]] = []
    for cluster_count in config.candidate_clusters:
        labels_by_resample: list[np.ndarray] = []
        silhouettes: list[float] = []
        minimum_shares: list[float] = []
        for seed in config.selection_seeds:
            rng = np.random.default_rng(seed)
            sample_indices = rng.integers(0, len(features), size=len(features))
            model = build_model(cluster_count, seed)
            model.fit(features.iloc[sample_indices])
            labels = model.predict(features)
            scaled = transform_for_clustering(model, features)
            labels_by_resample.append(labels)
            silhouettes.append(float(silhouette_score(scaled, labels)))
            minimum_shares.append(float(pd.Series(labels).value_counts(normalize=True).min()))
        pairwise_scores = [
            adjusted_rand_score(left, right) for left, right in combinations(labels_by_resample, 2)
        ]
        stability = float(np.mean(pairwise_scores))
        minimum_stability = float(np.min(pairwise_scores))
        silhouette = float(np.mean(silhouettes))
        minimum_share = float(np.min(minimum_shares))
        eligible = bool(
            minimum_share >= config.minimum_cluster_share
            and silhouette >= config.minimum_silhouette
            and minimum_stability >= config.minimum_resample_stability
        )
        selection_score = (
            config.silhouette_weight * silhouette + config.stability_weight * stability
        )
        records.append(
            {
                "cluster_count": cluster_count,
                "silhouette": silhouette,
                "silhouette_standard_deviation": float(np.std(silhouettes, ddof=1)),
                "resample_stability_ari": stability,
                "minimum_resample_ari": minimum_stability,
                "minimum_cluster_share": minimum_share,
                "eligible": eligible,
                "selection_score": selection_score,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("cluster_count").reset_index(drop=True)


def select_cluster_count(
    candidate_metrics: pd.DataFrame,
    governed_cluster_count: int | None = None,
    selection_tolerance: float = 0.0,
) -> int:
    """Select an eligible candidate while treating practically tied scores explicitly."""
    eligible = candidate_metrics[candidate_metrics["eligible"]].copy()
    if eligible.empty:
        raise ValueError("no candidate satisfies the minimum cluster-share guardrail")
    best_score = float(eligible["selection_score"].max())
    tied = eligible[eligible["selection_score"] >= best_score - selection_tolerance]
    if governed_cluster_count is not None:
        governed = tied[tied["cluster_count"] == governed_cluster_count]
        if not governed.empty:
            return governed_cluster_count
    return int(tied["cluster_count"].min())


def governed_candidate_is_supported(
    candidate_metrics: pd.DataFrame,
    governed_cluster_count: int,
    selection_tolerance: float,
) -> bool:
    """Check whether the governed taxonomy is eligible and close enough to the best score."""
    eligible = candidate_metrics[candidate_metrics["eligible"]]
    governed = eligible[eligible["cluster_count"] == governed_cluster_count]
    if governed.empty:
        return False
    best_score = float(eligible["selection_score"].max())
    governed_score = float(governed.iloc[0]["selection_score"])
    return governed_score >= best_score - selection_tolerance


def bootstrap_stability(
    features: pd.DataFrame,
    cluster_count: int,
    runs: int,
    seed: int,
) -> tuple[float, float]:
    """Fit bootstrap samples and compare predictions on a common development frame."""
    rng = np.random.default_rng(seed)
    label_sets: list[np.ndarray] = []
    for run in range(runs):
        sample_indices = rng.integers(0, len(features), size=len(features))
        model = build_model(cluster_count, seed + 100 + run)
        model.fit(features.iloc[sample_indices])
        label_sets.append(model.predict(features))
    scores = [adjusted_rand_score(left, right) for left, right in combinations(label_sets, 2)]
    return float(np.mean(scores)), float(np.min(scores))
