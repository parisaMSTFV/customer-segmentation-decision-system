"""Development-only model selection and stability analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from customer_segmentation.config import AnalysisConfig


def build_model(cluster_count: int, seed: int) -> Pipeline:
    """Create a scaling and K-means pipeline with a controlled seed."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(n_clusters=cluster_count, n_init=20, random_state=seed),
            ),
        ]
    )


def mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    """Measure agreement across repeated label vectors."""
    if len(label_sets) < 2:
        raise ValueError("at least two label vectors are required")
    scores = [adjusted_rand_score(left, right) for left, right in combinations(label_sets, 2)]
    return float(np.mean(scores))


def evaluate_candidates(features: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Evaluate candidate cluster counts using only development features."""
    records: list[dict[str, float | int | bool]] = []
    for cluster_count in config.candidate_clusters:
        labels_by_seed: list[np.ndarray] = []
        silhouettes: list[float] = []
        minimum_shares: list[float] = []
        for seed in config.selection_seeds:
            model = build_model(cluster_count, seed)
            labels = model.fit_predict(features)
            scaled = model.named_steps["scale"].transform(features)
            labels_by_seed.append(labels)
            silhouettes.append(float(silhouette_score(scaled, labels)))
            minimum_shares.append(float(pd.Series(labels).value_counts(normalize=True).min()))
        stability = mean_pairwise_ari(labels_by_seed)
        silhouette = float(np.mean(silhouettes))
        minimum_share = float(np.min(minimum_shares))
        eligible = minimum_share >= config.minimum_cluster_share
        selection_score = (
            config.silhouette_weight * silhouette + config.stability_weight * stability
        )
        records.append(
            {
                "cluster_count": cluster_count,
                "silhouette": silhouette,
                "seed_stability_ari": stability,
                "minimum_cluster_share": minimum_share,
                "eligible": eligible,
                "selection_score": selection_score,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("cluster_count").reset_index(drop=True)


def select_cluster_count(candidate_metrics: pd.DataFrame) -> int:
    """Select the highest-scoring eligible candidate with a smaller-k tie break."""
    eligible = candidate_metrics[candidate_metrics["eligible"]].copy()
    if eligible.empty:
        raise ValueError("no candidate satisfies the minimum cluster-share guardrail")
    selected = eligible.sort_values(
        ["selection_score", "cluster_count"], ascending=[False, True]
    ).iloc[0]
    return int(selected["cluster_count"])


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
