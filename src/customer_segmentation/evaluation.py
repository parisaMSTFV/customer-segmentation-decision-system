"""Holdout evaluation for the synthetic benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from customer_segmentation.features import RFM_COLUMNS
from customer_segmentation.modeling import transform_for_clustering


def evaluate_holdout(
    features: pd.DataFrame,
    truth: pd.Series,
    enhanced_model: Pipeline,
    fixed_k_baseline_model: Pipeline,
    self_selected_baseline_model: Pipeline,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate frozen development models on untouched synthetic holdout customers."""
    enhanced_labels = enhanced_model.predict(features)
    fixed_k_baseline_labels = fixed_k_baseline_model.predict(features.loc[:, RFM_COLUMNS])
    self_selected_baseline_labels = self_selected_baseline_model.predict(
        features.loc[:, RFM_COLUMNS]
    )
    scaled = transform_for_clustering(enhanced_model, features)
    rng = np.random.default_rng(seed)
    shuffled_truth = truth.iloc[rng.permutation(len(truth))]
    metrics = {
        "enhanced_synthetic_truth_ari": float(adjusted_rand_score(truth, enhanced_labels)),
        "rfm_fixed_k_synthetic_truth_ari": float(
            adjusted_rand_score(truth, fixed_k_baseline_labels)
        ),
        "rfm_self_selected_synthetic_truth_ari": float(
            adjusted_rand_score(truth, self_selected_baseline_labels)
        ),
        "shuffled_label_null_ari": float(adjusted_rand_score(shuffled_truth, enhanced_labels)),
        "holdout_silhouette": float(silhouette_score(scaled, enhanced_labels)),
        "holdout_davies_bouldin": float(davies_bouldin_score(scaled, enhanced_labels)),
        "holdout_calinski_harabasz": float(calinski_harabasz_score(scaled, enhanced_labels)),
        "minimum_holdout_segment_share": float(
            pd.Series(enhanced_labels).value_counts(normalize=True).min()
        ),
    }
    return metrics, enhanced_labels
