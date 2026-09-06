"""Configuration loading and project paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated analysis settings."""

    seed: int
    customer_count: int
    development_share: float
    candidate_clusters: tuple[int, ...]
    governed_cluster_count: int
    selection_seeds: tuple[int, ...]
    bootstrap_runs: int
    minimum_cluster_share: float
    minimum_cluster_customers: int
    minimum_external_customers: int
    selection_tolerance: float
    minimum_silhouette: float
    minimum_resample_stability: float
    maximum_clipped_customer_share: float
    maximum_feature_psi: float
    maximum_segment_share_psi: float
    maximum_centroid_shift: float
    silhouette_weight: float
    stability_weight: float


def load_config(path: Path | None = None) -> AnalysisConfig:
    """Load a validated config from disk or the wheel-bundled default."""
    if path is None:
        bundled = resources.files("customer_segmentation").joinpath("resources/analysis.json")
        raw = json.loads(bundled.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    weights = raw["selection_weights"]
    config = AnalysisConfig(
        seed=int(raw["seed"]),
        customer_count=int(raw["customer_count"]),
        development_share=float(raw["development_share"]),
        candidate_clusters=tuple(int(value) for value in raw["candidate_clusters"]),
        governed_cluster_count=int(raw["governed_cluster_count"]),
        selection_seeds=tuple(int(value) for value in raw["selection_seeds"]),
        bootstrap_runs=int(raw["bootstrap_runs"]),
        minimum_cluster_share=float(raw["minimum_cluster_share"]),
        minimum_cluster_customers=int(raw["minimum_cluster_customers"]),
        minimum_external_customers=int(raw["minimum_external_customers"]),
        selection_tolerance=float(raw["selection_tolerance"]),
        minimum_silhouette=float(raw["minimum_silhouette"]),
        minimum_resample_stability=float(raw["minimum_resample_stability"]),
        maximum_clipped_customer_share=float(raw["maximum_clipped_customer_share"]),
        maximum_feature_psi=float(raw["maximum_feature_psi"]),
        maximum_segment_share_psi=float(raw["maximum_segment_share_psi"]),
        maximum_centroid_shift=float(raw["maximum_centroid_shift"]),
        silhouette_weight=float(weights["silhouette"]),
        stability_weight=float(weights["stability"]),
    )
    if config.customer_count < 600:
        raise ValueError("customer_count must be at least 600")
    if not 0.5 <= config.development_share < 1:
        raise ValueError("development_share must be in [0.5, 1.0)")
    if len(config.candidate_clusters) < 2 or min(config.candidate_clusters) < 2:
        raise ValueError("candidate_clusters must contain at least two values >= 2")
    if len(set(config.candidate_clusters)) != len(config.candidate_clusters):
        raise ValueError("candidate_clusters must be unique")
    if config.governed_cluster_count not in config.candidate_clusters:
        raise ValueError("governed_cluster_count must be one of candidate_clusters")
    if len(config.selection_seeds) < 2:
        raise ValueError("selection_seeds must contain at least two values")
    if len(set(config.selection_seeds)) != len(config.selection_seeds):
        raise ValueError("selection_seeds must be unique")
    if config.bootstrap_runs < 2:
        raise ValueError("bootstrap_runs must be at least two")
    if not 0 < config.minimum_cluster_share < 0.5:
        raise ValueError("minimum_cluster_share must be in (0, 0.5)")
    if config.minimum_cluster_customers < 2:
        raise ValueError("minimum_cluster_customers must be at least two")
    if config.minimum_external_customers < (
        config.governed_cluster_count * config.minimum_cluster_customers
    ):
        raise ValueError("minimum_external_customers cannot undercut absolute cluster guardrails")
    if not 0 <= config.selection_tolerance < 1:
        raise ValueError("selection_tolerance must be in [0, 1)")
    if not -1 <= config.minimum_silhouette <= 1:
        raise ValueError("minimum_silhouette must be between minus one and one")
    if not 0 <= config.minimum_resample_stability <= 1:
        raise ValueError("minimum_resample_stability must be between zero and one")
    if not 0 < config.maximum_clipped_customer_share < 1:
        raise ValueError("maximum_clipped_customer_share must be in (0, 1)")
    if (
        config.maximum_feature_psi <= 0
        or config.maximum_segment_share_psi <= 0
        or config.maximum_centroid_shift <= 0
    ):
        raise ValueError("monitoring thresholds must be positive")
    if config.silhouette_weight < 0 or config.stability_weight < 0:
        raise ValueError("selection weights must be nonnegative")
    if abs(config.silhouette_weight + config.stability_weight - 1) > 1e-9:
        raise ValueError("selection weights must sum to one")
    return config
