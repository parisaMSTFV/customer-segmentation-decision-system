"""Configuration loading and project paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis.json"


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
    maximum_clipped_customer_share: float
    maximum_segment_share_psi: float
    maximum_centroid_shift: float
    silhouette_weight: float
    stability_weight: float


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AnalysisConfig:
    """Load and validate the JSON configuration."""
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
        maximum_clipped_customer_share=float(raw["maximum_clipped_customer_share"]),
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
    if config.governed_cluster_count not in config.candidate_clusters:
        raise ValueError("governed_cluster_count must be one of candidate_clusters")
    if len(config.selection_seeds) < 2:
        raise ValueError("selection_seeds must contain at least two values")
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
    if not 0 < config.maximum_clipped_customer_share < 1:
        raise ValueError("maximum_clipped_customer_share must be in (0, 1)")
    if config.maximum_segment_share_psi <= 0 or config.maximum_centroid_shift <= 0:
        raise ValueError("monitoring thresholds must be positive")
    if abs(config.silhouette_weight + config.stability_weight - 1) > 1e-9:
        raise ValueError("selection weights must sum to one")
    return config
