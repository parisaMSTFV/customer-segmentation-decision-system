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
    selection_seeds: tuple[int, ...]
    bootstrap_runs: int
    minimum_cluster_share: float
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
        selection_seeds=tuple(int(value) for value in raw["selection_seeds"]),
        bootstrap_runs=int(raw["bootstrap_runs"]),
        minimum_cluster_share=float(raw["minimum_cluster_share"]),
        silhouette_weight=float(weights["silhouette"]),
        stability_weight=float(weights["stability"]),
    )
    if config.customer_count < 600:
        raise ValueError("customer_count must be at least 600")
    if not 0.5 <= config.development_share < 1:
        raise ValueError("development_share must be in [0.5, 1.0)")
    if len(config.candidate_clusters) < 2 or min(config.candidate_clusters) < 2:
        raise ValueError("candidate_clusters must contain at least two values >= 2")
    if len(config.selection_seeds) < 2:
        raise ValueError("selection_seeds must contain at least two values")
    if config.bootstrap_runs < 2:
        raise ValueError("bootstrap_runs must be at least two")
    if not 0 < config.minimum_cluster_share < 0.5:
        raise ValueError("minimum_cluster_share must be in (0, 0.5)")
    if abs(config.silhouette_weight + config.stability_weight - 1) > 1e-9:
        raise ValueError("selection weights must sum to one")
    return config
