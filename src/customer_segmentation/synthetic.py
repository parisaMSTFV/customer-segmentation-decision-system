"""Synthetic customer-feature generation with a separate evaluator truth table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SYNTHETIC_SNAPSHOT_DATE = "2026-01-01"


@dataclass(frozen=True)
class PersonaSpec:
    """Parameters for one synthetic behavioral archetype."""

    name: str
    share: float
    orders: float
    aov: float
    recency: float
    margin_rate: float
    sessions: float
    conversion: float
    discount_share: float
    category_breadth: float
    return_rate: float
    satisfaction: float


PERSONAS = (
    PersonaSpec(
        "loyal_high_value", 0.15, 13.0, 155.0, 16.0, 0.27, 31.0, 0.18, 0.14, 6.0, 0.035, 4.55
    ),
    PersonaSpec(
        "high_value_at_risk", 0.12, 8.0, 175.0, 145.0, 0.24, 6.0, 0.06, 0.18, 4.0, 0.055, 3.75
    ),
    PersonaSpec(
        "growth_potential", 0.22, 4.0, 92.0, 38.0, 0.22, 23.0, 0.10, 0.28, 5.0, 0.060, 4.05
    ),
    PersonaSpec(
        "engaged_low_conversion", 0.18, 1.0, 72.0, 82.0, 0.19, 47.0, 0.025, 0.20, 7.0, 0.045, 3.90
    ),
    PersonaSpec(
        "discount_led_frequent", 0.15, 10.0, 63.0, 24.0, -0.06, 29.0, 0.14, 0.76, 4.0, 0.120, 3.55
    ),
    PersonaSpec(
        "dormant_low_value", 0.18, 0.5, 48.0, 245.0, 0.14, 2.5, 0.012, 0.42, 1.5, 0.080, 3.35
    ),
)


def _bounded_normal(
    rng: np.random.Generator,
    mean: float,
    relative_sd: float,
    size: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = rng.normal(mean, max(abs(mean) * relative_sd, 0.01), size=size)
    return np.clip(values, lower, upper)


def generate_customers(customer_count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate privacy-safe observations and a separate synthetic truth table."""
    rng = np.random.default_rng(seed)
    shares = np.array([spec.share for spec in PERSONAS], dtype=float)
    shares /= shares.sum()
    persona_indices = rng.choice(len(PERSONAS), size=customer_count, p=shares)
    rows: list[pd.DataFrame] = []
    truth_rows: list[pd.DataFrame] = []
    customer_offset = 0
    for persona_index, spec in enumerate(PERSONAS):
        size = int(np.sum(persona_indices == persona_index))
        if size == 0:
            continue
        latent_value = rng.normal(0, 0.10, size=size)
        latent_activity = rng.normal(0, 0.12, size=size)
        orders_rate = np.clip(spec.orders * np.exp(latent_activity), 0.05, None)
        orders = np.clip(rng.poisson(orders_rate), 0, 45)
        aov = np.clip(
            rng.lognormal(np.log(spec.aov) + 0.12 * latent_value, 0.15, size=size),
            8,
            700,
        )
        revenue = np.clip(orders * aov * rng.lognormal(0, 0.08, size=size), 0, None)
        recency = np.clip(
            rng.lognormal(np.log(spec.recency + 2) - 0.10 * latent_activity, 0.18, size=size),
            1,
            365,
        )
        margin_rate = _bounded_normal(rng, spec.margin_rate, 0.12, size, -0.30, 0.45)
        sessions_rate = np.clip(spec.sessions * np.exp(latent_activity), 0.1, None)
        sessions = np.clip(rng.poisson(sessions_rate), 0, 180)
        conversion = _bounded_normal(rng, spec.conversion, 0.18, size, 0.001, 0.45)
        discount = _bounded_normal(rng, spec.discount_share, 0.12, size, 0, 1)
        category_breadth = np.clip(
            rng.poisson(np.clip(spec.category_breadth * np.exp(0.15 * latent_activity), 0.2, None)),
            1,
            18,
        )
        return_rate = _bounded_normal(rng, spec.return_rate, 0.22, size, 0, 0.45)
        inactive_sessions = sessions == 0
        conversion[inactive_sessions] = 0
        satisfaction = _bounded_normal(rng, spec.satisfaction, 0.05, size, 1, 5)
        customer_ids = [
            f"SYN-CUST-{value:05d}" for value in range(customer_offset, customer_offset + size)
        ]
        customer_offset += size
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": customer_ids,
                    "snapshot_date": SYNTHETIC_SNAPSHOT_DATE,
                    "recency_days": np.rint(recency).astype(int),
                    "orders_12m": orders.astype(int),
                    "revenue_12m": revenue.round(2),
                    "margin_rate": margin_rate.round(4),
                    "sessions_90d": sessions.astype(int),
                    "conversion_rate_90d": conversion.round(4),
                    "discount_order_share": discount.round(4),
                    "category_breadth_12m": category_breadth.astype(int),
                    "return_rate": return_rate.round(4),
                    "satisfaction_score": satisfaction.round(3),
                }
            )
        )
        truth_rows.append(
            pd.DataFrame({"customer_id": customer_ids, "synthetic_persona": spec.name})
        )
    observations = pd.concat(rows, ignore_index=True)
    truth = pd.concat(truth_rows, ignore_index=True)
    order = rng.permutation(len(observations))
    return observations.iloc[order].reset_index(drop=True), truth.set_index("customer_id").loc[
        observations.iloc[order]["customer_id"]
    ].reset_index()
