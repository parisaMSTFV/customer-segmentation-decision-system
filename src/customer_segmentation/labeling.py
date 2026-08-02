"""Profile-based segment names and decision hypotheses."""

from __future__ import annotations

import pandas as pd

SEGMENT_ORDER = (
    "Loyal high value",
    "High value at risk",
    "Growth potential",
    "Engaged low conversion",
    "Discount-led frequent",
    "Dormant low value",
)

DECISIONS = {
    "Loyal high value": (
        "Protect retention and margin",
        "Test recognition benefits or premium cross-sell",
        "Keep a holdout; avoid discounts that subsidize existing demand",
    ),
    "High value at risk": (
        "Diagnose and recover valuable inactivity",
        "Test a time-boxed, reason-specific win-back journey",
        "Cap contact frequency and measure incremental reactivation",
    ),
    "Growth potential": (
        "Increase category depth profitably",
        "Test recommendation-led cross-sell before broad incentives",
        "Track contribution margin as well as conversion",
    ),
    "Engaged low conversion": (
        "Remove conversion friction",
        "Test onsite guidance, trust cues, or checkout diagnostics",
        "Do not assume price is the barrier without an experiment",
    ),
    "Discount-led frequent": (
        "Preserve frequency while improving economics",
        "Test margin-safe bundles and thresholded offers",
        "Enforce contribution-margin and subsidy guardrails",
    ),
    "Dormant low value": (
        "Control reacquisition cost",
        "Use low-cost owned-channel tests or suppress paid targeting",
        "Reassess periodically; never treat the label as permanent",
    ),
}


def create_cluster_profiles(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Aggregate observed customer features by model cluster."""
    profiled = frame.copy()
    profiled["cluster_id"] = labels.to_numpy()
    means = profiled.groupby("cluster_id").mean(numeric_only=True)
    counts = profiled.groupby("cluster_id").size().rename("customers")
    profiles = means.join(counts)
    profiles["share"] = profiles["customers"] / profiles["customers"].sum()
    return profiles.reset_index()


def _standardize_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column for column in profiles.columns if column not in {"cluster_id", "customers", "share"}
    ]
    values = profiles.set_index("cluster_id")[feature_columns]
    standard_deviation = values.std(ddof=0).replace(0, 1)
    return (values - values.mean()) / standard_deviation


def assign_business_names(profiles: pd.DataFrame) -> dict[int, str]:
    """Assign six unique descriptive names from relative cluster profiles."""
    if len(profiles) != len(SEGMENT_ORDER):
        raise ValueError("the decision playbook requires exactly six clusters")
    z = _standardize_profiles(profiles)
    remaining = set(int(value) for value in profiles["cluster_id"])
    assignments: dict[int, str] = {}

    def choose(name: str, score: pd.Series) -> None:
        available = score.loc[sorted(remaining)]
        cluster_id = int(available.idxmax())
        assignments[cluster_id] = name
        remaining.remove(cluster_id)

    choose(
        "Dormant low value",
        z["recency_days"] - z["sessions_90d"] - z["orders_12m"] - z["revenue_12m"],
    )
    choose(
        "Discount-led frequent",
        z["discount_order_share"] + z["orders_12m"] - z["margin_rate"],
    )
    choose(
        "Engaged low conversion",
        z["sessions_90d"] + z["category_breadth_12m"] - z["orders_12m"] - z["conversion_rate_90d"],
    )
    choose(
        "High value at risk",
        z["revenue_12m"] + z["recency_days"] - z["sessions_90d"],
    )
    choose(
        "Loyal high value",
        z["revenue_12m"] + z["orders_12m"] + z["margin_rate"] - z["recency_days"],
    )
    assignments[remaining.pop()] = "Growth potential"
    return assignments


def name_profiles(profiles: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Attach business names and present profiles in a stable order."""
    named = profiles.copy()
    named["segment_name"] = named["cluster_id"].map(names)
    named["segment_name"] = pd.Categorical(
        named["segment_name"], categories=SEGMENT_ORDER, ordered=True
    )
    return named.sort_values("segment_name").reset_index(drop=True)


def build_decision_playbook(profiles: pd.DataFrame) -> pd.DataFrame:
    """Create action hypotheses and safeguards for each discovered segment."""
    rows = []
    for row in profiles.itertuples(index=False):
        objective, action, guardrail = DECISIONS[str(row.segment_name)]
        rows.append(
            {
                "segment_name": str(row.segment_name),
                "customer_share": float(row.share),
                "decision_objective": objective,
                "testable_action": action,
                "guardrail": guardrail,
                "impact_status": "Hypothesis; requires a randomized or matched evaluation",
            }
        )
    return pd.DataFrame(rows)
