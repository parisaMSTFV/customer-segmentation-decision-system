"""Reproducible visual and HTML reporting."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

COLORS = {
    "navy": "#243B53",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#E76F51",
    "blue": "#4C78A8",
    "gray": "#7D8597",
    "ivory": "#F6F1E7",
}
PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A"]


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_model_selection(
    candidates: pd.DataFrame,
    selected_cluster_count: int,
    output_path: Path,
) -> None:
    """Plot the two development-only selection criteria."""
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.set_facecolor("#FCFAF5")
    axis.plot(
        candidates["cluster_count"],
        candidates["silhouette"],
        marker="o",
        linewidth=2.5,
        color=COLORS["teal"],
        label="Mean development silhouette",
    )
    axis.plot(
        candidates["cluster_count"],
        candidates["seed_stability_ari"],
        marker="o",
        linewidth=2.5,
        color=COLORS["navy"],
        label="Mean seed stability (ARI)",
    )
    selected = candidates[candidates["cluster_count"] == selected_cluster_count].iloc[0]
    axis.scatter(
        [selected_cluster_count],
        [selected["silhouette"]],
        s=170,
        facecolor=COLORS["gold"],
        edgecolor=COLORS["navy"],
        linewidth=1.5,
        zorder=4,
        label=f"Selected k = {selected_cluster_count}",
    )
    axis.set_xticks(candidates["cluster_count"])
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Candidate number of clusters")
    axis.set_ylabel("Score")
    axis.set_title("Model selection uses development customers only", loc="left", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    _save(fig, output_path)


def plot_evaluation_summary(metrics: dict[str, float], output_path: Path) -> None:
    """Compare baseline, enhanced recovery, holdout separation, and stability."""
    labels = [
        "RFM baseline\ntruth ARI",
        "Enhanced\ntruth ARI",
        "Holdout\nsilhouette",
        "Bootstrap\nstability ARI",
    ]
    values = [
        metrics["rfm_baseline_synthetic_truth_ari"],
        metrics["enhanced_synthetic_truth_ari"],
        metrics["holdout_silhouette"],
        metrics["bootstrap_mean_pairwise_ari"],
    ]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.set_facecolor("#FCFAF5")
    bars = axis.bar(
        labels, values, color=[COLORS["gray"], COLORS["teal"], COLORS["blue"], COLORS["navy"]]
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, weight="bold")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score (higher is better)")
    axis.set_title("Synthetic holdout evaluation and robustness", loc="left", weight="bold")
    axis.text(
        0,
        -0.17,
        "Truth ARI exists only for this synthetic benchmark; silhouette and stability "
        "transfer to unlabeled data.",
        transform=axis.transAxes,
        color=COLORS["gray"],
    )
    axis.grid(axis="y", alpha=0.2)
    _save(fig, output_path)


def plot_segment_map(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    name_map: dict[int, str],
    output_path: Path,
) -> None:
    """Project holdout customers to two dimensions for a diagnostic view."""
    points = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.set_facecolor("#FCFAF5")
    for color, cluster_id in zip(PALETTE, sorted(name_map), strict=True):
        mask = labels == cluster_id
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=18,
            alpha=0.55,
            color=color,
            linewidth=0,
            label=name_map[cluster_id],
        )
        center = points[mask].mean(axis=0)
        axis.text(
            center[0],
            center[1],
            name_map[cluster_id],
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": color,
            },
        )
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.set_title("Holdout customer map (diagnostic projection)", loc="left", weight="bold")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(alpha=0.15)
    _save(fig, output_path)


def plot_segment_profiles(profiles: pd.DataFrame, output_path: Path) -> None:
    """Plot relative feature profiles for business interpretation."""
    feature_columns = [
        "recency_days",
        "orders_12m",
        "revenue_12m",
        "margin_rate",
        "sessions_90d",
        "conversion_rate_90d",
        "discount_order_share",
        "category_breadth_12m",
        "return_rate",
        "satisfaction_score",
    ]
    raw = profiles.set_index("segment_name")[feature_columns]
    standard_deviation = raw.std(ddof=0).replace(0, 1)
    normalized = (raw - raw.mean()) / standard_deviation
    labels = [
        "Recency",
        "Orders",
        "Revenue",
        "Margin rate",
        "Sessions",
        "Conversion",
        "Discount share",
        "Category breadth",
        "Return rate",
        "Satisfaction",
    ]
    fig, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    image = axis.imshow(normalized, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(normalized.index)), normalized.index)
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 1.0 else COLORS["navy"],
            )
    axis.set_title("Relative holdout segment profiles (column z-scores)", loc="left", weight="bold")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Relative to segment means")
    _save(fig, output_path)


def plot_decision_playbook(playbook: pd.DataFrame, output_path: Path) -> None:
    """Render the decision hypotheses and safeguards as a compact review table."""
    table = playbook[["segment_name", "decision_objective", "testable_action", "guardrail"]].copy()
    for column, width in (
        ("segment_name", 20),
        ("decision_objective", 25),
        ("testable_action", 36),
        ("guardrail", 42),
    ):
        table[column] = table[column].map(
            lambda value, wrap_width=width: "\n".join(textwrap.wrap(str(value), width=wrap_width))
        )
    fig, axis = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.axis("off")
    rendered = axis.table(
        cellText=table.values,
        colLabels=["Segment", "Decision objective", "Testable action", "Guardrail"],
        loc="center",
        cellLoc="left",
        colWidths=[0.17, 0.21, 0.28, 0.34],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10)
    rendered.scale(1, 3.6)
    for (row, _column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#D8D2C4")
        if row == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FCFAF5" if row % 2 else "#F0EBE0")
            cell.set_text_props(va="center")
    axis.set_title(
        "Segment activation hypotheses — measurement required",
        loc="left",
        weight="bold",
        pad=18,
        fontsize=17,
    )
    _save(fig, output_path)


def write_decision_brief(
    metrics: dict[str, float],
    profiles: pd.DataFrame,
    playbook: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a portable, escaped HTML review artifact."""
    profile_view = profiles[
        [
            "segment_name",
            "customers",
            "share",
            "orders_12m",
            "revenue_12m",
            "recency_days",
            "sessions_90d",
            "margin_rate",
        ]
    ].copy()
    profile_view["share"] = profile_view["share"].map(lambda value: f"{value:.1%}")
    profile_view["revenue_12m"] = profile_view["revenue_12m"].map(lambda value: f"{value:,.0f}")
    profile_view["margin_rate"] = profile_view["margin_rate"].map(lambda value: f"{value:.1%}")
    metric_cards = "".join(
        f"<div class='card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in (
            ("Selected clusters", f"{int(metrics['selected_cluster_count'])}"),
            ("Holdout truth ARI", f"{metrics['enhanced_synthetic_truth_ari']:.3f}"),
            ("RFM baseline ARI", f"{metrics['rfm_baseline_synthetic_truth_ari']:.3f}"),
            ("Bootstrap stability", f"{metrics['bootstrap_mean_pairwise_ari']:.3f}"),
        )
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic Customer Segmentation Decision Brief</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:0;background:#f6f1e7;color:#243b53}}
main{{max-width:1180px;margin:auto;padding:40px}}
h1{{margin-bottom:6px}}
.boundary{{background:#fff3cd;border-left:5px solid #e9c46a;padding:14px 18px;margin:22px 0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}
.card{{background:white;padding:18px;border-radius:10px}}
.card span{{display:block;color:#66788a;font-size:13px}}
.card strong{{font-size:26px}}
table{{border-collapse:collapse;width:100%;background:white;margin:16px 0 34px}}
th,td{{padding:11px;border-bottom:1px solid #e8e2d6;text-align:left;vertical-align:top}}
th{{background:#243b53;color:white}}h2{{margin-top:34px}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:22px}}}}
</style></head><body><main>
<h1>Synthetic Customer Segmentation Decision Brief</h1>
<p>Behavioral and value segmentation evaluated on held-out synthetic customers.</p>
<div class="boundary"><strong>Decision boundary:</strong> segment actions are hypotheses for
controlled tests. No campaign lift, ROI improvement, or causal impact is claimed.</div>
<div class="cards">{metric_cards}</div>
<h2>Holdout profiles</h2>{profile_view.to_html(index=False, escape=True, border=0)}
<h2>Activation playbook</h2>{playbook.to_html(index=False, escape=True, border=0)}
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
