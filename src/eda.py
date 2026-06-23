"""Exploratory data analysis — generate the key charts as PNGs.

Saves a set of publication-quality figures to reports/figures/ that tell the project's
data story: satisfaction distribution, the delivery→satisfaction relationship, category
performance, geographic patterns and the seller Trust Score distribution.

Run
---
    python -m src.eda
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.etl.queries import run_query  # noqa: E402
from src.utils.config import figures_dir, processed_dir  # noqa: E402
import pandas as pd  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, name: str) -> str:
    path = figures_dir() / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_review_distribution() -> str:
    df = run_query("review_score_distribution")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="review_score", y="n_reviews", ax=ax, color="#4C72B0")
    for _, r in df.iterrows():
        ax.text(r["review_score"] - 1, r["n_reviews"], f'{r["pct"]:.0f}%',
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Customer Review Score Distribution", fontweight="bold")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    return _save(fig, "01_review_distribution.png")


def fig_delivery_vs_satisfaction() -> str:
    df = run_query("delivery_vs_satisfaction")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="delivery_bucket", y="avg_review_score", ax=ax1, color="#55A868")
    ax1.set_title("Avg review score by delivery timeliness", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Avg review score")
    ax1.tick_params(axis="x", rotation=20)

    sns.barplot(data=df, x="delivery_bucket", y="pct_negative", ax=ax2, color="#C44E52")
    ax2.set_title("% negative reviews by delivery timeliness", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("% negative (1–2 stars)")
    ax2.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, "02_delivery_vs_satisfaction.png")


def fig_top_categories() -> str:
    df = run_query("top_categories").head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=df, y="category", x="revenue", ax=ax, color="#4C72B0")
    ax.set_title("Top 12 categories by revenue", fontweight="bold")
    ax.set_xlabel("Revenue (BRL)")
    ax.set_ylabel("")
    return _save(fig, "03_top_categories.png")


def fig_state_performance() -> str:
    df = run_query("state_performance").head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(df["avg_delivery_days"], df["avg_review_score"],
                    s=df["orders"] / 50, alpha=0.6, c=df["avg_review_score"],
                    cmap="RdYlGn")
    for _, r in df.iterrows():
        ax.annotate(r["state"], (r["avg_delivery_days"], r["avg_review_score"]),
                    fontsize=8, ha="center")
    ax.set_title("State performance: delivery time vs satisfaction\n(bubble size = order volume)",
                 fontweight="bold")
    ax.set_xlabel("Avg delivery days")
    ax.set_ylabel("Avg review score")
    fig.colorbar(sc, ax=ax, label="Avg review score")
    return _save(fig, "04_state_performance.png")


def fig_trust_distribution() -> str:
    path = processed_dir() / "seller_trust_scores.parquet"
    if not path.exists():
        return ""
    df = pd.read_parquet(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["trust_score"], bins=30, kde=True, ax=ax1, color="#4C72B0")
    ax1.set_title("Seller Trust Score distribution", fontweight="bold")
    ax1.set_xlabel("Trust score (0–100)")

    order = ["A — Highly trusted", "B — Trusted", "C — Mixed", "D — Risky", "F — High risk"]
    counts = df["trust_grade"].value_counts().reindex(order).fillna(0)
    sns.barplot(x=counts.values, y=counts.index, ax=ax2,
                palette=["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"])
    ax2.set_title("Sellers by Trust Grade", fontweight="bold")
    ax2.set_xlabel("Number of sellers")
    fig.tight_layout()
    return _save(fig, "05_trust_distribution.png")


def generate_all() -> list[str]:
    print("=" * 64)
    print("OlistTrust — generating EDA figures")
    print("=" * 64)
    paths = [
        fig_review_distribution(),
        fig_delivery_vs_satisfaction(),
        fig_top_categories(),
        fig_state_performance(),
        fig_trust_distribution(),
    ]
    for p in paths:
        if p:
            print(f"  saved {p}")
    print("=" * 64)
    return paths


if __name__ == "__main__":
    generate_all()
