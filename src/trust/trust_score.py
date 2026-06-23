"""Seller Trust Score — a transparent, weighted composite metric.

The "novel" deliverable of the project: a single 0–100 score per seller that blends the
signals customers actually care about, each normalized to [0, 1] and combined with the
weights in config.yaml. Unlike a black-box model, every component is interpretable, so we
can show *why* a seller is trusted or risky.

Components (all higher = better)
--------------------------------
* avg_review_score      — mean star rating (1–5 → 0–1)
* on_time_delivery_rate — share of orders delivered on/before the estimate
* low_complaint_rate    — 1 − share of 1–2 star reviews
* delivery_speed        — faster-than-estimate margin (normalized)
* order_volume          — log order count (experience / reliability)
* cancel_rate_penalty   — 1 − cancellation rate

Run
---
    python -m src.trust.trust_score
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.utils.config import db_path, load_config, processed_dir

_SELLER_SQL = """
WITH seller_orders AS (
    SELECT
        i.seller_id,
        i.order_id,
        r.review_score,
        o.order_status,
        CAST(julianday(o.order_estimated_delivery_date)
             - julianday(o.order_delivered_customer_date) AS REAL) AS days_vs_estimate
    FROM order_items i
    JOIN orders o        ON i.order_id = o.order_id
    LEFT JOIN order_reviews r ON i.order_id = r.order_id
    GROUP BY i.seller_id, i.order_id
)
SELECT
    so.seller_id,
    s.seller_state,
    COUNT(DISTINCT so.order_id)                                              AS n_orders,
    AVG(so.review_score)                                                     AS avg_review_score,
    AVG(CASE WHEN so.days_vs_estimate >= 0 THEN 1.0 ELSE 0.0 END)            AS on_time_rate,
    AVG(CASE WHEN so.review_score <= 2 THEN 1.0 ELSE 0.0 END)               AS complaint_rate,
    AVG(so.days_vs_estimate)                                                 AS avg_days_early,
    AVG(CASE WHEN so.order_status = 'canceled' THEN 1.0 ELSE 0.0 END)        AS cancel_rate
FROM seller_orders so
LEFT JOIN sellers s ON so.seller_id = s.seller_id
GROUP BY so.seller_id
HAVING n_orders >= 5;
"""


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_trust_scores() -> pd.DataFrame:
    cfg = load_config()
    w = cfg["trust_score"]["weights"]

    with sqlite3.connect(db_path()) as conn:
        df = pd.read_sql_query(_SELLER_SQL, conn)

    df = df.dropna(subset=["avg_review_score"]).reset_index(drop=True)

    # ---- normalize each component to [0, 1] (higher = better) ----
    comp = pd.DataFrame(index=df.index)
    comp["avg_review_score"] = (df["avg_review_score"] - 1) / 4.0          # 1..5 -> 0..1
    comp["on_time_delivery_rate"] = df["on_time_rate"].clip(0, 1)
    comp["low_complaint_rate"] = 1.0 - df["complaint_rate"].clip(0, 1)
    # avg_days_early can be NULL if a seller's orders were never delivered;
    # treat missing delivery info as the worst observed value before normalizing.
    days_early = df["avg_days_early"].clip(-10, 20)
    comp["delivery_speed"] = _minmax(days_early.fillna(days_early.min()))
    comp["order_volume"] = _minmax(np.log1p(df["n_orders"]))
    comp["cancel_rate_penalty"] = 1.0 - df["cancel_rate"].clip(0, 1)

    comp = comp.fillna(0.0)  # any residual NaN -> 0 (worst) so scores stay valid


    # ---- weighted sum -> 0..100 ----
    score = sum(comp[k] * w[k] for k in w)
    df["trust_score"] = (100 * score).round(2)

    # attach normalized components (handy for the dashboard explanation)
    for k in w:
        df[f"c_{k}"] = comp[k].round(4)

    # ---- letter grade ----
    grades = cfg["trust_score"]["grades"]

    def grade(v: float) -> str:
        for g in grades:
            if v >= g["min"]:
                return g["label"]
        return grades[-1]["label"]

    df["trust_grade"] = df["trust_score"].map(grade)
    df = df.sort_values("trust_score", ascending=False).reset_index(drop=True)
    return df


def build_and_save() -> pd.DataFrame:
    df = compute_trust_scores()
    out = processed_dir() / "seller_trust_scores.parquet"
    df.to_parquet(out, index=False)

    print("=" * 64)
    print("OlistTrust — Seller Trust Scores")
    print("=" * 64)
    print(f"  Sellers scored: {len(df):,}  (min 5 orders)")
    print(f"  Output        : {out}")
    print(f"\n  Score distribution: mean={df['trust_score'].mean():.1f}  "
          f"median={df['trust_score'].median():.1f}  "
          f"min={df['trust_score'].min():.1f}  max={df['trust_score'].max():.1f}")
    print("\n  Grade breakdown:")
    print(df["trust_grade"].value_counts().sort_index().to_string())
    print("\n  Top 5 most-trusted sellers:")
    cols = ["seller_id", "seller_state", "n_orders", "avg_review_score",
            "on_time_rate", "trust_score", "trust_grade"]
    print(df[cols].head(5).to_string(index=False))
    print("\n  Bottom 5 (highest risk):")
    print(df[cols].tail(5).to_string(index=False))
    print("=" * 64)
    return df


if __name__ == "__main__":
    build_and_save()
