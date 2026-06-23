"""Build the order-level feature matrix used by the ML + DL models.

Prediction task
---------------
**Will this order receive a negative review (review_score <= 3)?**

This is a realistic, business-relevant target: if we can predict dissatisfaction at (or
shortly after) purchase time from delivery, price, freight, product and basket signals,
the marketplace can intervene early (expedite shipping, proactive support, etc.).

We build features almost entirely in **SQL** (joins + aggregations over the relational
tables) and return a tidy modelling frame. The label is derived from the review score.

Run
---
    python -m src.features.build_features
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.utils.config import db_path, load_config, processed_dir

# Feature columns the models consume (order matters for the DL model / SHAP).
FEATURE_COLUMNS = [
    "price_total",
    "freight_total",
    "freight_ratio",
    "n_items",
    "n_sellers",
    "n_categories",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty",
    "product_description_lenght",
    "delivery_days",
    "approval_delay_hours",
    "estimated_delivery_days",
    "purchase_dow",
    "purchase_month",
    "payment_installments",
    "payment_value",
    "seller_avg_review_hist",
    "customer_state_code",
]

_FEATURE_SQL = """
WITH item_agg AS (
    SELECT
        order_id,
        SUM(price)                       AS price_total,
        SUM(freight_value)               AS freight_total,
        COUNT(*)                         AS n_items,
        COUNT(DISTINCT seller_id)        AS n_sellers,
        COUNT(DISTINCT product_id)       AS n_products
    FROM order_items
    GROUP BY order_id
),
prod_agg AS (
    SELECT
        i.order_id,
        AVG(p.product_weight_g)                                              AS product_weight_g,
        AVG(p.product_length_cm * p.product_height_cm * p.product_width_cm)  AS product_volume_cm3,
        AVG(p.product_photos_qty)                                            AS product_photos_qty,
        AVG(p.product_description_lenght)                                    AS product_description_lenght,
        COUNT(DISTINCT p.product_category_name)                              AS n_categories
    FROM order_items i
    LEFT JOIN products p ON i.product_id = p.product_id
    GROUP BY i.order_id
),
pay_agg AS (
    SELECT
        order_id,
        SUM(payment_value)         AS payment_value,
        MAX(payment_installments)  AS payment_installments
    FROM order_payments
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.customer_id,
    c.customer_state,
    ia.price_total,
    ia.freight_total,
    ia.n_items,
    ia.n_sellers,
    pa2.n_categories,
    pa2.product_weight_g,
    pa2.product_volume_cm3,
    pa2.product_photos_qty,
    pa2.product_description_lenght,
    pay.payment_value,
    pay.payment_installments,
    CAST(julianday(o.order_delivered_customer_date)
         - julianday(o.order_purchase_timestamp) AS REAL)        AS delivery_days,
    CAST((julianday(o.order_approved_at)
         - julianday(o.order_purchase_timestamp)) * 24 AS REAL)  AS approval_delay_hours,
    CAST(julianday(o.order_estimated_delivery_date)
         - julianday(o.order_purchase_timestamp) AS REAL)        AS estimated_delivery_days,
    o.order_purchase_timestamp,
    r.review_score
FROM orders o
JOIN item_agg ia        ON o.order_id = ia.order_id
LEFT JOIN prod_agg pa2  ON o.order_id = pa2.order_id
LEFT JOIN pay_agg pay   ON o.order_id = pay.order_id
LEFT JOIN customers c   ON o.customer_id = c.customer_id
JOIN order_reviews r    ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND r.review_score IS NOT NULL;
"""

# Historical seller reputation (avg review over all their orders) — a strong predictor.
_SELLER_HIST_SQL = """
SELECT i.order_id, AVG(seller_hist.avg_score) AS seller_avg_review_hist
FROM order_items i
JOIN (
    SELECT i2.seller_id, AVG(r.review_score) AS avg_score
    FROM order_items i2
    JOIN order_reviews r ON i2.order_id = r.order_id
    GROUP BY i2.seller_id
) seller_hist ON i.seller_id = seller_hist.seller_id
GROUP BY i.order_id;
"""


def build_feature_frame() -> pd.DataFrame:
    cfg = load_config()
    thr = cfg["target"]["negative_score_threshold"]

    with sqlite3.connect(db_path()) as conn:
        df = pd.read_sql_query(_FEATURE_SQL, conn)
        seller_hist = pd.read_sql_query(_SELLER_HIST_SQL, conn)

    df = df.merge(seller_hist, on="order_id", how="left")

    # ---- derived features ----
    df["freight_ratio"] = df["freight_total"] / df["price_total"].replace(0, np.nan)
    ts = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["purchase_dow"] = ts.dt.dayofweek
    df["purchase_month"] = ts.dt.month
    # encode state as an integer code (kept simple + interpretable)
    df["customer_state_code"] = df["customer_state"].astype("category").cat.codes

    # ---- label ----
    df["is_negative"] = (df["review_score"] <= thr).astype(int)

    # ---- clean ----
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # median impute numeric features
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    keep = ["order_id", "customer_state", "review_score", "is_negative",
            "order_purchase_timestamp"] + FEATURE_COLUMNS
    return df[keep]


def build_and_save() -> pd.DataFrame:
    df = build_feature_frame()
    out = processed_dir() / "order_features.parquet"
    df.to_parquet(out, index=False)

    print("=" * 64)
    print("OlistTrust — order feature matrix built")
    print("=" * 64)
    print(f"  Output      : {out}")
    print(f"  Rows        : {len(df):,}")
    print(f"  Features    : {len(FEATURE_COLUMNS)}")
    rate = df["is_negative"].mean()
    print(f"  Negative-review rate (target) : {rate:.3f}")
    print("\n  Mean feature values (satisfied vs negative):")
    preview = ["delivery_days", "estimated_delivery_days", "freight_ratio",
               "seller_avg_review_hist", "price_total"]
    print(df.groupby("is_negative")[preview].mean().round(2).to_string())
    print("=" * 64)
    return df


if __name__ == "__main__":
    build_and_save()
