"""Load the 9 raw Olist CSVs into a single SQLite database (the project's SQL layer).

Why SQLite?
-----------
The Olist dataset is genuinely *relational* (orders ↔ items ↔ products ↔ sellers ↔
reviews ↔ customers). Loading it into SQLite lets us do the heavy joins/aggregations
in real SQL — which is exactly what a data-analyst role expects to see — instead of
hiding everything inside pandas.

After loading we create indexes on the join keys and a couple of convenience VIEWS
that the analytics queries build on.

Run
---
    python -m src.etl.build_database
"""

from __future__ import annotations

import sqlite3
import time

import pandas as pd

from src.utils.config import db_path, load_config, raw_dir

# Columns we parse as dates (speeds up downstream date math).
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
    "order_items": ["shipping_limit_date"],
}

# Indexes to create: table -> list of columns.
INDEXES = {
    "orders": ["order_id", "customer_id", "order_status"],
    "order_items": ["order_id", "product_id", "seller_id"],
    "order_payments": ["order_id"],
    "order_reviews": ["order_id", "review_score"],
    "customers": ["customer_id", "customer_unique_id"],
    "products": ["product_id", "product_category_name"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name"],
}


def _load_csv(table: str, filename: str) -> pd.DataFrame:
    path = raw_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file {path}. Run scripts/download_data.sh first."
        )
    parse_dates = DATE_COLUMNS.get(table)
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df


def build() -> None:
    cfg = load_config()
    out = db_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # rebuild from scratch for reproducibility

    t0 = time.time()
    conn = sqlite3.connect(out)
    print("=" * 64)
    print("OlistTrust — building SQLite database")
    print("=" * 64)

    for table, filename in cfg["tables"].items():
        df = _load_csv(table, filename)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  loaded {table:22s} {len(df):>8,} rows  ({len(df.columns)} cols)")

    # indexes
    cur = conn.cursor()
    for table, cols in INDEXES.items():
        for col in cols:
            idx = f"idx_{table}_{col}"
            cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{table}" ("{col}");')
    conn.commit()
    print("  created indexes on all join keys")

    _create_views(conn)
    print("  created analytical views: v_order_core, v_seller_orders")

    conn.close()
    dt = time.time() - t0
    print(f"\n✅ Database ready at {out}  ({dt:.1f}s)")
    print("=" * 64)


def _create_views(conn: sqlite3.Connection) -> None:
    """Convenience views used by the analytics + feature SQL."""
    cur = conn.cursor()

    # One row per (order, seller) with the customer review + delivery facts.
    cur.execute("DROP VIEW IF EXISTS v_order_core;")
    cur.execute(
        """
        CREATE VIEW v_order_core AS
        SELECT
            o.order_id,
            o.customer_id,
            c.customer_unique_id,
            c.customer_state,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_approved_at,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            r.review_score,
            r.review_comment_message,
            -- delivery in days (NULL if not delivered)
            CAST(julianday(o.order_delivered_customer_date)
                 - julianday(o.order_purchase_timestamp) AS REAL) AS delivery_days,
            -- estimate error: positive = delivered EARLY (good)
            CAST(julianday(o.order_estimated_delivery_date)
                 - julianday(o.order_delivered_customer_date) AS REAL) AS days_vs_estimate
        FROM orders o
        LEFT JOIN customers c       ON o.customer_id = c.customer_id
        LEFT JOIN order_reviews r   ON o.order_id    = r.order_id;
        """
    )

    # One row per (order, seller) so we can aggregate to the seller level.
    cur.execute("DROP VIEW IF EXISTS v_seller_orders;")
    cur.execute(
        """
        CREATE VIEW v_seller_orders AS
        SELECT
            i.seller_id,
            i.order_id,
            i.product_id,
            i.price,
            i.freight_value,
            oc.review_score,
            oc.delivery_days,
            oc.days_vs_estimate,
            oc.order_status
        FROM order_items i
        LEFT JOIN v_order_core oc ON i.order_id = oc.order_id;
        """
    )
    conn.commit()


if __name__ == "__main__":
    build()
