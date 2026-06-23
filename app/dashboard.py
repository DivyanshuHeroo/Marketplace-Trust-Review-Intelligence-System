"""OlistTrust — interactive Streamlit dashboard.

Five tabs tie the whole project together:
  1. Overview          — headline KPIs + review distribution
  2. Delivery Insights — the delivery→satisfaction story (SQL-powered)
  3. Seller Trust      — searchable Trust Score leaderboard + component breakdown
  4. Review NLP        — Portuguese sentiment explorer + top terms
  5. Risk Predictor    — live negative-review probability from the trained ML model

Run
---
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# make `src` importable when run via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.etl.queries import list_queries, run_query, run_sql  # noqa: E402
from src.nlp.sentiment import lexicon_label, lexicon_score  # noqa: E402
from src.utils.config import db_path, load_config, models_dir, processed_dir  # noqa: E402

st.set_page_config(page_title="OlistTrust", page_icon="🛒", layout="wide")


# ──────────────────────────── data loaders (cached) ─────────────────────────
@st.cache_data(show_spinner=False)
def q(name: str) -> pd.DataFrame:
    return run_query(name)


@st.cache_data(show_spinner=False)
def load_trust() -> pd.DataFrame:
    p = processed_dir() / "seller_trust_scores.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sentiment_sample() -> pd.DataFrame:
    p = processed_dir() / "review_sentiment_sample.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ml_metrics() -> pd.DataFrame:
    p = processed_dir() / "ml_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    p = models_dir() / "negative_review_model.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def overview_kpis() -> dict:
    with sqlite3.connect(db_path()) as conn:
        cur = conn.cursor()
        out = {}
        out["orders"] = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        out["sellers"] = cur.execute("SELECT COUNT(*) FROM sellers").fetchone()[0]
        out["customers"] = cur.execute(
            "SELECT COUNT(DISTINCT customer_unique_id) FROM customers").fetchone()[0]
        out["avg_score"] = cur.execute(
            "SELECT AVG(review_score) FROM order_reviews").fetchone()[0]
        out["revenue"] = cur.execute("SELECT SUM(price) FROM order_items").fetchone()[0]
    return out


# ──────────────────────────────── header ───────────────────────────────────
st.title("🛒 OlistTrust — Marketplace Trust & Review Intelligence")
st.caption(
    "End-to-end analytics on the Brazilian Olist e-commerce dataset · "
    "SQL · ML (XGBoost/LightGBM) · Deep Learning (Keras) · NLP · SHAP"
)

tab_overview, tab_delivery, tab_trust, tab_nlp, tab_predict, tab_sql = st.tabs(
    ["📊 Overview", "🚚 Delivery Insights", "🏅 Seller Trust",
     "💬 Review NLP", "🤖 Risk Predictor", "🧪 SQL Explorer"]
)


# ──────────────────────────────── Overview ─────────────────────────────────
with tab_overview:
    k = overview_kpis()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{k['orders']:,}")
    c2.metric("Sellers", f"{k['sellers']:,}")
    c3.metric("Customers", f"{k['customers']:,}")
    c4.metric("Avg review", f"{k['avg_score']:.2f} ★")
    c5.metric("GMV (BRL)", f"{k['revenue']/1e6:.1f}M")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        df = q("review_score_distribution")
        fig = px.bar(df, x="review_score", y="n_reviews", text="pct",
                     title="Review Score Distribution",
                     labels={"review_score": "Stars", "n_reviews": "Reviews"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df = q("monthly_orders_revenue")
        fig = px.line(df, x="month", y="orders", markers=True,
                      title="Monthly Order Volume")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top categories by revenue")
    cats = q("top_categories").head(12)
    fig = px.bar(cats, x="revenue", y="category", orientation="h",
                 color="avg_review_score", color_continuous_scale="RdYlGn",
                 labels={"revenue": "Revenue (BRL)", "category": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────── Delivery Insights ────────────────────────────
with tab_delivery:
    st.subheader("Late delivery is the #1 driver of bad reviews")
    df = q("delivery_vs_satisfaction")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="delivery_bucket", y="avg_review_score",
                     color="avg_review_score", color_continuous_scale="RdYlGn",
                     title="Avg review score by delivery timeliness")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df, x="delivery_bucket", y="pct_negative",
                     color="pct_negative", color_continuous_scale="Reds",
                     title="% negative reviews by delivery timeliness")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("State performance")
    sp = q("state_performance")
    fig = px.scatter(sp, x="avg_delivery_days", y="avg_review_score",
                     size="orders", color="avg_review_score",
                     color_continuous_scale="RdYlGn", text="state",
                     title="Delivery time vs satisfaction by state (bubble = volume)")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────── Seller Trust ──────────────────────────────
with tab_trust:
    trust = load_trust()
    if trust.empty:
        st.warning("Run `python -m src.trust.trust_score` to generate Trust Scores.")
    else:
        st.subheader("Seller Trust Score leaderboard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sellers scored", f"{len(trust):,}")
        c2.metric("Mean trust score", f"{trust['trust_score'].mean():.1f}")
        c3.metric("A-grade sellers",
                  f"{(trust['trust_grade'].str.startswith('A')).sum():,}")

        grades = ["All"] + sorted(trust["trust_grade"].unique().tolist())
        sel = st.selectbox("Filter by grade", grades)
        view = trust if sel == "All" else trust[trust["trust_grade"] == sel]
        show_cols = ["seller_id", "seller_state", "n_orders", "avg_review_score",
                     "on_time_rate", "trust_score", "trust_grade"]
        st.dataframe(view[show_cols].head(200), use_container_width=True)

        st.divider()
        st.subheader("Inspect a seller's Trust Score breakdown")
        sid = st.selectbox("Seller", trust["seller_id"].head(300).tolist())
        row = trust[trust["seller_id"] == sid].iloc[0]
        cfg = load_config()
        comp_cols = {f"c_{k}": k for k in cfg["trust_score"]["weights"]}
        comp = pd.DataFrame({
            "component": list(comp_cols.values()),
            "normalized_value": [row[c] for c in comp_cols],
            "weight": [cfg["trust_score"]["weights"][v] for v in comp_cols.values()],
        })
        comp["contribution"] = (comp["normalized_value"] * comp["weight"] * 100).round(2)
        cc1, cc2 = st.columns([1, 2])
        cc1.metric("Trust score", f"{row['trust_score']:.1f}")
        cc1.write(f"**Grade:** {row['trust_grade']}")
        cc1.write(f"**Orders:** {int(row['n_orders'])}")
        fig = px.bar(comp, x="contribution", y="component", orientation="h",
                     title="Points contributed by each component",
                     color="contribution", color_continuous_scale="Blues")
        cc2.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────── Review NLP ────────────────────────────────
with tab_nlp:
    st.subheader("Portuguese review sentiment")
    sample = load_sentiment_sample()
    if not sample.empty:
        dist = sample["lexicon_sentiment"].value_counts().reset_index()
        dist.columns = ["sentiment", "count"]
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.pie(dist, names="sentiment", values="count",
                         title="Lexicon sentiment mix",
                         color="sentiment",
                         color_discrete_map={"positive": "#1a9850",
                                             "neutral": "#fee08b",
                                             "negative": "#d73027"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            byscore = sample.groupby("review_score")["text"].count().reset_index()
            byscore.columns = ["review_score", "n"]
            fig = px.bar(byscore, x="review_score", y="n",
                         title="Reviews with text, by star rating")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Try the sentiment scorer")
    txt = st.text_area("Enter a Portuguese review",
                       "Produto excelente, entrega rápida, recomendo!")
    if txt.strip():
        score = lexicon_score(txt)
        label = lexicon_label(txt)
        color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[label]
        st.write(f"### {color} {label.upper()}  (score = {score:+.2f})")


# ────────────────────────────── Risk Predictor ─────────────────────────────
with tab_predict:
    st.subheader("🤖 Negative-review risk predictor")
    bundle = load_model_bundle()
    metrics = load_ml_metrics()
    if not metrics.empty:
        st.caption("Model comparison (held-out future test set)")
        st.dataframe(metrics, use_container_width=True)

    if bundle is None:
        st.warning("Train the model first: `python -m src.models.train_ml`")
    else:
        st.write(f"**Active model:** {bundle['model_name']}")
        st.write("Adjust the key order attributes to estimate the risk of a negative review:")
        c1, c2, c3 = st.columns(3)
        delivery_days = c1.slider("Delivery days", 1, 60, 12)
        est_days = c2.slider("Estimated delivery days", 5, 60, 24)
        seller_hist = c3.slider("Seller avg review (history)", 1.0, 5.0, 4.1, 0.1)
        price = c1.slider("Order price (BRL)", 10, 2000, 120)
        freight = c2.slider("Freight (BRL)", 0, 200, 20)
        n_items = c3.slider("Items in order", 1, 10, 1)

        # build a feature row using dataset medians, then override the sliders
        feat = bundle["features"]
        med = pd.read_parquet(processed_dir() / "order_features.parquet")[feat].median()
        row = med.copy()
        row["delivery_days"] = delivery_days
        row["estimated_delivery_days"] = est_days
        row["seller_avg_review_hist"] = seller_hist
        row["price_total"] = price
        row["freight_total"] = freight
        row["freight_ratio"] = freight / max(price, 1)
        row["n_items"] = n_items
        proba = float(bundle["model"].predict_proba(row.values.reshape(1, -1))[:, 1][0])

        st.metric("Predicted negative-review probability", f"{proba*100:.1f}%")
        if proba > 0.5:
            st.error("⚠️ High risk — consider proactive intervention (expedite shipping / support).")
        elif proba > 0.3:
            st.warning("Moderate risk.")
        else:
            st.success("✅ Low risk.")

    # SHAP global importance
    shap_csv = processed_dir() / "shap_importance.csv"
    if shap_csv.exists():
        st.divider()
        st.subheader("Why the model decides — global SHAP importance")
        imp = pd.read_csv(shap_csv).head(12)
        fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                     color="mean_abs_shap", color_continuous_scale="Viridis")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────── SQL Explorer ──────────────────────────────
with tab_sql:
    st.subheader("🧪 SQL Explorer")
    st.caption("Run the curated analytical queries or write your own against the SQLite DB.")
    named = st.selectbox("Saved query", list_queries())
    if st.button("Run saved query"):
        st.dataframe(run_query(named), use_container_width=True)

    st.divider()
    custom = st.text_area("Custom SQL (SELECT only)",
                          "SELECT order_status, COUNT(*) AS n FROM orders GROUP BY order_status;")
    if st.button("Execute custom SQL"):
        try:
            if custom.strip().lower().startswith("select") or "with" in custom.strip().lower()[:5]:
                st.dataframe(run_sql(custom), use_container_width=True)
            else:
                st.error("Only SELECT/WITH queries are allowed.")
        except Exception as e:
            st.error(f"Query error: {e}")
