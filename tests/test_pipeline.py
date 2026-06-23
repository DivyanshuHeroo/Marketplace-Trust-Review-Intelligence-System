"""Smoke + unit tests for the OlistTrust pipeline.

These run quickly and validate the core building blocks. Stages that need the SQLite
database are skipped automatically if it hasn't been built yet, so the suite is safe to
run in any state.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import db_path, load_config  # noqa: E402

DB_EXISTS = db_path().exists()
needs_db = pytest.mark.skipif(not DB_EXISTS, reason="database not built yet")


# ───────────────────────────── config / pure logic ─────────────────────────
def test_config_loads_and_weights_sum_to_one():
    cfg = load_config()
    weights = cfg["trust_score"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "trust weights must sum to 1.0"


def test_lexicon_sentiment_polarity():
    from src.nlp.sentiment import lexicon_label, lexicon_score
    assert lexicon_score("produto excelente, recomendo, amei") > 0
    assert lexicon_score("péssimo, horrível, não recomendo") < 0
    assert lexicon_label("produto excelente recomendo") == "positive"
    assert lexicon_label("produto péssimo horrível") == "negative"


def test_negation_flips_sentiment():
    from src.nlp.sentiment import lexicon_score
    # "não recomendo" should not be scored as positive
    assert lexicon_score("não recomendo") <= 0


def test_named_queries_parse():
    from src.etl.queries import list_queries
    names = list_queries()
    assert "delivery_vs_satisfaction" in names
    assert "seller_leaderboard" in names
    assert len(names) >= 6


# ──────────────────────────────── DB-backed ────────────────────────────────
@needs_db
def test_database_has_core_tables():
    import sqlite3
    with sqlite3.connect(db_path()) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ["orders", "order_items", "order_reviews", "products", "sellers"]:
        assert t in tables


@needs_db
def test_analytics_query_runs():
    from src.etl.queries import run_query
    df = run_query("review_score_distribution")
    assert len(df) == 5
    assert df["n_reviews"].sum() > 90000


@needs_db
def test_feature_frame_shape_and_target():
    from src.features.build_features import FEATURE_COLUMNS, build_feature_frame
    df = build_feature_frame()
    assert len(df) > 50000
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert set(df["is_negative"].unique()).issubset({0, 1})
    assert df[FEATURE_COLUMNS].isna().sum().sum() == 0  # fully imputed


@needs_db
def test_trust_scores_in_range():
    from src.trust.trust_score import compute_trust_scores
    df = compute_trust_scores()
    assert df["trust_score"].between(0, 100).all()
    assert df["trust_score"].isna().sum() == 0
    assert "trust_grade" in df.columns
