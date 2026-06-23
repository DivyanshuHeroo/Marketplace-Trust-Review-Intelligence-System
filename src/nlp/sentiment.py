"""NLP on the Portuguese review comments.

Olist reviews are in Brazilian Portuguese, so off-the-shelf English sentiment models don't
apply. We take two complementary, dependency-light approaches:

1. **Lexicon sentiment** — a curated Portuguese positive/negative word list with simple
   negation handling ("não recomendo"). Fast, transparent, no training needed.

2. **TF-IDF + Logistic Regression classifier** — trained to predict the (known) review
   score bucket from the comment text. This is a proper supervised NLP model that learns
   which words drive good/bad reviews, and exposes the most predictive terms.

Both are evaluated against the real review_score so we can quote honest metrics.

Run
---
    python -m src.nlp.sentiment
"""

from __future__ import annotations

import re
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.config import db_path, processed_dir

warnings.filterwarnings("ignore")

# ── Portuguese sentiment lexicon (compact, hand-picked) ──────────────────────
POSITIVE_PT = {
    "bom", "boa", "otimo", "ótima", "ótimo", "otima", "excelente", "perfeito", "perfeita",
    "maravilhoso", "maravilhosa", "recomendo", "satisfeito", "satisfeita", "rapido", "rápida",
    "rápido", "rapida", "amei", "adorei", "gostei", "lindo", "linda", "qualidade", "super",
    "feliz", "facil", "fácil", "confiavel", "confiável", "agil", "ágil", "top",
}
NEGATIVE_PT = {
    "ruim", "pessimo", "péssimo", "pessima", "péssima", "horrivel", "horrível", "demora",
    "demorou", "atraso", "atrasado", "atrasada", "nao", "não", "nunca", "quebrado", "quebrada",
    "defeito", "errado", "errada", "decepcao", "decepção", "decepcionado", "pessimamente",
    "estorno", "reclamacao", "reclamação", "problema", "cancelado", "devolver", "devolucao",
    "devolução", "enganado", "lixo", "arrependido", "pior",
}
NEGATORS = {"nao", "não", "nunca", "nem", "jamais"}

_WORD_RE = re.compile(r"[a-zà-ú]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def lexicon_score(text: str) -> float:
    """Return a sentiment score in [-1, 1] using the PT lexicon + negation flip."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    score = 0
    for i, tok in enumerate(tokens):
        val = 0
        if tok in POSITIVE_PT:
            val = 1
        elif tok in NEGATIVE_PT:
            val = -1
        if val != 0 and i > 0 and tokens[i - 1] in NEGATORS:
            val = -val  # negation flips polarity
        score += val
    return float(np.clip(score / np.sqrt(len(tokens)), -1, 1))


def lexicon_label(text: str) -> str:
    s = lexicon_score(text)
    if s > 0.05:
        return "positive"
    if s < -0.05:
        return "negative"
    return "neutral"


def _load_reviews() -> pd.DataFrame:
    sql = """
        SELECT review_score, review_comment_message AS text
        FROM order_reviews
        WHERE review_comment_message IS NOT NULL
          AND TRIM(review_comment_message) <> ''
    """
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(sql, conn)


def evaluate_lexicon(df: pd.DataFrame) -> dict:
    """Check how well the lexicon agrees with the star rating."""
    truth = (df["review_score"] <= 2).astype(int)  # 1 = negative
    pred = df["text"].map(lambda t: 1 if lexicon_score(t) < -0.05 else 0)
    return {
        "lexicon_accuracy": round(accuracy_score(truth, pred), 4),
        "lexicon_f1_negative": round(f1_score(truth, pred, zero_division=0), 4),
    }


def train_tfidf_classifier(df: pd.DataFrame) -> dict:
    """Supervised TF-IDF + LogReg predicting negative (score<=2) from text."""
    y = (df["review_score"] <= 2).astype(int).values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(df["text"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "tfidf_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "tfidf_f1_negative": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "tfidf_accuracy": round(float(accuracy_score(y_te, preds)), 4),
    }

    # most predictive words for each polarity
    coefs = clf.coef_[0]
    vocab = np.array(vec.get_feature_names_out())
    top_neg = vocab[np.argsort(coefs)[-12:]][::-1]
    top_pos = vocab[np.argsort(coefs)[:12]]
    metrics["top_negative_terms"] = list(top_neg)
    metrics["top_positive_terms"] = list(top_pos)
    return metrics


def main() -> None:
    df = _load_reviews()
    print("=" * 64)
    print("OlistTrust — NLP sentiment on Portuguese reviews")
    print("=" * 64)
    print(f"  Reviews with text: {len(df):,}")

    lex = evaluate_lexicon(df)
    tfidf = train_tfidf_classifier(df)

    print("\n  Lexicon (vs star rating):")
    for k in ("lexicon_accuracy", "lexicon_f1_negative"):
        print(f"    {k:22s}: {lex[k]}")

    print("\n  TF-IDF + LogReg classifier:")
    for k in ("tfidf_roc_auc", "tfidf_f1_negative", "tfidf_accuracy"):
        print(f"    {k:22s}: {tfidf[k]}")

    print("\n  Top NEGATIVE terms:", ", ".join(tfidf["top_negative_terms"]))
    print("  Top POSITIVE terms:", ", ".join(tfidf["top_positive_terms"]))

    # save a sample of labelled reviews for the dashboard
    sample = df.sample(min(5000, len(df)), random_state=42).copy()
    sample["lexicon_sentiment"] = sample["text"].map(lexicon_label)
    sample.to_parquet(processed_dir() / "review_sentiment_sample.parquet", index=False)
    print(f"\n  Saved labelled sample -> {processed_dir() / 'review_sentiment_sample.parquet'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
