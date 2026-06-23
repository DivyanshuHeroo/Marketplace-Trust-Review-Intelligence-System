"""End-to-end pipeline runner for OlistTrust.

Runs every stage in order so the whole project reproduces with one command:

    python run_pipeline.py            # full pipeline (skips DL by default = fast)
    python run_pipeline.py --with-dl  # also train the Keras deep-learning model
    python run_pipeline.py --skip-db  # reuse an existing SQLite database

Stages
------
1. Build SQLite database from raw CSVs
2. Build order-level feature matrix
3. Train classic ML models (LogReg / XGBoost / LightGBM)
4. SHAP explainability
5. NLP sentiment (lexicon + TF-IDF)
6. Seller Trust Scores
7. EDA figures
8. (optional) Keras deep-learning model
"""

from __future__ import annotations

import argparse
import time


def _step(n: int, title: str) -> None:
    print(f"\n{'#' * 64}\n# STEP {n}: {title}\n{'#' * 64}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OlistTrust pipeline")
    ap.add_argument("--with-dl", action="store_true", help="train the Keras DL model too")
    ap.add_argument("--skip-db", action="store_true", help="reuse existing database")
    args = ap.parse_args()

    t0 = time.time()

    if not args.skip_db:
        _step(1, "Build SQLite database")
        from src.etl.build_database import build
        build()
    else:
        print("Skipping database build (--skip-db).")

    _step(2, "Build feature matrix")
    from src.features.build_features import build_and_save
    build_and_save()

    _step(3, "Train classic ML models")
    from src.models.train_ml import train as train_ml
    train_ml()

    _step(4, "SHAP explainability")
    from src.models.explain import compute_shap
    compute_shap()

    _step(5, "NLP sentiment")
    from src.nlp.sentiment import main as nlp_main
    nlp_main()

    _step(6, "Seller Trust Scores")
    from src.trust.trust_score import build_and_save as trust_build
    trust_build()

    _step(7, "EDA figures")
    from src.eda import generate_all
    generate_all()

    if args.with_dl:
        _step(8, "Deep learning (Keras)")
        from src.models.train_dl import train as train_dl
        train_dl()

    dt = time.time() - t0
    print(f"\n✅ Pipeline complete in {dt:.1f}s.")
    print("   Launch the dashboard:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
