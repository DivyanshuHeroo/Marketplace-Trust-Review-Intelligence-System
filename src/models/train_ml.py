"""Train + evaluate classic ML models to predict negative reviews.

Models
------
* Logistic Regression  — interpretable linear baseline
* XGBoost              — gradient boosting
* LightGBM             — gradient boosting (usually the best here)

We use a **time-based split** (train on earlier orders, test on later orders) because in
production you always predict the future from the past — a random split would leak temporal
information and inflate scores. Metrics: ROC-AUC, PR-AUC, F1, plus a classification report.

The best model is persisted to models/ for the dashboard + SHAP explainer.

Run
---
    python -m src.models.train_ml
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS, build_feature_frame
from src.utils.config import load_config, models_dir, processed_dir

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False


def _load() -> pd.DataFrame:
    path = processed_dir() / "order_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_feature_frame()


def _time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_tr = train[FEATURE_COLUMNS].values
    X_te = test[FEATURE_COLUMNS].values
    y_tr = train["is_negative"].values
    y_te = test["is_negative"].values
    return X_tr, X_te, y_tr, y_te


def _evaluate(name, model, X_te, y_te) -> dict:
    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "accuracy": round(float((preds == y_te).mean()), 4),
    }


def train() -> dict:
    cfg = load_config()
    seed = cfg["random_seed"]
    df = _load()
    X_tr, X_te, y_tr, y_te = _time_split(df)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    print("=" * 64)
    print("OlistTrust — predicting negative reviews (classic ML)")
    print("=" * 64)
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}  "
          f"(time-based split) | negative rate train={y_tr.mean():.3f}")

    results = []
    models = {}

    # 1) Logistic Regression (scaled)
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    logreg.fit(X_tr, y_tr)
    results.append(_evaluate("LogisticRegression", logreg, X_te, y_te))
    models["LogisticRegression"] = logreg

    # 2) XGBoost
    if _HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=-1, tree_method="hist",
        )
        xgb.fit(X_tr, y_tr)
        results.append(_evaluate("XGBoost", xgb, X_te, y_te))
        models["XGBoost"] = xgb

    # 3) LightGBM
    if _HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=500, num_leaves=48, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        lgbm.fit(X_tr, y_tr)
        results.append(_evaluate("LightGBM", lgbm, X_te, y_te))
        models["LightGBM"] = lgbm

    res_df = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    print("\n  Results (held-out future test set):")
    print(res_df.to_string())

    best_name = res_df.index[0]
    best_model = models[best_name]
    print(f"\n  Best model: {best_name} (ROC-AUC={res_df.loc[best_name,'roc_auc']})")

    proba = best_model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("\n  Classification report (best model):")
    print(classification_report(y_te, preds, target_names=["satisfied", "negative"],
                                zero_division=0))
    print("  Confusion matrix [[TN, FP], [FN, TP]]:",
          confusion_matrix(y_te, preds).tolist())

    # persist best model + metrics + feature list
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURE_COLUMNS,
        "metrics": res_df.reset_index().to_dict(orient="records"),
    }
    out = models_dir() / "negative_review_model.joblib"
    joblib.dump(bundle, out)
    res_df.reset_index().to_csv(processed_dir() / "ml_metrics.csv", index=False)
    print(f"\n  Saved best model -> {out}")
    print("=" * 64)
    return bundle


if __name__ == "__main__":
    train()
