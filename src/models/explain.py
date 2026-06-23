"""SHAP explainability for the negative-review model.

Produces:
  * a global feature-importance table (mean |SHAP|)         -> data/processed/shap_importance.csv
  * a SHAP summary bar chart                                -> reports/figures/shap_importance.png

Tree models (XGBoost / LightGBM) use the fast exact TreeExplainer. For the Logistic
Regression pipeline we fall back to a small KernelExplainer on a sample.

Run
---
    python -m src.models.explain
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.utils.config import figures_dir, models_dir, processed_dir

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402


def _load_bundle():
    path = models_dir() / "negative_review_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the model first: python -m src.models.train_ml")
    return joblib.load(path)


def compute_shap(sample_size: int = 2000) -> pd.DataFrame:
    bundle = _load_bundle()
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_parquet(processed_dir() / "order_features.parquet")
    X = df[features].sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    name = bundle["model_name"]
    if name in ("XGBoost", "LightGBM"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # some versions return [class0, class1]
            shap_values = shap_values[1]
    else:
        # Logistic Regression pipeline -> Kernel explainer on a small background
        f = lambda data: model.predict_proba(data)[:, 1]
        background = shap.sample(X, 100, random_state=42)
        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(X.iloc[:200], nsamples=100)
        X = X.iloc[:200]

    importance = (
        pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_csv = processed_dir() / "shap_importance.csv"
    importance.to_csv(out_csv, index=False)

    # summary bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    fig_path = figures_dir() / "shap_importance.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    print("=" * 64)
    print(f"OlistTrust — SHAP explainability ({name})")
    print("=" * 64)
    print(importance.to_string(index=False))
    print(f"\n  Saved importance table -> {out_csv}")
    print(f"  Saved summary chart    -> {fig_path}")
    print("=" * 64)
    return importance


if __name__ == "__main__":
    compute_shap()
