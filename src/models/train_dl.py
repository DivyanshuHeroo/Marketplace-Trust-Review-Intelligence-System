"""Deep-learning model (TensorFlow / Keras) for negative-review prediction.

This complements the gradient-boosted models with a neural net on the same tabular
features, so the project demonstrates a full ML→DL progression.

Architecture
------------
A compact feed-forward network with BatchNorm + Dropout, trained with class weights to
handle the ~21% positive rate. We use the same **time-based split** as the classic models
and an EarlyStopping callback on validation AUC.

Run
---
    python -m src.models.train_dl
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF info logs
# On macOS, scikit-learn (libomp) and TensorFlow can load two OpenMP runtimes into
# the same process, which deadlocks the training loop at 0% CPU. Allowing the
# duplicate runtime and bounding thread pools makes training run reliably.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
warnings.filterwarnings("ignore")

# IMPORTANT: import TensorFlow *before* scikit-learn. If sklearn (and its libomp)
# initialises first, TF and sklearn end up with two OpenMP runtimes in one process
# and the Keras training loop deadlocks at 0% CPU on macOS. Importing TF first avoids
# this entirely.
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features.build_features import FEATURE_COLUMNS, build_feature_frame  # noqa: E402
from src.utils.config import load_config, models_dir, processed_dir  # noqa: E402



def _load() -> pd.DataFrame:
    path = processed_dir() / "order_features.parquet"
    return pd.read_parquet(path) if path.exists() else build_feature_frame()


def _time_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.1):
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    cut_test = int(n * (1 - test_frac))
    cut_val = int(cut_test * (1 - val_frac))
    tr, va, te = df.iloc[:cut_val], df.iloc[cut_val:cut_test], df.iloc[cut_test:]
    return tr, va, te


def build_model(input_dim: int, seed: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    # Plain Dense + Dropout (no BatchNorm / no in-loop metrics) — this keeps the
    # Keras 3 training step lightweight, which trains reliably on macOS CPU.
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model



def train() -> dict:
    import tensorflow as tf
    from tensorflow import keras

    cfg = load_config()
    seed = cfg["random_seed"]
    np.random.seed(seed)

    df = _load()
    tr, va, te = _time_split(df)

    # cast to float32 — float64 tensors can stall the Keras 3 training loop on macOS
    scaler = StandardScaler().fit(tr[FEATURE_COLUMNS].values)
    X_tr = scaler.transform(tr[FEATURE_COLUMNS].values).astype("float32")
    X_va = scaler.transform(va[FEATURE_COLUMNS].values).astype("float32")
    X_te = scaler.transform(te[FEATURE_COLUMNS].values).astype("float32")
    y_tr = tr["is_negative"].values.astype("float32")
    y_va = va["is_negative"].values.astype("float32")
    y_te = te["is_negative"].values.astype("float32")

    # Handle imbalance by random oversampling of the minority class with numpy.
    # (We deliberately avoid Keras `class_weight` / `sample_weight`, which can stall
    # the Keras 3 training loop on macOS CPU.) Oversampling lets us call `fit` with
    # plain arrays — the configuration that trains reliably here.
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    pos_os = rng.choice(pos_idx, size=len(neg_idx), replace=True)  # up to majority size
    bal_idx = rng.permutation(np.concatenate([neg_idx, pos_os]))
    X_tr_bal = X_tr[bal_idx]
    y_tr_bal = y_tr[bal_idx]



    print("=" * 64)
    print("OlistTrust — deep learning (Keras) negative-review model")
    print("=" * 64)
    print(f"  Train {len(X_tr):,} | Val {len(X_va):,} | Test {len(X_te):,}")

    model = build_model(X_tr.shape[1], seed)

    # Manual epoch loop with early stopping on validation ROC-AUC. Running fit one
    # epoch at a time (no in-loop Keras metrics/callbacks) is what trains reliably on
    # macOS CPU, while still giving us proper early stopping and best-weight restore.
    max_epochs, patience = 40, 6
    best_auc, best_weights, wait, epochs_trained = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.fit(X_tr_bal, y_tr_bal, epochs=1, batch_size=512, shuffle=True, verbose=0)
        val_proba = model.predict(X_va, verbose=0).ravel()

        val_auc = roc_auc_score(y_va, val_proba)
        epochs_trained = epoch
        print(f"  epoch {epoch:2d} — val ROC-AUC {val_auc:.4f}")
        if val_auc > best_auc + 1e-4:
            best_auc, best_weights, wait = val_auc, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stopping (no val improvement for {patience} epochs)")
                break
    if best_weights is not None:
        model.set_weights(best_weights)

    proba = model.predict(X_te, verbose=0).ravel()

    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model": "Keras-MLP",
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "accuracy": round(float((preds == y_te).mean()), 4),
        "best_val_roc_auc": round(float(best_auc), 4),
        "epochs_trained": int(epochs_trained),
    }


    print("\n  Test metrics:")
    for k, v in metrics.items():
        print(f"    {k:16s}: {v}")

    # persist
    model_path = models_dir() / "dl_negative_review.keras"
    model.save(model_path)
    import joblib
    joblib.dump(scaler, models_dir() / "dl_scaler.joblib")
    pd.DataFrame([metrics]).to_csv(processed_dir() / "dl_metrics.csv", index=False)
    print(f"\n  Saved Keras model -> {model_path}")
    print("=" * 64)
    return metrics


if __name__ == "__main__":
    train()
