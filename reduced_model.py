"""
reduced_model.py
XGBoost trained on top SHAP-selected features (post-hoc feature selection).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek

from metrics import compute_metrics
from training import derive_threshold_via_val


def train_reduced_model(result, top_features, n_features=10):
    """
    Train XGBoost on top SHAP-selected features.

    NOTE: SHAP and reduced model share the same training data.
    This is a post-hoc feature selection limitation and should
    be disclosed in the manuscript limitations section.
    """
    target        = result["target"]
    selected_cols = top_features.index.tolist()[:n_features]
    xgb_params    = result["xgb_params"]

    print(f"\n[Reduced Model] {target} -- Top {n_features} features")

    X_train_orig = result["X_train_original"][selected_cols].copy()
    X_test_red   = result["X_test"][selected_cols].copy()
    y_train_orig = result["y_train_original"].copy()
    y_test       = result["y_test"]
    weights      = result["weights_test"]

    red_thr = derive_threshold_via_val(
        xgb.XGBClassifier, xgb_params,
        X_train_orig, y_train_orig
    )

    smt                      = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train_orig, y_train_orig)
    X_train_res              = pd.DataFrame(X_train_res, columns=selected_cols)
    y_train_res              = pd.Series(y_train_res)

    model_red = xgb.XGBClassifier(**xgb_params)
    model_red.fit(X_train_res, y_train_res)

    prob_red    = model_red.predict_proba(X_test_red)[:, 1]
    pred_red    = (prob_red >= red_thr).astype(int)
    metrics_red = compute_metrics(y_test, pred_red, prob_red, weights)

    print(f"  Threshold (val)={red_thr:.3f}  AUC={metrics_red['AUC-ROC']:.4f} | "
          f"Sens={metrics_red['Sensitivity']:.1f}%  Spec={metrics_red['Specificity']:.1f}%")

    return {
        "selected_features" : selected_cols,
        "n_features"        : n_features,
        "metrics"           : metrics_red,
        "model"             : model_red,
        "prob"              : prob_red,
        "threshold"         : red_thr,
    }
