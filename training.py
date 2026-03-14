"""
training.py
Model training, threshold derivation, cross-validation, and main pipeline.
"""

import time
import numpy as np
import pandas as pd

from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, train_test_split
)
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

from config import OUTPUT_DIR, WEIGHT_COL, VALIDATION_NOTE
from metrics import find_optimal_threshold, compute_metrics
from statistics import compare_models_statistically
from tuning import tune_xgb, tune_lgb


def derive_threshold_via_val(model_cls, model_params, X_train_full, y_train_full,
                              val_size=0.2, use_scaler=False):
    """
    Youden threshold from a held-out validation split — no data leakage.
    use_scaler: set True for LogisticRegression.
    """
    feature_names = list(X_train_full.columns)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, stratify=y_train_full, random_state=42
    )

    smt = SMOTETomek(random_state=42)
    X_tr_res, y_tr_res = smt.fit_resample(X_tr, y_tr)
    X_tr_res = pd.DataFrame(X_tr_res, columns=feature_names)
    X_val    = pd.DataFrame(X_val.values, columns=feature_names)
    y_tr_res = pd.Series(y_tr_res)

    if use_scaler:
        scaler   = StandardScaler()
        X_tr_res = pd.DataFrame(scaler.fit_transform(X_tr_res), columns=feature_names)
        X_val    = pd.DataFrame(scaler.transform(X_val), columns=feature_names)

    model_for_thr = model_cls(**model_params)
    model_for_thr.fit(X_tr_res, y_tr_res)

    prob_val  = model_for_thr.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val.values, prob_val)

    return float(threshold)


def select_best_model_by_cv(models_map, X, y, n_splits=5):
    """
    Stratified k-fold CV for all models.
    LogisticRegression is wrapped with StandardScaler inside ImbPipeline
    to ensure convergence without modifying model architecture.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_aucs = {}

    for name, model in models_map.items():
        if name == "Logistic Regression":
            clf_step = SklearnPipeline([
                ("scaler", StandardScaler()),
                ("lr", model),
            ])
        else:
            clf_step = model

        pipeline = ImbPipeline([
            ("resampler", SMOTETomek(random_state=42)),
            ("clf", clf_step),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_aucs[name] = float(scores.mean())
        print(f"  CV AUC [{name}]: {scores.mean():.4f} +/- {scores.std():.4f}")

    best = max(cv_aucs, key=cv_aucs.get)
    print(f"  Best by CV: {best}")
    return best, cv_aucs


def compute_cv_metrics(model_cls, model_params, X, y, n_splits=5, n_repeats=5,
                       use_scaler=False):
    """
    Repeated stratified k-fold CV with SMOTE-Tomek inside each fold.
    Youden threshold is derived from each validation fold — no leakage.
    use_scaler: set True for LogisticRegression.
    """
    rskf    = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    records = []

    for train_idx, val_idx in rskf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smt                = SMOTETomek(random_state=42)
        X_tr_res, y_tr_res = smt.fit_resample(X_tr, y_tr)
        X_tr_res           = pd.DataFrame(X_tr_res, columns=X.columns)

        if use_scaler:
            scaler   = StandardScaler()
            X_tr_res = pd.DataFrame(scaler.fit_transform(X_tr_res), columns=X.columns)
            X_val    = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

        model = model_cls(**model_params)
        model.fit(X_tr_res, y_tr_res)

        prob_val  = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, prob_val)
        pred      = (prob_val >= threshold).astype(int)
        records.append(compute_metrics(y_val, pred, prob_val))

    cv_df   = pd.DataFrame(records)
    mean_df = cv_df.mean().round(4).rename(lambda c: f"{c}_mean")
    std_df  = cv_df.std().round(4).rename(lambda c: f"{c}_std")
    return pd.concat([mean_df, std_df]).to_frame().T


def run_pipeline(train_df, test_df, feature_cols, target, use_external_test=True, n_trials=30):
    """
    Full training pipeline for a single target (GAD or SAD).
    Trains LR, RF, XGBoost, LightGBM with SMOTE-Tomek and Optuna tuning.
    """
    t_pipeline = time.time()

    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")

    if use_external_test:
        X_train_full = train_df[feature_cols].copy()
        y_train_full = train_df[target].copy()
        X_test       = test_df[feature_cols].copy()
        y_test       = test_df[target].copy()
        weights_test = test_df[WEIGHT_COL].values if WEIGHT_COL in test_df.columns else None
        print(f"\nUsing external test set")
    else:
        print(f"\n{'!'*60}")
        print("VALIDATION NOTE:")
        print(VALIDATION_NOTE)
        print(f"{'!'*60}\n")
        X_full = train_df[feature_cols].copy()
        y_full = train_df[target].copy()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_full, y_full, test_size=0.25, stratify=y_full, random_state=42
        )
        weights_test = None

    print(f"Features : {len(feature_cols)}")
    print(f"Train    : {y_train_full.mean():.1%} positive (n={len(y_train_full)})")
    print(f"Test     : {y_test.mean():.1%} positive (n={len(y_test)})")

    # SMOTE-Tomek on full training set
    t0 = time.time()
    print("\nApplying SMOTE-Tomek...")
    smt          = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train_full, y_train_full)
    X_res_df     = pd.DataFrame(X_res, columns=feature_cols)
    y_res_s      = pd.Series(y_res)
    print(f"  After resampling: {y_res_s.sum()} pos / {(y_res_s==0).sum()} neg "
          f"[{time.time()-t0:.1f}s]")

    # Logistic Regression (solver=saga, StandardScaler)
    t0 = time.time()
    print("\n[1/4] Logistic Regression (saga solver, StandardScaler)...")
    lr_params = {"solver": "saga", "max_iter": 10000, "random_state": 42}
    lr_thr    = derive_threshold_via_val(
        LogisticRegression, lr_params, X_train_full, y_train_full, use_scaler=True
    )

    scaler_lr    = StandardScaler()
    X_res_lr     = pd.DataFrame(scaler_lr.fit_transform(X_res_df), columns=feature_cols)
    X_test_lr    = pd.DataFrame(scaler_lr.transform(X_test), columns=feature_cols)

    lr = LogisticRegression(**lr_params)
    lr.fit(X_res_lr, y_res_s)
    lr_prob_tr = lr.predict_proba(X_res_lr)[:, 1]
    lr_pred_tr = (lr_prob_tr >= lr_thr).astype(int)
    lr_prob    = lr.predict_proba(X_test_lr)[:, 1]
    lr_pred    = (lr_prob >= lr_thr).astype(int)
    lr_metrics_tr = compute_metrics(y_res_s, lr_pred_tr, lr_prob_tr)
    lr_metrics    = compute_metrics(y_test, lr_pred, lr_prob, weights_test)
    print(f"  Threshold (val)={lr_thr:.3f}  Done [{time.time()-t0:.1f}s]")

    # Random Forest
    t0 = time.time()
    print("[2/4] Random Forest...")
    rf_params = {"n_estimators": 300, "max_depth": 8, "random_state": 42, "n_jobs": -1}
    rf_thr    = derive_threshold_via_val(RandomForestClassifier, rf_params, X_train_full, y_train_full)

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_res_df, y_res_s)
    rf_prob_tr = rf.predict_proba(X_res_df)[:, 1]
    rf_pred_tr = (rf_prob_tr >= rf_thr).astype(int)
    rf_prob    = rf.predict_proba(X_test)[:, 1]
    rf_pred    = (rf_prob >= rf_thr).astype(int)
    rf_metrics_tr = compute_metrics(y_res_s, rf_pred_tr, rf_prob_tr)
    rf_metrics    = compute_metrics(y_test, rf_pred, rf_prob, weights_test)
    print(f"  Threshold (val)={rf_thr:.3f}  Done [{time.time()-t0:.1f}s]")

    # XGBoost
    t0 = time.time()
    print(f"[3/4] XGBoost ({n_trials} trials)...")
    xgb_params      = tune_xgb(X_train_full, y_train_full, n_trials)
    xgb_full_params = {**xgb_params, "eval_metric": "auc", "base_score": 0.5, "random_state": 42}
    xgb_thr         = derive_threshold_via_val(xgb.XGBClassifier, xgb_full_params, X_train_full, y_train_full)

    xgb_model = xgb.XGBClassifier(**xgb_full_params)
    xgb_model.fit(X_res_df, y_res_s)
    xgb_prob_tr = xgb_model.predict_proba(X_res_df)[:, 1]
    xgb_pred_tr = (xgb_prob_tr >= xgb_thr).astype(int)
    xgb_prob    = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred    = (xgb_prob >= xgb_thr).astype(int)
    xgb_metrics_tr = compute_metrics(y_res_s, xgb_pred_tr, xgb_prob_tr)
    xgb_metrics    = compute_metrics(y_test, xgb_pred, xgb_prob, weights_test)
    print(f"  Threshold (val)={xgb_thr:.3f}  Done [{time.time()-t0:.1f}s]")

    # LightGBM
    t0 = time.time()
    print(f"[4/4] LightGBM ({n_trials} trials)...")
    lgb_params      = tune_lgb(X_train_full, y_train_full, n_trials)
    lgb_full_params = {**lgb_params, "random_state": 42, "verbose": -1}
    lgb_thr         = derive_threshold_via_val(lgb.LGBMClassifier, lgb_full_params, X_train_full, y_train_full)

    lgb_model = lgb.LGBMClassifier(**lgb_full_params)
    lgb_model.fit(X_res_df, y_res_s)
    lgb_prob_tr = lgb_model.predict_proba(X_res_df)[:, 1]
    lgb_pred_tr = (lgb_prob_tr >= lgb_thr).astype(int)
    lgb_prob    = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred    = (lgb_prob >= lgb_thr).astype(int)
    lgb_metrics_tr = compute_metrics(y_res_s, lgb_pred_tr, lgb_prob_tr)
    lgb_metrics    = compute_metrics(y_test, lgb_pred, lgb_prob, weights_test)
    print(f"  Threshold (val)={lgb_thr:.3f}  Done [{time.time()-t0:.1f}s]")

    # Model selection by CV AUC
    print("\n[CV Selection] Selecting best model by 5-fold CV AUC...")
    models_for_cv = {
        "Logistic Regression" : LogisticRegression(**lr_params),
        "Random Forest"       : RandomForestClassifier(**rf_params),
        "XGBoost"             : xgb.XGBClassifier(**xgb_full_params),
        "LightGBM"            : lgb.LGBMClassifier(**lgb_full_params),
    }
    best_name, cv_selection_aucs = select_best_model_by_cv(models_for_cv, X_train_full, y_train_full)

    models_map = {
        "Logistic Regression" : lr,
        "Random Forest"       : rf,
        "XGBoost"             : xgb_model,
        "LightGBM"            : lgb_model,
    }
    probs_map = {
        "Logistic Regression" : lr_prob,
        "Random Forest"       : rf_prob,
        "XGBoost"             : xgb_prob,
        "LightGBM"            : lgb_prob,
    }
    best_model = models_map[best_name]
    best_prob  = probs_map[best_name]

    results = {
        "Logistic Regression" : lr_metrics,
        "Random Forest"       : rf_metrics,
        "XGBoost"             : xgb_metrics,
        "LightGBM"            : lgb_metrics,
    }
    results_train = {
        "Logistic Regression" : lr_metrics_tr,
        "Random Forest"       : rf_metrics_tr,
        "XGBoost"             : xgb_metrics_tr,
        "LightGBM"            : lgb_metrics_tr,
    }
    results_table       = pd.DataFrame(results).T
    results_table_train = pd.DataFrame(results_train).T

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(results_table.to_string())
    print("\nTRAIN RESULTS (resampled)")
    print("="*60)
    print(results_table_train.to_string())
    print()
    print(f"Best (by CV, not test AUC): {best_name}  "
          f"Test AUC={results[best_name]['AUC-ROC']:.4f}")

    # Repeated 5x5 CV for best model
    t0 = time.time()
    print(f"\n[CV] Repeated 5x5 stratified CV for {best_name}...")
    use_scaler_cv = (best_name == "Logistic Regression")
    cv_summary = compute_cv_metrics(
        type(best_model), best_model.get_params(),
        X_train_full, y_train_full,
        use_scaler=use_scaler_cv
    )
    print(f"  AUC  mean={cv_summary['AUC-ROC_mean'].values[0]:.4f} "
          f"+/- {cv_summary['AUC-ROC_std'].values[0]:.4f}")
    print(f"  Sens mean={cv_summary['Sensitivity_mean'].values[0]:.2f} "
          f"+/- {cv_summary['Sensitivity_std'].values[0]:.2f}")
    print(f"  Spec mean={cv_summary['Specificity_mean'].values[0]:.2f} "
          f"+/- {cv_summary['Specificity_std'].values[0]:.2f}")
    print(f"  CV done [{time.time()-t0:.1f}s]")

    # Bootstrap pairwise AUC comparison
    print("\n[Stats] Bootstrap pairwise AUC comparison (n_bootstrap=1000)...")
    stat_comparison = compare_models_statistically(
        y_test.values,
        {k: np.asarray(v) for k, v in probs_map.items()}
    )
    print(stat_comparison.to_string(index=False))

    print(f"\nTotal pipeline time for {target}: {time.time()-t_pipeline:.1f}s")

    return {
        "target"            : target,
        "results_table"     : results_table,
        "results_table_tr"  : results_table_train,
        "cv_summary"        : cv_summary,
        "cv_selection_aucs" : cv_selection_aucs,
        "stat_comparison"   : stat_comparison,
        "best_name"         : best_name,
        "best_model"        : best_model,
        "best_prob"         : best_prob,
        "thresholds"        : {
            "Logistic Regression" : lr_thr,
            "Random Forest"       : rf_thr,
            "XGBoost"             : xgb_thr,
            "LightGBM"            : lgb_thr,
        },
        "xgb_params"        : xgb_full_params,
        "X_res"             : X_res_df,
        "y_res"             : y_res_s,
        "X_train_original"  : X_train_full,
        "y_train_original"  : y_train_full,
        "X_test"            : X_test,
        "y_test"            : y_test,
        "weights_test"      : weights_test,
        "all_probs"         : {
            "Logistic Regression" : lr_prob,
            "Random Forest"       : rf_prob,
            "XGBoost"             : xgb_prob,
            "LightGBM"            : lgb_prob,
        },
        "all_models"        : {
            "Logistic Regression" : lr,
            "Random Forest"       : rf,
            "XGBoost"             : xgb_model,
            "LightGBM"            : lgb_model,
        },
        "scaler_lr"         : scaler_lr,
        "use_external_test" : use_external_test,
        "validation_note"   : VALIDATION_NOTE,
    }
