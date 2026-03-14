"""
tuning.py
Optuna hyperparameter tuning for XGBoost and LightGBM.
SMOTE-Tomek is applied inside each CV fold via ImbPipeline.
"""

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb


def tune_xgb(X, y, n_trials=30):
    """
    Optuna tuning for XGBoost.
    Resampling is applied inside each CV fold via ImbPipeline.
    """
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"        : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.5, 2.0),
        }
        clf = xgb.XGBClassifier(
            **params,
            eval_metric="auc",
            base_score=0.5,
            random_state=42,
        )
        pipeline = ImbPipeline([
            ("resampler", SMOTETomek(random_state=42)),
            ("clf", clf),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgb(X, y, n_trials=30):
    """
    Optuna tuning for LightGBM.
    Resampling is applied inside each CV fold via ImbPipeline.
    """
    def objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"         : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", 20, 100),
            "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples" : trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 0.5, 2.0),
        }
        clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        pipeline = ImbPipeline([
            ("resampler", SMOTETomek(random_state=42)),
            ("clf", clf),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
