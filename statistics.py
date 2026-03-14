"""
statistics.py
Bootstrap pairwise AUC comparison between models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auc_difference(y_true, prob_a, prob_b, n_bootstrap=1000, random_state=42):
    """
    Bootstrap pairwise AUC comparison.
    Reference: DeLong et al. (1988), doi:10.2307/2531595.
    """
    rng   = np.random.RandomState(random_state)
    y_arr = np.asarray(y_true)
    pa    = np.asarray(prob_a)
    pb    = np.asarray(prob_b)
    n     = len(y_arr)
    diffs = []
    effective_n = 0

    for _ in range(n_bootstrap):
        idx      = rng.choice(n, n, replace=True)
        y_sample = y_arr[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_sample, pa[idx])
            auc_b = roc_auc_score(y_sample, pb[idx])
            diffs.append(auc_a - auc_b)
            effective_n += 1
        except Exception:
            continue

    diffs = np.array(diffs)
    if diffs.size == 0:
        observed_diff = float(roc_auc_score(y_arr, pa) - roc_auc_score(y_arr, pb))
        return {
            "AUC_diff"              : round(observed_diff, 4),
            "CI_2.5"                : np.nan,
            "CI_97.5"               : np.nan,
            "p_value"               : np.nan,
            "n_bootstrap_effective" : 0,
        }

    observed_diff   = float(roc_auc_score(y_arr, pa) - roc_auc_score(y_arr, pb))
    p_value         = float(np.mean(np.abs(diffs) >= np.abs(observed_diff)))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return {
        "AUC_diff"              : round(observed_diff, 4),
        "CI_2.5"                : round(float(ci_low), 4),
        "CI_97.5"               : round(float(ci_high), 4),
        "p_value"               : round(p_value, 4),
        "n_bootstrap_effective" : int(effective_n),
    }


def compare_models_statistically(y_test, probs_dict):
    """Pairwise bootstrap AUC comparison for all model combinations."""
    models = list(probs_dict.keys())
    rows   = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            res    = bootstrap_auc_difference(y_test, probs_dict[ma], probs_dict[mb])
            rows.append({"Model_A": ma, "Model_B": mb, **res})
    return pd.DataFrame(rows)
