"""
metrics.py
Classification metrics, calibration metrics, and threshold selection.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, brier_score_loss
)
from scipy import stats


def find_optimal_threshold(y_true, y_prob):
    """
    Youden-index optimal classification threshold.
    J = Sensitivity + Specificity - 1 = TPR - FPR.
    Maximizes sensitivity and specificity simultaneously.
    In a screening context, false negatives are more costly than
    false positives; Youden index is therefore a conservative choice
    (Fluss et al., 2005, doi:10.1177/0272989X05276375).
    Threshold is derived from validation-fold probabilities only.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_prob))
    youden_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[youden_idx])


def compute_metrics(y_true, y_pred, y_prob, weights=None):
    """
    Classification metrics with optional sample weights.
    NOTE: Sampling weights are available only for the external test set.
    CV metrics are unweighted estimates on the PAS training data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if weights is not None:
        weights = np.asarray(weights)

    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    if cm.size == 1:
        tn = float(cm[0, 0])
        fp = fn = tp = 0.0
    else:
        tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    kw = {"sample_weight": weights} if weights is not None else {}

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return {
        "Accuracy"    : round(float(np.average(y_true == y_pred, weights=weights)) * 100, 2),
        "AUC-ROC"     : round(float(roc_auc_score(y_true, y_prob, **kw)), 4),
        "F1"          : round(float(f1_score(y_true, y_pred, **kw)), 4),
        "MCC"         : round(float(mcc), 4),
        "Sensitivity" : round(float(sens * 100), 2),
        "Specificity" : round(float(spec * 100), 2),
        "PPV"         : round(float(ppv * 100), 2),
        "NPV"         : round(float(npv * 100), 2),
        "Brier"       : round(float(brier_score_loss(y_true, y_prob, sample_weight=weights)), 4),
    }


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-width bins.
    Lower is better; ECE < 0.05 is generally considered well-calibrated
    (Guo et al., 2017, doi:10.48550/arXiv.1706.04599).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n      = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """
    Hosmer-Lemeshow goodness-of-fit test.
    Groups observations into deciles of predicted probability, then
    computes chi-squared between observed and expected event counts.
    p > 0.05 indicates adequate calibration.
    Reference: Hosmer & Lemeshow (2000), ISBN 978-0471722144.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n      = len(y_true)

    order    = np.argsort(y_prob)
    y_true_s = y_true[order]
    y_prob_s = y_prob[order]

    groups  = np.array_split(np.arange(n), n_groups)
    hl_stat = 0.0
    valid_g = 0

    for grp in groups:
        if len(grp) == 0:
            continue
        obs_pos = y_true_s[grp].sum()
        exp_pos = y_prob_s[grp].sum()
        obs_neg = len(grp) - obs_pos
        exp_neg = len(grp) - exp_pos

        if exp_pos > 0:
            hl_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            hl_stat += (obs_neg - exp_neg) ** 2 / exp_neg
        valid_g += 1

    df      = valid_g - 2
    p_value = 1.0 - stats.chi2.cdf(hl_stat, df) if df > 0 else np.nan

    return {
        "HL_stat"  : round(float(hl_stat), 4),
        "HL_df"    : int(df),
        "HL_pvalue": round(float(p_value), 4),
    }
