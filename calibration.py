"""
calibration.py
Calibration curves with ECE and Hosmer-Lemeshow test.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from config import OUTPUT_DIR
from metrics import compute_ece, hosmer_lemeshow_test


def plot_calibration_curves(probs_dict, y_test, target):
    """
    Reliability diagram for all models.
    Reports ECE and Hosmer-Lemeshow goodness-of-fit per model.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    color_map = {
        "Logistic Regression" : "#999999",
        "Random Forest"       : "#2196F3",
        "XGBoost"             : "#F44336",
        "LightGBM"            : "#4CAF50",
    }

    y_arr             = np.asarray(y_test)
    calibration_stats = {}

    print(f"\n  Calibration metrics -- {target}:")
    print(f"  {'Model':<25} {'Brier':>7} {'ECE':>7} {'HL_stat':>9} {'HL_p':>8} {'Calibrated?':>12}")
    print(f"  {'-'*70}")

    for name, prob in probs_dict.items():
        if prob is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_arr, prob, n_bins=10, strategy="quantile")
        brier               = brier_score_loss(y_arr, prob)
        ece                 = compute_ece(y_arr, prob, n_bins=10)
        hl                  = hosmer_lemeshow_test(y_arr, prob, n_groups=10)
        calibrated          = "Yes" if hl["HL_pvalue"] > 0.05 else "No"

        calibration_stats[name] = {
            "Brier"              : round(brier, 4),
            "ECE"                : ece,
            **hl,
            "Calibrated_pgt0.05" : calibrated,
        }

        print(f"  {name:<25} {brier:>7.4f} {ece:>7.4f} {hl['HL_stat']:>9.3f} "
              f"{hl['HL_pvalue']:>8.4f} {calibrated:>12}")

        ax.plot(
            mean_pred, frac_pos,
            marker="o", lw=2,
            color=color_map.get(name, "black"),
            label=f"{name} (Brier={brier:.3f}, ECE={ece:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=15)
    ax.set_ylabel("Fraction of Positives", fontsize=15)
    ax.set_title(f"Calibration Curves -- {target}", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/calibration_{target}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/calibration_{target}.svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: calibration_{target}.png / .svg")

    return calibration_stats
