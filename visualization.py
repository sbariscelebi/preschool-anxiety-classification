"""
visualization.py
ROC curves for all models including the reduced model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from config import OUTPUT_DIR


def plot_roc_curves(result, reduced_result):
    """ROC curves for all models and the reduced model."""
    target  = result["target"]
    y_test  = result["y_test"].values
    weights = result["weights_test"]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Logistic Regression" : "#999999",
        "Random Forest"       : "#2196F3",
        "XGBoost"             : "#F44336",
        "LightGBM"            : "#4CAF50",
        "Reduced"             : "#FF9800",
    }
    labels = {
        "Logistic Regression" : "Logistic Regression",
        "Random Forest"       : "Random Forest",
        "XGBoost"             : "XGBoost (Full)",
        "LightGBM"            : "LightGBM (Full)",
        "Reduced"             : f"XGBoost (Top-{reduced_result['n_features']})",
    }

    probs_map = {**result["all_probs"], "Reduced": reduced_result["prob"]}

    for name, prob in probs_map.items():
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, prob, sample_weight=weights)
        auc_val     = roc_auc_score(y_test, prob, sample_weight=weights)
        ax.plot(
            fpr, tpr,
            color=colors.get(name, "black"),
            lw=2.5,
            label=f"{labels.get(name, name)} (AUC={auc_val:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.4, label="Random")
    ax.set_xlabel("1 - Specificity", fontsize=16)
    ax.set_ylabel("Sensitivity", fontsize=16)
    ax.set_title(f"ROC Curves -- {target}", fontsize=18, fontweight="bold")
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_curves_{target}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/roc_curves_{target}.svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: roc_curves_{target}.png / .svg")
