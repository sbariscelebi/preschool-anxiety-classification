"""
shap_analysis.py
SHAP-based feature importance using TreeExplainer.
Computed on the original (non-resampled) training distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from config import OUTPUT_DIR


def run_shap(result):
    """
    SHAP analysis on the original training distribution.
    Uses TreeExplainer for all tree-based models.

    NOTE: SHAP values are computed on the full training set.
    The reduced model is then trained on the same set.
    This constitutes post-hoc feature selection, which may
    introduce optimistic bias in reduced model performance.
    This limitation should be reported in the manuscript.
    Reference: Lundberg & Lee (2017), doi:10.48550/arXiv.1705.07874.
    """
    target         = result["target"]
    model          = result["best_model"]
    X_train_origin = result["X_train_original"]
    X_test         = result["X_test"]
    best_name      = result["best_name"]

    print(f"\n[SHAP] {target} -- {best_name} (TreeExplainer, original distribution)...")
    t0 = time.time() if False else __import__("time").time()

    background  = shap.sample(X_train_origin, min(100, len(X_train_origin)), random_state=42)
    explainer   = shap.TreeExplainer(model, data=background)
    shap_values = explainer(X_train_origin)

    if len(shap_values.values.shape) == 3:
        shap_values.values = shap_values.values[:, :, 1]

    import time as _time
    print(f"  SHAP computed in {_time.time() - t0:.1f}s")

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=20, color="black")
    ax.tick_params(axis="x", labelsize=20, colors="black", width=1.5, length=6)
    ax.tick_params(axis="y", labelsize=20, colors="black", width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    ax.set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")
    plt.title(f"SHAP Feature Importance -- {target}", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm_{target}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm_{target}.svg", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: shap_beeswarm_{target}.png / .svg")

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    ax = plt.gca()
    ax.tick_params(axis="x", labelsize=20, colors="black", width=1.5, length=6)
    ax.tick_params(axis="y", labelsize=20, colors="black", width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    ax.set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")
    plt.title(f"Mean SHAP Values -- {target}", fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("mean(|SHAP value|)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_bar_{target}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{OUTPUT_DIR}/shap_bar_{target}.svg", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: shap_bar_{target}.png / .svg")

    # Waterfall — highest risk case in test set
    probs         = result["best_prob"]
    high_risk_idx = int(np.argmax(probs))
    shap_single   = explainer(X_test.iloc[[high_risk_idx]])
    if shap_single.values.ndim == 3:
        shap_single.values = shap_single.values[:, :, 1]
    plt.figure(figsize=(11, 7))
    shap.plots.waterfall(shap_single[0], show=False)
    ax = plt.gca()
    ax.tick_params(axis="x", labelsize=18, colors="black", width=1.5)
    ax.tick_params(axis="y", labelsize=14, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    plt.title(
        f"Individual Explanation -- Highest Risk ({target})\n"
        f"Risk Score = {probs[high_risk_idx]:.3f}",
        fontsize=18, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_{target}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_{target}.svg", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: shap_waterfall_{target}.png / .svg")

    # Feature ranking
    mean_abs_shap      = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_train_origin.columns)
    top_features       = feature_importance.sort_values(ascending=False).head(10)

    print(f"\nTop 10 PAPA items by SHAP (original distribution):")
    for i, (item, score) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {item[:55]:<55s} | {score:.4f}")

    return top_features
