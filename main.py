"""
main.py
Entry point for the Preschool Anxiety Classification Pipeline v3.

Dataset  : Carpenter et al. (2016) PAPA Dataset
           Training: n=917 (PAS Study)
           Testing : n=307 (PTRTS -- demographics only, see VALIDATION_NOTE)

Targets  : GAD (Generalized Anxiety Disorder)
           SAD (Separation Anxiety Disorder)

Usage    : python main.py
"""

from data_loader import load_excel, check_compatibility
from training import run_pipeline
from shap_analysis import run_shap
from reduced_model import train_reduced_model
from visualization import plot_roc_curves
from calibration import plot_calibration_curves
from export import build_predictions_df, export_results
from config import VALIDATION_NOTE


def main():
    TRAIN_PATH = "/kaggle/input/datasets/selahattinbareleb/anksiyeteverikumesi/Training Data.xlsx"
    TEST_PATH  = "/kaggle/input/datasets/selahattinbareleb/anksiyeteverikumesi/Testing Data.xlsx"

    print("="*60)
    print("PRESCHOOL ANXIETY CLASSIFICATION PIPELINE v3")
    print("XGBoost + SHAP + SMOTE-Tomek")
    print("="*60)

    print("\n[1] Loading Training Data...")
    train_df, train_items = load_excel(TRAIN_PATH)

    print("\n[2] Loading Testing Data...")
    test_df, test_items = load_excel(TEST_PATH)

    print("\n[3] Checking feature compatibility...")
    common_items, is_compatible = check_compatibility(train_items, test_items)

    if is_compatible:
        print(f"  Compatible: {len(common_items)} common features")
        feature_cols      = sorted(common_items)
        use_external_test = True
    else:
        print(f"  Incompatible -- using internal 75/25 split")
        print(f"  {VALIDATION_NOTE}")
        feature_cols      = train_items
        use_external_test = False

    # GAD
    gad_result    = run_pipeline(train_df, test_df, feature_cols, "GAD", use_external_test, n_trials=30)
    gad_top_feats = run_shap(gad_result)
    gad_reduced   = train_reduced_model(gad_result, gad_top_feats, n_features=10)
    plot_roc_curves(gad_result, gad_reduced)
    gad_cal_stats = plot_calibration_curves(gad_result["all_probs"], gad_result["y_test"], "GAD")

    # SAD
    sad_result    = run_pipeline(train_df, test_df, feature_cols, "SAD", use_external_test, n_trials=30)
    sad_top_feats = run_shap(sad_result)
    sad_reduced   = train_reduced_model(sad_result, sad_top_feats, n_features=10)
    plot_roc_curves(sad_result, sad_reduced)
    sad_cal_stats = plot_calibration_curves(sad_result["all_probs"], sad_result["y_test"], "SAD")

    # Export
    gad_preds  = build_predictions_df(gad_result, gad_reduced)
    sad_preds  = build_predictions_df(sad_result, sad_reduced)
    comparison = export_results(
        gad_result, sad_result, gad_reduced, sad_reduced,
        gad_preds, sad_preds, gad_cal_stats, sad_cal_stats
    )

    print("\nFINAL COMPARISON TABLE (Test Set)")
    print("="*60)
    print(comparison[comparison.index.get_level_values("Split") == "Test"].to_string())
    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()
