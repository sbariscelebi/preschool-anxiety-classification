"""
export.py
Builds prediction DataFrames and exports all results to Excel.
"""

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, VALIDATION_NOTE


def build_predictions_df(result, reduced_result):
    """Train and test predictions in a single flat DataFrame."""
    all_probs  = result["all_probs"]
    thresholds = result["thresholds"]
    all_models = result["all_models"]
    red_thr    = reduced_result["threshold"]

    y_test = result["y_test"].reset_index(drop=True)
    X_test = result["X_test"].reset_index(drop=True)

    test_rows = pd.DataFrame({"Split": "Test", "True_Label": y_test})

    for model_name, prob in all_probs.items():
        thr   = thresholds[model_name]
        short = model_name.replace(" ", "_")
        test_rows[f"{short}_Prob"]      = np.round(np.asarray(prob), 4)
        test_rows[f"{short}_Predicted"] = (np.asarray(prob) >= thr).astype(int)

    red_prob = np.asarray(reduced_result["prob"])
    test_rows["XGBoost_Reduced_Prob"]      = np.round(red_prob, 4)
    test_rows["XGBoost_Reduced_Predicted"] = (red_prob >= red_thr).astype(int)

    y_res = result["y_res"].reset_index(drop=True)
    X_res = result["X_res"].reset_index(drop=True)

    train_rows = pd.DataFrame({"Split": "Train (resampled)", "True_Label": y_res})

    for model_name, model_obj in all_models.items():
        thr   = thresholds[model_name]
        short = model_name.replace(" ", "_")
        if model_name == "Logistic Regression" and "scaler_lr" in result:
            X_input = pd.DataFrame(
                result["scaler_lr"].transform(X_res),
                columns=X_res.columns
            )
        else:
            X_input = X_res
        prob_tr = model_obj.predict_proba(X_input)[:, 1]
        train_rows[f"{short}_Prob"]      = np.round(prob_tr, 4)
        train_rows[f"{short}_Predicted"] = (prob_tr >= thr).astype(int)

    X_res_red   = X_res[reduced_result["selected_features"]]
    prob_red_tr = reduced_result["model"].predict_proba(X_res_red)[:, 1]
    train_rows["XGBoost_Reduced_Prob"]      = np.round(prob_red_tr, 4)
    train_rows["XGBoost_Reduced_Predicted"] = (prob_red_tr >= red_thr).astype(int)

    return pd.concat([train_rows, test_rows], ignore_index=True)


def export_results(gad_result, sad_result, gad_reduced, sad_reduced,
                   gad_preds, sad_preds, gad_cal_stats, sad_cal_stats):
    """
    Exports all results to Excel with the following sheets:
      Sheet 1 -- All_Results
      Sheet 2 -- CV_and_Thresholds
      Sheet 3 -- Statistical_Comparison
      Sheet 4 -- Calibration_ECE_HL
      Sheet 5 -- Validation_Note
      Sheet 6 -- GAD_Predictions
      Sheet 7 -- SAD_Predictions
    """
    output_path = f"{OUTPUT_DIR}/results_all_models.xlsx"

    # Sheet 1
    comparison_rows = []
    for label, split, metrics in [
        ("GAD | Logistic Regression", "Train (resampled)", gad_result["results_table_tr"].loc["Logistic Regression"].to_dict()),
        ("GAD | Logistic Regression", "Test",              gad_result["results_table"].loc["Logistic Regression"].to_dict()),
        ("GAD | Random Forest",       "Train (resampled)", gad_result["results_table_tr"].loc["Random Forest"].to_dict()),
        ("GAD | Random Forest",       "Test",              gad_result["results_table"].loc["Random Forest"].to_dict()),
        ("GAD | XGBoost",             "Train (resampled)", gad_result["results_table_tr"].loc["XGBoost"].to_dict()),
        ("GAD | XGBoost",             "Test",              gad_result["results_table"].loc["XGBoost"].to_dict()),
        ("GAD | LightGBM",            "Train (resampled)", gad_result["results_table_tr"].loc["LightGBM"].to_dict()),
        ("GAD | LightGBM",            "Test",              gad_result["results_table"].loc["LightGBM"].to_dict()),
        ("GAD | XGBoost Reduced",     "Test",              gad_reduced["metrics"]),
        ("SAD | Logistic Regression", "Train (resampled)", sad_result["results_table_tr"].loc["Logistic Regression"].to_dict()),
        ("SAD | Logistic Regression", "Test",              sad_result["results_table"].loc["Logistic Regression"].to_dict()),
        ("SAD | Random Forest",       "Train (resampled)", sad_result["results_table_tr"].loc["Random Forest"].to_dict()),
        ("SAD | Random Forest",       "Test",              sad_result["results_table"].loc["Random Forest"].to_dict()),
        ("SAD | XGBoost",             "Train (resampled)", sad_result["results_table_tr"].loc["XGBoost"].to_dict()),
        ("SAD | XGBoost",             "Test",              sad_result["results_table"].loc["XGBoost"].to_dict()),
        ("SAD | LightGBM",            "Train (resampled)", sad_result["results_table_tr"].loc["LightGBM"].to_dict()),
        ("SAD | LightGBM",            "Test",              sad_result["results_table"].loc["LightGBM"].to_dict()),
        ("SAD | XGBoost Reduced",     "Test",              sad_reduced["metrics"]),
    ]:
        row = {"Target_Model": label, "Split": split}
        row.update(metrics)
        comparison_rows.append(row)

    all_results_df = pd.DataFrame(comparison_rows).set_index(["Target_Model", "Split"])

    # Sheet 2
    cv_rows = []
    for target, result in [("GAD", gad_result), ("SAD", sad_result)]:
        best = result["best_name"]
        for model_name, cv_auc in result["cv_selection_aucs"].items():
            cv_rows.append({
                "Target"           : target,
                "Model"            : model_name,
                "CV_Selection_AUC" : round(cv_auc, 4),
                "Selected_Best"    : "(BEST)" if model_name == best else "",
                "Youden_Threshold" : round(result["thresholds"].get(model_name, np.nan), 4),
                "Threshold_Note"   : "Youden index; maximizes Sens+Spec (Fluss et al. 2005)",
            })
    cv_df = pd.DataFrame(cv_rows)

    repeated_cv_rows = []
    for target, result in [("GAD", gad_result), ("SAD", sad_result)]:
        cv_s = result["cv_summary"]
        row  = {"Target": target, "Model": f"{result['best_name']} (5x5 repeated CV)"}
        for col in cv_s.columns:
            row[col] = cv_s[col].values[0]
        repeated_cv_rows.append(row)
    repeated_cv_df = pd.DataFrame(repeated_cv_rows).set_index(["Target", "Model"])

    feat_rows = []
    for rank in range(10):
        feat_rows.append({
            "Rank"            : rank + 1,
            "GAD_Top_Feature" : gad_reduced["selected_features"][rank] if rank < len(gad_reduced["selected_features"]) else "",
            "SAD_Top_Feature" : sad_reduced["selected_features"][rank] if rank < len(sad_reduced["selected_features"]) else "",
        })
    feat_df = pd.DataFrame(feat_rows).set_index("Rank")

    # Sheet 3
    gad_stat = gad_result["stat_comparison"].copy()
    gad_stat.insert(0, "Target", "GAD")
    sad_stat = sad_result["stat_comparison"].copy()
    sad_stat.insert(0, "Target", "SAD")
    stat_df = pd.concat([gad_stat, sad_stat], ignore_index=True)

    # Sheet 4
    cal_rows = []
    for target_name, cal_stats in [("GAD", gad_cal_stats), ("SAD", sad_cal_stats)]:
        for model_name, stats_dict in cal_stats.items():
            row = {"Target": target_name, "Model": model_name}
            row.update(stats_dict)
            cal_rows.append(row)
    cal_df = pd.DataFrame(cal_rows)

    # Sheet 5
    note_df = pd.DataFrame({
        "Item"  : [
            "Validation approach",
            "Reason",
            "Training sample",
            "Test sample (available)",
            "Reference",
            "Post-hoc SHAP note",
        ],
        "Detail": [
            "Internal stratified 75/25 split on PAS training sample",
            "PTRTS test dataset contains only demographic variables, not raw PAPA items",
            "PAS Study, n=917, Duke Pediatric Primary Care",
            "PTRTS Study, n=307 -- demographics only",
            "Carpenter et al. (2016), doi:10.1371/journal.pone.0165524",
            "SHAP computed on full training set; reduced model trained on same set (post-hoc). Report as limitation.",
        ]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_Results")
        cv_df.to_excel(writer, sheet_name="CV_and_Thresholds", index=False)
        repeated_cv_df.to_excel(writer, sheet_name="CV_and_Thresholds", startrow=len(cv_df) + 3)
        feat_df.to_excel(writer, sheet_name="CV_and_Thresholds", startrow=len(cv_df) + len(repeated_cv_df) + 6)
        stat_df.to_excel(writer, sheet_name="Statistical_Comparison", index=False)
        cal_df.to_excel(writer, sheet_name="Calibration_ECE_HL", index=False)
        note_df.to_excel(writer, sheet_name="Validation_Note", index=False)
        gad_preds.to_excel(writer, sheet_name="GAD_Predictions", index=False)
        sad_preds.to_excel(writer, sheet_name="SAD_Predictions", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved: {output_path}")
    print(f"  Sheet 1 -- All_Results")
    print(f"  Sheet 2 -- CV_and_Thresholds")
    print(f"  Sheet 3 -- Statistical_Comparison")
    print(f"  Sheet 4 -- Calibration_ECE_HL")
    print(f"  Sheet 5 -- Validation_Note")
    print(f"  Sheet 6 -- GAD_Predictions")
    print(f"  Sheet 7 -- SAD_Predictions")
    print(f"{'='*60}\n")

    return all_results_df
