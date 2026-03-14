# Preschool Anxiety Classification Pipeline

Machine learning pipeline for binary classification of **Generalized Anxiety Disorder (GAD)** and **Separation Anxiety Disorder (SAD)** in preschool children using PAPA (Preschool Age Psychiatric Assessment) instrument items.

## Dataset

- **Training**: PAS Study, n = 917 (Carpenter et al., 2016, doi:10.1371/journal.pone.0165524)
- **Testing**: PTRTS Study, n = 307 (demographics only — see Validation Note below)

> **Validation Note**: The publicly available PTRTS dataset does not include raw PAPA items. Performance is therefore evaluated via internal stratified 75/25 split of the PAS training sample. External validation on PTRTS was not feasible with the available public data.

## Models

| Model | Tuning |
|-------|--------|
| Logistic Regression | Fixed (saga solver) |
| Random Forest | Fixed (300 trees) |
| XGBoost | Optuna (30 trials) |
| LightGBM | Optuna (30 trials) |

## Key Methods

- **Class imbalance**: SMOTE-Tomek applied inside each CV fold (no leakage)
- **Threshold**: Youden index from held-out validation split
- **Interpretability**: SHAP TreeExplainer (beeswarm, bar, waterfall plots)
- **Calibration**: Expected Calibration Error (ECE) + Hosmer-Lemeshow test
- **Model selection**: 5-fold stratified CV AUC
- **Stability**: Repeated 5x5 stratified CV for best model
- **Statistical comparison**: Bootstrap pairwise AUC (DeLong, 1988)

## Project Structure

```
anxiety_pipeline/
├── main.py            # Entry point
├── config.py          # Constants and matplotlib settings
├── data_loader.py     # Excel loading and preprocessing
├── metrics.py         # Classification and calibration metrics
├── statistics.py      # Bootstrap AUC comparison
├── tuning.py          # Optuna hyperparameter tuning
├── training.py        # Model training and CV pipeline
├── calibration.py     # Calibration curves and statistics
├── shap_analysis.py   # SHAP feature importance plots
├── reduced_model.py   # Top-10 feature reduced model
├── visualization.py   # ROC curves
├── export.py          # Excel export (7 sheets)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Update the data paths in `main.py`, then run:

```bash
python main.py
```

Outputs are saved to the `outputs/` directory.

## Citation

If you use this code, please cite:

> Gradient Boosting and SHAP-Based Classification of Preschool Anxiety Disorders from Raw PAPA Interview Data 

## License

MIT
