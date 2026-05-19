"""
ablation_analysis.py
====================
Systematic feature ablation study for the manuscript revision.

Runs 7 feature configurations through both cross-validation and temporal split:
  1. All features
  2. No literature
  3. Literature only
  4. Genetics only
  5. Genetics + literature
  6. No genetics
  7. No somatic mutation

For each: AUROC, AUPRC, baseline AUPRC, AUPRC/baseline ratio.

Outputs:
  - results/ablation_table.csv
  - results/ablation_summary.txt
  - Prints LaTeX-ready table
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "genetic_association",
    "somatic_mutation",
    "literature",
    "rna_expression",
    "animal_model",
    "affected_pathway",
]

RANDOM_STATE = 42
N_FOLDS = 5
TEMPORAL_CUTOFF = 2015

# Define ablation configurations
ABLATION_CONFIGS = {
    "All features": FEATURE_COLS,
    "No literature": [c for c in FEATURE_COLS if c != "literature"],
    "Literature only": ["literature"],
    "Genetics only": ["genetic_association"],
    "Genetics + literature": ["genetic_association", "literature"],
    "No genetics": [c for c in FEATURE_COLS if c != "genetic_association"],
    "No somatic mutation": [c for c in FEATURE_COLS if c != "somatic_mutation"],
}


def run_cv(df, features, y, random_state=RANDOM_STATE):
    """Run cross-validation and return AUROC, AUPRC, baseline."""
    X = df[features].values
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=random_state, verbosity=0,
    )

    probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, probas)
    auprc = average_precision_score(y, probas)
    baseline = y.mean()
    return auroc, auprc, baseline


def run_temporal(df, features, random_state=RANDOM_STATE):
    """Run temporal split and return AUROC, AUPRC, baseline."""
    approved_mask = df["label"] == 1
    has_year = df["first_approval"].notna()

    train_mask = (~approved_mask) | (approved_mask & has_year & (df["first_approval"] <= TEMPORAL_CUTOFF))
    test_mask = approved_mask & has_year & (df["first_approval"] > TEMPORAL_CUTOFF)

    np.random.seed(random_state)
    stalled_no_year = df.index[(~approved_mask) & (~has_year)]
    n_test_positives = test_mask.sum()
    n_stalled_for_test = min(n_test_positives * 18, len(stalled_no_year))
    stalled_test_idx = np.random.choice(stalled_no_year, size=n_stalled_for_test, replace=False)

    test_mask_full = test_mask.copy()
    test_mask_full.iloc[stalled_test_idx] = True
    train_mask_full = train_mask.copy()
    train_mask_full.iloc[stalled_test_idx] = False

    X_train = df.loc[train_mask_full, features].values
    y_train = df.loc[train_mask_full, "label"].values
    X_test = df.loc[test_mask_full, features].values
    y_test = df.loc[test_mask_full, "label"].values

    if (y_test == 1).sum() < 10 or (y_test == 0).sum() < 10:
        return np.nan, np.nan, np.nan

    pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw,
        eval_metric="logloss", random_state=random_state, verbosity=0,
    )
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, probas)
    auprc = average_precision_score(y_test, probas)
    baseline = y_test.mean()
    return auroc, auprc, baseline


def main():
    print("=" * 70)
    print("FEATURE ABLATION ANALYSIS")
    print("=" * 70)

    # Load data
    df = pd.read_csv("data/final_dataset.csv").fillna(0)

    # Merge temporal data
    ta_df = pd.read_csv("data/disease_therapeutic_areas.csv")
    df = df.merge(ta_df[["efo_id_norm", "ot_therapeutic_area"]], on="efo_id_norm", how="left")
    df["ot_therapeutic_area"] = df["ot_therapeutic_area"].fillna("Other")

    approval_years_df = pd.read_csv("data/drug_approval_years.csv")
    drug_data = pd.read_csv("data/ot_drug_data.csv")
    drug_with_years = drug_data.merge(approval_years_df, on="drug_id", how="left")
    td_approval_year = (
        drug_with_years.groupby(["ensembl_id", "efo_id_norm"])["first_approval"]
        .min().reset_index()
    )
    df = df.merge(td_approval_year, on=["ensembl_id", "efo_id_norm"], how="left")

    y = df["label"].values
    print(f"Dataset: {len(df):,} pairs, {y.sum():,} approved\n")

    results = []

    for config_name, features in ABLATION_CONFIGS.items():
        print(f"  Running: {config_name} ({len(features)} features: {', '.join(features)})")

        # Cross-validation
        auroc_cv, auprc_cv, baseline_cv = run_cv(df, features, y)
        ratio_cv = auprc_cv / baseline_cv if baseline_cv > 0 else np.nan

        # Temporal split
        auroc_t, auprc_t, baseline_t = run_temporal(df, features)
        ratio_t = auprc_t / baseline_t if baseline_t and baseline_t > 0 else np.nan

        results.append({
            "Configuration": config_name,
            "Features": ", ".join(features),
            "N_features": len(features),
            "CV_AUROC": auroc_cv,
            "CV_AUPRC": auprc_cv,
            "CV_Baseline": baseline_cv,
            "CV_Ratio": ratio_cv,
            "Temporal_AUROC": auroc_t,
            "Temporal_AUPRC": auprc_t,
            "Temporal_Baseline": baseline_t,
            "Temporal_Ratio": ratio_t,
        })

        print(f"    CV:       AUROC={auroc_cv:.3f}  AUPRC={auprc_cv:.3f}  "
              f"baseline={baseline_cv:.3f}  ratio={ratio_cv:.2f}x")
        if not np.isnan(auroc_t):
            print(f"    Temporal: AUROC={auroc_t:.3f}  AUPRC={auprc_t:.3f}  "
                  f"baseline={baseline_t:.3f}  ratio={ratio_t:.2f}x")
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/ablation_table.csv", index=False)

    # Print LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX TABLE")
    print("=" * 70)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Feature ablation analysis. Performance of XGBoost under systematic feature removal, evaluated by cross-validation and temporal split. Baseline AUPRC reflects the positive class prevalence in each evaluation.}")
    print(r"\label{tab:ablation}")
    print(r"\small")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{4}{c}{Cross-validation} & \multicolumn{4}{c}{Temporal split} \\")
    print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    print(r"Configuration & AUROC & AUPRC & Baseline & Ratio & AUROC & AUPRC & Baseline & Ratio \\")
    print(r"\midrule")

    for _, row in results_df.iterrows():
        name = row["Configuration"]
        cv_auroc = f"{row['CV_AUROC']:.3f}"
        cv_auprc = f"{row['CV_AUPRC']:.3f}"
        cv_base = f"{row['CV_Baseline']:.3f}"
        cv_ratio = f"{row['CV_Ratio']:.2f}$\\times$"
        if np.isnan(row["Temporal_AUROC"]):
            t_auroc = t_auprc = t_base = t_ratio = "---"
        else:
            t_auroc = f"{row['Temporal_AUROC']:.3f}"
            t_auprc = f"{row['Temporal_AUPRC']:.3f}"
            t_base = f"{row['Temporal_Baseline']:.3f}"
            t_ratio = f"{row['Temporal_Ratio']:.2f}$\\times$"
        print(f"{name} & {cv_auroc} & {cv_auprc} & {cv_base} & {cv_ratio} & "
              f"{t_auroc} & {t_auprc} & {t_base} & {t_ratio} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save summary
    with open("results/ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write("FEATURE ABLATION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(results_df.to_string(index=False, float_format="%.3f"))
        f.write("\n\nKey findings:\n")

        # Compare genetics-only vs literature-only
        gen_only = results_df[results_df["Configuration"] == "Genetics only"].iloc[0]
        lit_only = results_df[results_df["Configuration"] == "Literature only"].iloc[0]
        no_lit = results_df[results_df["Configuration"] == "No literature"].iloc[0]
        all_feat = results_df[results_df["Configuration"] == "All features"].iloc[0]

        f.write(f"\n1. Genetics only: CV AUROC={gen_only['CV_AUROC']:.3f}, "
                f"AUPRC={gen_only['CV_AUPRC']:.3f} ({gen_only['CV_Ratio']:.2f}x baseline)\n")
        f.write(f"2. Literature only: CV AUROC={lit_only['CV_AUROC']:.3f}, "
                f"AUPRC={lit_only['CV_AUPRC']:.3f} ({lit_only['CV_Ratio']:.2f}x baseline)\n")
        f.write(f"3. No literature: CV AUROC={no_lit['CV_AUROC']:.3f}, "
                f"AUPRC={no_lit['CV_AUPRC']:.3f} ({no_lit['CV_Ratio']:.2f}x baseline)\n")
        f.write(f"4. All features: CV AUROC={all_feat['CV_AUROC']:.3f}, "
                f"AUPRC={all_feat['CV_AUPRC']:.3f} ({all_feat['CV_Ratio']:.2f}x baseline)\n")

        if not np.isnan(gen_only["Temporal_AUROC"]):
            f.write(f"\nTemporal genetics-only: AUROC={gen_only['Temporal_AUROC']:.3f}, "
                    f"AUPRC={gen_only['Temporal_AUPRC']:.3f}\n")
        if not np.isnan(no_lit["Temporal_AUROC"]):
            f.write(f"Temporal no-literature: AUROC={no_lit['Temporal_AUROC']:.3f}, "
                    f"AUPRC={no_lit['Temporal_AUPRC']:.3f}\n")

    print("\n[+] Saved results/ablation_table.csv")
    print("[+] Saved results/ablation_summary.txt")


if __name__ == "__main__":
    main()
