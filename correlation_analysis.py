"""
correlation_analysis.py
=======================
Feature correlation and somatic mutation confounding analysis.

Generates:
  1. Full feature correlation matrix (global and oncology-only)
  2. VIF (variance inflation factor) diagnostics
  3. SHAP dependence plot for somatic mutation vs literature
  4. Quantified correlation between literature and somatic mutation

Outputs:
  - results/correlation_matrix.png
  - results/correlation_oncology.png
  - results/vif_table.csv
  - results/shap_dependence_somatic.png
  - results/correlation_summary.txt
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "genetic_association",
    "somatic_mutation",
    "literature",
    "rna_expression",
    "animal_model",
    "affected_pathway",
]

FEATURE_LABELS = {
    "genetic_association": "Genetic\nAssociation",
    "somatic_mutation": "Somatic\nMutation",
    "literature": "Literature\nMining",
    "rna_expression": "RNA\nExpression",
    "animal_model": "Animal\nModel",
    "affected_pathway": "Affected\nPathway",
}

FEATURE_LABELS_SHORT = {
    "genetic_association": "Genetic Assoc.",
    "somatic_mutation": "Somatic Mut.",
    "literature": "Literature",
    "rna_expression": "RNA Expr.",
    "animal_model": "Animal Model",
    "affected_pathway": "Affected Path.",
}

RANDOM_STATE = 42


def compute_correlations(df, features, title_suffix=""):
    """Compute and return Spearman correlation matrix for features."""
    corr = df[features].corr(method="spearman")
    return corr


def plot_correlation_matrix(corr, labels, save_path, title):
    """Plot annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(corr, dtype=bool)
    # Show full matrix
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_vif(df, features):
    """Compute variance inflation factors."""
    X = df[features].values
    # Add small constant to avoid singular matrix with sparse features
    X = X + 1e-10
    vif_data = []
    for i, feat in enumerate(features):
        vif_val = variance_inflation_factor(X, i)
        vif_data.append({"Feature": FEATURE_LABELS_SHORT.get(feat, feat), "VIF": vif_val})
    return pd.DataFrame(vif_data)


def main():
    print("=" * 70)
    print("FEATURE CORRELATION & SOMATIC MUTATION ANALYSIS")
    print("=" * 70)

    # Load data
    df = pd.read_csv("data/final_dataset.csv").fillna(0)
    ta_df = pd.read_csv("data/disease_therapeutic_areas.csv")
    df = df.merge(ta_df[["efo_id_norm", "ot_therapeutic_area"]], on="efo_id_norm", how="left")
    df["ot_therapeutic_area"] = df["ot_therapeutic_area"].fillna("Other")

    print(f"Dataset: {len(df):,} pairs\n")

    labels = [FEATURE_LABELS[f] for f in FEATURE_COLS]
    labels_short = [FEATURE_LABELS_SHORT[f] for f in FEATURE_COLS]

    # 1. Global correlation matrix
    print("1. Global feature correlations (Spearman):")
    corr_global = compute_correlations(df, FEATURE_COLS)
    print(corr_global.round(3).to_string())
    plot_correlation_matrix(
        corr_global, labels,
        "results/correlation_matrix.png",
        "Feature Correlation Matrix\n(All target–disease pairs, Spearman ρ)"
    )
    print("\n[+] Saved results/correlation_matrix.png")

    # 2. Oncology-only correlations
    onc = df[df["ot_therapeutic_area"] == "Oncology"]
    print(f"\n2. Oncology-only correlations (n={len(onc):,}):")
    corr_onc = compute_correlations(onc, FEATURE_COLS)
    print(corr_onc.round(3).to_string())
    plot_correlation_matrix(
        corr_onc, labels,
        "results/correlation_oncology.png",
        "Feature Correlation Matrix — Oncology\n(Spearman ρ)"
    )
    print("\n[+] Saved results/correlation_oncology.png")

    # Key correlation: somatic mutation vs literature
    rho_som_lit_global = corr_global.loc["somatic_mutation", "literature"]
    rho_som_lit_onc = corr_onc.loc["somatic_mutation", "literature"]
    rho_som_gen_global = corr_global.loc["somatic_mutation", "genetic_association"]
    rho_som_gen_onc = corr_onc.loc["somatic_mutation", "genetic_association"]

    print(f"\n  Key correlations:")
    print(f"    Somatic-Literature (global):   rho = {rho_som_lit_global:.3f}")
    print(f"    Somatic-Literature (oncology): rho = {rho_som_lit_onc:.3f}")
    print(f"    Somatic-Genetic (global):      rho = {rho_som_gen_global:.3f}")
    print(f"    Somatic-Genetic (oncology):    rho = {rho_som_gen_onc:.3f}")

    # 3. VIF analysis
    print(f"\n3. Variance Inflation Factors (global):")
    vif_global = compute_vif(df, FEATURE_COLS)
    print(vif_global.to_string(index=False, float_format="%.2f"))

    print(f"\n   VIF (oncology only):")
    vif_onc = compute_vif(onc, FEATURE_COLS)
    print(vif_onc.to_string(index=False, float_format="%.2f"))

    vif_combined = vif_global.rename(columns={"VIF": "VIF_Global"}).merge(
        vif_onc.rename(columns={"VIF": "VIF_Oncology"}), on="Feature"
    )
    vif_combined.to_csv("results/vif_table.csv", index=False)
    print("\n[+] Saved results/vif_table.csv")

    # 4. SHAP dependence plot for somatic mutation
    print(f"\n4. SHAP dependence analysis...")
    X = df[FEATURE_COLS].values
    y = df["label"].values
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_model.fit(X, y)

    explainer = shap.TreeExplainer(
        xgb_model, data=X,
        feature_perturbation='interventional',
        model_output='raw'
    )
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # SHAP dependence: somatic mutation colored by literature
    som_idx = FEATURE_COLS.index("somatic_mutation")
    lit_idx = FEATURE_COLS.index("literature")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Somatic mutation SHAP vs somatic mutation value, colored by literature
    sc1 = axes[0].scatter(
        X[:, som_idx], shap_values[:, som_idx],
        c=X[:, lit_idx], cmap="RdYlBu_r", alpha=0.3, s=5, rasterized=True,
    )
    axes[0].axhline(0, color="k", linestyle="--", lw=0.5, alpha=0.5)
    axes[0].set_xlabel("Somatic Mutation Score", fontsize=11)
    axes[0].set_ylabel("SHAP Value (Somatic Mutation)", fontsize=11)
    axes[0].set_title("SHAP Dependence: Somatic Mutation\n(colored by Literature Mining)", fontsize=11, fontweight="bold")
    plt.colorbar(sc1, ax=axes[0], label="Literature Mining Score", shrink=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Somatic mutation SHAP vs literature value
    sc2 = axes[1].scatter(
        X[:, lit_idx], shap_values[:, som_idx],
        c=X[:, som_idx], cmap="RdYlBu_r", alpha=0.3, s=5, rasterized=True,
    )
    axes[1].axhline(0, color="k", linestyle="--", lw=0.5, alpha=0.5)
    axes[1].set_xlabel("Literature Mining Score", fontsize=11)
    axes[1].set_ylabel("SHAP Value (Somatic Mutation)", fontsize=11)
    axes[1].set_title("Somatic Mutation SHAP vs Literature\n(colored by Somatic Mutation Score)", fontsize=11, fontweight="bold")
    plt.colorbar(sc2, ax=axes[1], label="Somatic Mutation Score", shrink=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/shap_dependence_somatic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved results/shap_dependence_somatic.png")

    # 5. Conditional analysis: somatic mutation effect with/without literature
    print(f"\n5. Conditional somatic mutation analysis:")

    # Mean SHAP for somatic mutation when literature is high vs low
    lit_median = np.median(X[X[:, lit_idx] > 0, lit_idx]) if (X[:, lit_idx] > 0).any() else 0
    high_lit_mask = X[:, lit_idx] > lit_median
    low_lit_mask = X[:, lit_idx] <= lit_median

    mean_shap_som_high_lit = shap_values[high_lit_mask, som_idx].mean()
    mean_shap_som_low_lit = shap_values[low_lit_mask, som_idx].mean()

    print(f"    Mean SHAP(somatic) when literature > median: {mean_shap_som_high_lit:.4f}")
    print(f"    Mean SHAP(somatic) when literature <= median: {mean_shap_som_low_lit:.4f}")

    # Among non-zero somatic pairs
    has_somatic = X[:, som_idx] > 0
    if has_somatic.any():
        mean_shap_som_nonzero = shap_values[has_somatic, som_idx].mean()
        mean_shap_som_nonzero_high_lit = shap_values[has_somatic & high_lit_mask, som_idx].mean()
        mean_shap_som_nonzero_low_lit = shap_values[has_somatic & low_lit_mask, som_idx].mean()
        print(f"\n    Among pairs with somatic evidence (n={has_somatic.sum():,}):")
        print(f"      Mean SHAP(somatic): {mean_shap_som_nonzero:.4f}")
        print(f"      With high literature: {mean_shap_som_nonzero_high_lit:.4f}")
        print(f"      With low literature:  {mean_shap_som_nonzero_low_lit:.4f}")

    # Save summary
    summary_lines = [
        "CORRELATION AND SOMATIC MUTATION ANALYSIS SUMMARY",
        "=" * 60,
        "",
        "GLOBAL SPEARMAN CORRELATIONS:",
        corr_global.round(3).to_string(),
        "",
        "ONCOLOGY-ONLY SPEARMAN CORRELATIONS:",
        corr_onc.round(3).to_string(),
        "",
        "KEY CORRELATIONS:",
        f"  Somatic-Literature (global):   rho = {rho_som_lit_global:.3f}",
        f"  Somatic-Literature (oncology): rho = {rho_som_lit_onc:.3f}",
        f"  Somatic-Genetic (global):      rho = {rho_som_gen_global:.3f}",
        f"  Somatic-Genetic (oncology):    rho = {rho_som_gen_onc:.3f}",
        "",
        "VIF (GLOBAL):",
        vif_global.to_string(index=False, float_format="%.2f"),
        "",
        "VIF (ONCOLOGY):",
        vif_onc.to_string(index=False, float_format="%.2f"),
        "",
        "CONDITIONAL SHAP ANALYSIS:",
        f"  Mean SHAP(somatic) when literature > median: {mean_shap_som_high_lit:.4f}",
        f"  Mean SHAP(somatic) when literature <= median: {mean_shap_som_low_lit:.4f}",
    ]

    with open("results/correlation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n[+] Saved results/correlation_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
