"""
model_trial_success_v2.py
=========================
Extended analysis adding:
  1. Per-AZ-area genetic evidence enrichment (does OR=3.25 hold in oncology?)
  2. Missed opportunity targets (Phase 1/2 with high genetic evidence)
  3. Somatic mutation paradox analysis
  4. Full summary saved as UTF-8

Run:
    python model_trial_success_v2.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report,
)
from xgboost import XGBClassifier
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "genetic_association",
    "somatic_mutation",
    "literature",
    "rna_expression",
    "animal_model",
    "affected_pathway",
]

FEATURE_LABELS = {
    "genetic_association": "Genetic Association",
    "somatic_mutation":    "Somatic Mutation",
    "literature":          "Literature Mining",
    "rna_expression":      "RNA Expression",
    "animal_model":        "Animal Model",
    "affected_pathway":    "Affected Pathway",
}

AZ_AREAS = {
    "Oncology":     ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma",
                     "sarcoma", "glioma", "myeloma", "neoplasm", "tumor", "tumour"],
    "CVRM":         ["cardiovascular", "cardiac", "heart failure", "coronary",
                     "hypertension", "diabetes", "metabolic", "renal", "kidney",
                     "stroke", "atrial fibrillation"],
    "Respiratory":  ["asthma", "copd", "pulmonary", "respiratory", "lung disease",
                     "bronchitis", "emphysema"],
    "Rare Disease": ["rare", "orphan", "eosinophilic", "amyloid", "nephropathy",
                     "hereditary", "gaucher", "fabry", "wilson"],
    "Immunology":   ["rheumatoid", "lupus", "inflammatory", "autoimmune",
                     "psoriasis", "crohn", "colitis", "scleroderma"],
}

RANDOM_STATE = 42
N_FOLDS      = 5

# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("CLINICAL TRIAL SUCCESS — EXTENDED ANALYSIS v3")
print("=" * 60)

df = pd.read_csv("data/final_dataset.csv").fillna(0)
print(f"\nDataset: {len(df):,} pairs | "
      f"{df['label'].sum():,} approved | "
      f"{(df['label']==0).sum():,} stalled")

X = df[FEATURE_COLS].values
y = df["label"].values

pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
print(f"Class ratio: 1:{pos_weight:.0f}  scale_pos_weight={pos_weight:.1f}")

# ── Load enrichment data ─────────────────────────────────────────────────────

# Ontology-based therapeutic areas (replaces keyword matching)
ta_df = pd.read_csv("data/disease_therapeutic_areas.csv")
df = df.merge(ta_df[["efo_id_norm", "ot_therapeutic_area"]], on="efo_id_norm", how="left")
df["ot_therapeutic_area"] = df["ot_therapeutic_area"].fillna("Other")

# Approval years for temporal validation
approval_years_df = pd.read_csv("data/drug_approval_years.csv")
drug_data = pd.read_csv("data/ot_drug_data.csv")
# Get earliest approval year per target-disease pair
drug_with_years = drug_data.merge(approval_years_df, on="drug_id", how="left")
td_approval_year = (
    drug_with_years.groupby(["ensembl_id", "efo_id_norm"])["first_approval"]
    .min()
    .reset_index()
)
df = df.merge(td_approval_year, on=["ensembl_id", "efo_id_norm"], how="left")

# Refresh X, y after merge
X = df[FEATURE_COLS].values
y = df["label"].values

# ── Models ────────────────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ]),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    ),
}

print(f"\n{'='*60}")
print("MODEL PERFORMANCE (5-fold stratified CV)")
print(f"{'='*60}")

results    = {}
oof_probas = {}

for name, model in models.items():
    print(f"\n-- {name} --")
    probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    auroc = roc_auc_score(y, probas)
    auprc = average_precision_score(y, probas)

    print(f"  AUROC: {auroc:.3f}")
    print(f"  AUPRC: {auprc:.3f}  <- primary metric (imbalanced data)")
    print(f"\n  Classification report (threshold=0.5):")
    preds = (probas >= 0.5).astype(int)
    print(classification_report(y, preds,
                                target_names=["Stalled (Ph1/2)", "Approved (Ph4)"],
                                zero_division=0))

    results[name]    = {"auroc": auroc, "auprc": auprc, "probas": probas}
    oof_probas[name] = probas

# ── ROC + PRC curves ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Logistic Regression": "#4C72B0", "XGBoost": "#DD8452"}

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y, res["probas"])
    axes[0].plot(fpr, tpr,
                 label=f"{name} (AUROC={res['auroc']:.3f})",
                 color=colors[name], lw=2)

axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[0].set_xlabel("False Positive Rate", fontsize=12)
axes[0].set_ylabel("True Positive Rate", fontsize=12)
axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.02])

for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y, res["probas"])
    axes[1].plot(rec, prec,
                 label=f"{name} (AUPRC={res['auprc']:.3f})",
                 color=colors[name], lw=2)

baseline = y.mean()
axes[1].axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                label=f"Baseline ({baseline:.3f})")
axes[1].set_xlabel("Recall", fontsize=12)
axes[1].set_ylabel("Precision", fontsize=12)
axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.suptitle("Predicting Drug Approval from Open Targets Evidence",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/roc_prc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[+] Saved results/roc_prc_curves.png")

# ── Sensitivity: ablation without literature (temporal leakage check) ─────────

print(f"\n{'='*60}")
print("SENSITIVITY ANALYSIS: ABLATION WITHOUT LITERATURE MINING")
print("(Addresses temporal leakage concern — literature scores may")
print(" reflect post-approval publication activity)")
print(f"{'='*60}")

FEATURE_COLS_NO_LIT = [c for c in FEATURE_COLS if c != "literature"]
X_no_lit = df[FEATURE_COLS_NO_LIT].values

models_no_lit = {
    "LR (no literature)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        )),
    ]),
    "XGB (no literature)": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    ),
}

for name, model in models_no_lit.items():
    probas = cross_val_predict(model, X_no_lit, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, probas)
    auprc = average_precision_score(y, probas)
    print(f"\n  {name}:  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")

print("\n  (Compare with full-feature model to quantify literature leakage impact)")

# ── Temporal validation split ─────────────────────────────────────────────────

print(f"\n{'='*60}")
print("TEMPORAL VALIDATION")
print("(Train on pre-2015 approvals, test on post-2015)")
print("(Addresses retrospective-only design concern)")
print(f"{'='*60}")

TEMPORAL_CUTOFF = 2015

# For approved pairs: use first_approval year
# For stalled pairs: include in training (they have no approval date)
has_year = df["first_approval"].notna()
approved_mask = df["label"] == 1

# Training set: approved before cutoff + all stalled pairs without year info
train_mask = (~approved_mask) | (approved_mask & has_year & (df["first_approval"] <= TEMPORAL_CUTOFF))
# Test set: approved after cutoff
test_mask = approved_mask & has_year & (df["first_approval"] > TEMPORAL_CUTOFF)

# Add stalled pairs to test set to avoid trivial 100% positive test
np.random.seed(RANDOM_STATE)
stalled_no_year = df.index[(~approved_mask) & (~has_year)]
n_test_positives = test_mask.sum()
n_stalled_for_test = min(n_test_positives * 18, len(stalled_no_year))
stalled_test_idx = np.random.choice(stalled_no_year, size=n_stalled_for_test, replace=False)
test_mask_full = test_mask.copy()
test_mask_full.iloc[stalled_test_idx] = True
train_mask_full = train_mask.copy()
train_mask_full.iloc[stalled_test_idx] = False

X_train_t = df.loc[train_mask_full, FEATURE_COLS].values
y_train_t = df.loc[train_mask_full, "label"].values
X_test_t = df.loc[test_mask_full, FEATURE_COLS].values
y_test_t = df.loc[test_mask_full, "label"].values

n_train_pos = y_train_t.sum()
n_test_pos = y_test_t.sum()
print(f"\n  Cutoff year: {TEMPORAL_CUTOFF}")
print(f"  Training set: {len(y_train_t):,} pairs ({n_train_pos:,} approved, pre-{TEMPORAL_CUTOFF})")
print(f"  Test set:     {len(y_test_t):,} pairs ({n_test_pos:,} approved, post-{TEMPORAL_CUTOFF})")

if n_test_pos >= 10 and (y_test_t == 0).sum() >= 10:
    pw_t = (y_train_t == 0).sum() / max((y_train_t == 1).sum(), 1)

    xgb_temporal = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw_t,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_temporal.fit(X_train_t, y_train_t)
    probas_t = xgb_temporal.predict_proba(X_test_t)[:, 1]

    auroc_t = roc_auc_score(y_test_t, probas_t)
    auprc_t = average_precision_score(y_test_t, probas_t)
    baseline_t = y_test_t.mean()

    print(f"\n  Temporal XGBoost:  AUROC={auroc_t:.3f}  AUPRC={auprc_t:.3f}")
    print(f"  Temporal baseline: AUPRC={baseline_t:.3f}")
    print(f"  Lift over baseline: {auprc_t/baseline_t:.2f}x")

    # Also test without literature (temporal + no leakage)
    FEATURE_COLS_NO_LIT_T = [c for c in FEATURE_COLS if c != "literature"]
    X_train_t_nl = df.loc[train_mask_full, FEATURE_COLS_NO_LIT_T].values
    X_test_t_nl = df.loc[test_mask_full, FEATURE_COLS_NO_LIT_T].values

    xgb_temporal_nl = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw_t,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_temporal_nl.fit(X_train_t_nl, y_train_t)
    probas_t_nl = xgb_temporal_nl.predict_proba(X_test_t_nl)[:, 1]

    auroc_t_nl = roc_auc_score(y_test_t, probas_t_nl)
    auprc_t_nl = average_precision_score(y_test_t, probas_t_nl)

    print(f"\n  Temporal XGB (no literature):  AUROC={auroc_t_nl:.3f}  AUPRC={auprc_t_nl:.3f}")
    print(f"  (Cleanest test: temporal split + no literature leakage)")

    # Fisher's exact on temporal test set
    test_df = df.loc[test_mask_full].copy()
    has_gen_t = test_df["genetic_association"] > 0
    appr_t = test_df["label"] == 1
    ct_t = [
        [(has_gen_t & appr_t).sum(), (has_gen_t & ~appr_t).sum()],
        [(~has_gen_t & appr_t).sum(), (~has_gen_t & ~appr_t).sum()],
    ]
    or_t, p_t = fisher_exact(ct_t)
    print(f"\n  Genetic enrichment in temporal test set:")
    print(f"    OR={or_t:.2f}  p={p_t:.2e}")
    if p_t < 0.05:
        print(f"    -> Genetic enrichment HOLDS in prospective validation")
    else:
        print(f"    -> Not significant in temporal test (may need larger test set)")
else:
    print("\n  Insufficient test data for temporal validation")
    print(f"  (need >=10 positives and negatives, got {n_test_pos} pos, {(y_test_t==0).sum()} neg)")

# ── SHAP analysis ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SHAP ANALYSIS")
print("(Note: SHAP values reflect model reliance on each feature,")
print(" not causal importance. Literature mining may be inflated")
print(" by temporal leakage from post-approval publications.)")
print(f"{'='*60}")

xgb_full = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
)
xgb_full.fit(X, y)

explainer   = shap.TreeExplainer(
    xgb_full,
    data=X,
    feature_perturbation='interventional',
    model_output='raw'  # log-odds space — centers SHAP values around 0
)
shap_values = explainer.shap_values(X)

# For binary XGBoost, shap_values may be a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_values = shap_values[1]

feature_labels = [FEATURE_LABELS[f] for f in FEATURE_COLS]
mean_abs_shap  = np.abs(shap_values).mean(axis=0)

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
order = np.argsort(mean_abs_shap)
ax.barh([feature_labels[i] for i in order],
        mean_abs_shap[order],
        color="#4C72B0", edgecolor="white")
ax.set_xlabel("Mean |SHAP value|  (impact on approval prediction)", fontsize=11)
ax.set_title("What Evidence Types Predict Drug Approval?\n(XGBoost, full dataset)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, X, feature_names=feature_labels,
                  show=False, plot_size=None)
plt.title("SHAP Value Distribution by Evidence Type",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("results/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nFeature importance (mean |SHAP|):")
shap_df = pd.DataFrame({
    "Feature":       feature_labels,
    "mean_SHAP":     mean_abs_shap,
    "pct_nonzero":   [(X[:, i] > 0).mean() * 100
                      for i in range(len(FEATURE_COLS))],
}).sort_values("mean_SHAP", ascending=False)
print(shap_df.to_string(index=False, float_format="%.4f"))
print("[+] Saved SHAP figures")

# ── Assign therapeutic areas (ontology-based) ────────────────────────────────

# Use ontology-based classification from Open Targets (replaces keyword matching)
df_results = df.copy()
df_results["xgb_proba"] = oof_probas["XGBoost"]
df_results["az_area"] = df_results["ot_therapeutic_area"]

# Also keep keyword-based for comparison
def assign_az_area(disease_name):
    if pd.isna(disease_name):
        return "Other"
    dn = str(disease_name).lower()
    for area, keywords in AZ_AREAS.items():
        if any(kw in dn for kw in keywords):
            return area
    return "Other"

df_results["az_area_keyword"] = df_results["disease_name"].apply(assign_az_area)

# Report agreement between ontology and keyword methods
agreement = (df_results["az_area"] == df_results["az_area_keyword"]).mean()
print(f"\n  Ontology vs keyword classification agreement: {agreement:.1%}")
print(f"  (Using ontology-based classification as primary method)")

# ── AZ therapeutic area performance ──────────────────────────────────────────

print(f"\n{'='*60}")
print("AZ THERAPEUTIC AREA ANALYSIS")
print(f"{'='*60}")

area_stats = []
for area in list(AZ_AREAS.keys()) + ["Other"]:
    subset = df_results[df_results["az_area"] == area]
    if len(subset) < 10:
        continue
    pos   = int(subset["label"].sum())
    neg   = int((subset["label"] == 0).sum())
    total = len(subset)
    auroc_area = roc_auc_score(subset["label"], subset["xgb_proba"]) \
                 if pos > 0 and neg > 0 else np.nan
    auprc_area = average_precision_score(subset["label"], subset["xgb_proba"]) \
                 if pos > 0 else np.nan
    area_stats.append({
        "Area": area, "N": total, "Approved": pos,
        "Stalled": neg, "Approval_Rate": pos / total,
        "AUROC": auroc_area, "AUPRC": auprc_area,
    })
    print(f"\n  {area}  (n={total:,}, {pos:,} approved, {pos/total:.1%})")
    if not np.isnan(auroc_area):
        print(f"    AUROC={auroc_area:.3f}  AUPRC={auprc_area:.3f}")

area_df = pd.DataFrame(area_stats).dropna(subset=["AUROC"])

if len(area_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order1 = area_df.sort_values("AUROC", ascending=True)
    bar_c  = ["#DD8452" if a != "Other" else "#9E9E9E"
              for a in order1["Area"]]
    axes[0].barh(order1["Area"], order1["AUROC"],
                 color=bar_c, edgecolor="white")
    axes[0].axvline(0.5, color="k", linestyle="--", lw=1, alpha=0.5,
                    label="Random")
    axes[0].set_xlabel("AUROC", fontsize=11)
    axes[0].set_title("Prediction Performance by\nAZ Therapeutic Area",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlim([0, 1])
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    order2 = area_df.sort_values("Approval_Rate", ascending=True)
    axes[1].barh(order2["Area"], order2["Approval_Rate"] * 100,
                 color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel("Approval Rate (%)", fontsize=11)
    axes[1].set_title("Approval Rate by\nAZ Therapeutic Area",
                      fontsize=12, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("AstraZeneca Therapeutic Area Breakdown",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("results/az_therapeutic_areas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[+] Saved results/az_therapeutic_areas.png")

# ── Core hypothesis: genetic evidence vs approval ─────────────────────────────

print(f"\n{'='*60}")
print("CORE HYPOTHESIS: DOES GENETIC EVIDENCE PREDICT APPROVAL?")
print(f"{'='*60}")

has_genetic = df["genetic_association"] > 0
approved    = df["label"] == 1

rate_with    = (has_genetic & approved).sum() / has_genetic.sum()
rate_without = (~has_genetic & approved).sum() / (~has_genetic).sum()

print(f"\n  Global:")
print(f"    With genetic evidence    {has_genetic.sum():>6,} pairs  "
      f"approval rate: {rate_with:.1%}")
print(f"    Without genetic evidence {(~has_genetic).sum():>6,} pairs  "
      f"approval rate: {rate_without:.1%}")

ct = [
    [(has_genetic & approved).sum(),  (has_genetic & ~approved).sum()],
    [(~has_genetic & approved).sum(), (~has_genetic & ~approved).sum()],
]
odds_ratio, p_value = fisher_exact(ct)
print(f"\n    Fisher's exact: OR={odds_ratio:.2f}, p={p_value:.2e}")
if p_value < 0.05:
    print(f"    -> Genetic evidence IS significantly associated with approval "
          f"(OR={odds_ratio:.2f})")
else:
    print(f"    -> No significant association")

# ── Per-area genetic enrichment ───────────────────────────────────────────────

print(f"\n{'='*60}")
print("GENETIC ENRICHMENT BY AZ THERAPEUTIC AREA")
print(f"{'='*60}")

area_genetic_stats = []

for area in list(AZ_AREAS.keys()) + ["Other"]:
    subset = df_results[df_results["az_area"] == area].copy()
    if len(subset) < 20:
        continue

    has_gen_area = subset["genetic_association"] > 0
    appr_area    = subset["label"] == 1

    n_with    = has_gen_area.sum()
    n_without = (~has_gen_area).sum()

    if n_with < 5 or n_without < 5:
        continue

    rate_w = (has_gen_area & appr_area).sum() / n_with
    rate_wo = (~has_gen_area & appr_area).sum() / n_without

    ct_area = [
        [(has_gen_area & appr_area).sum(),
         (has_gen_area & ~appr_area).sum()],
        [(~has_gen_area & appr_area).sum(),
         (~has_gen_area & ~appr_area).sum()],
    ]
    or_area, p_area = fisher_exact(ct_area)

    area_genetic_stats.append({
        "Area":             area,
        "N_with_genetic":   int(n_with),
        "N_without":        int(n_without),
        "Rate_with":        rate_w,
        "Rate_without":     rate_wo,
        "Odds_Ratio":       or_area,
        "p_value":          p_area,
        "Significant":      p_area < 0.05,
    })

    sig = "*" if p_area < 0.05 else " "
    print(f"\n  {area} {sig}")
    print(f"    With genetic evidence:    {n_with:>5,} pairs  "
          f"approval rate {rate_w:.1%}")
    print(f"    Without genetic evidence: {n_without:>5,} pairs  "
          f"approval rate {rate_wo:.1%}")
    print(f"    OR={or_area:.2f}  p={p_area:.2e}")

genetic_area_df = pd.DataFrame(area_genetic_stats)

# Plot per-area odds ratios
if len(genetic_area_df) > 0:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = genetic_area_df.sort_values("Odds_Ratio", ascending=True)
    bar_colors = [
        "#2ecc71" if sig else "#e74c3c"
        for sig in order["Significant"]
    ]
    bars = ax.barh(order["Area"], order["Odds_Ratio"],
                   color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(1.0, color="k", linestyle="--", lw=1.5, alpha=0.6,
               label="No enrichment (OR=1)")
    ax.set_xlabel("Odds Ratio (genetic vs no genetic evidence)", fontsize=11)
    ax.set_title("Genetic Evidence Enrichment for Approval\nby AZ Therapeutic Area",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    import matplotlib.patches as mpatches
    sig_patch   = mpatches.Patch(color="#2ecc71", label="Significant (p<0.05)")
    insig_patch = mpatches.Patch(color="#e74c3c", label="Not significant")
    ax.legend(handles=[sig_patch, insig_patch, plt.Line2D(
        [0], [0], color="k", linestyle="--", lw=1.5, alpha=0.6,
        label="No enrichment (OR=1)")],
        fontsize=9)

    for bar, (_, row) in zip(bars, order.iterrows()):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"OR={row['Odds_Ratio']:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/genetic_enrichment_by_area.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[+] Saved results/genetic_enrichment_by_area.png")

# ── Somatic mutation paradox ──────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SOMATIC MUTATION PARADOX")
print(f"{'='*60}")
print("(High somatic mutation evidence -> negative SHAP -> less likely approved)")

has_somatic = df["somatic_mutation"] > 0
rate_s_with  = (has_somatic & approved).sum() / has_somatic.sum()
rate_s_wo    = (~has_somatic & approved).sum() / (~has_somatic).sum()
ct_som = [
    [(has_somatic & approved).sum(),  (has_somatic & ~approved).sum()],
    [(~has_somatic & approved).sum(), (~has_somatic & ~approved).sum()],
]
or_som, p_som = fisher_exact(ct_som)

print(f"\n  With somatic mutation evidence:    {has_somatic.sum():,} pairs  "
      f"approval rate {rate_s_with:.1%}")
print(f"  Without somatic mutation evidence: {(~has_somatic).sum():,} pairs  "
      f"approval rate {rate_s_wo:.1%}")
print(f"  OR={or_som:.2f}  p={p_som:.2e}")
print(f"\n  Interpretation: Somatic mutation evidence marks targets as")
print(f"  biologically relevant in cancer but does NOT predict approval.")
print(f"  Likely reflects 'undruggable' drivers (RAS, TP53, MYC) that")
print(f"  are extensively studied but rarely successfully targeted.")

# ── Missed opportunity targets ────────────────────────────────────────────────

print(f"\n{'='*60}")
print("MISSED OPPORTUNITY TARGETS")
print("(Phase 1/2 stalled, high genetic evidence)")
print(f"{'='*60}")

# Use 90th percentile of NON-ZERO genetic scores (not all data,
# since ~94% of pairs have genetic_association=0, making the global
# 90th percentile effectively 0)
nonzero_genetic = df.loc[df["genetic_association"] > 0, "genetic_association"]
threshold_90 = nonzero_genetic.quantile(0.90)
threshold_95 = nonzero_genetic.quantile(0.95)
print(f"\n  90th percentile genetic score (non-zero): {threshold_90:.4f}")
print(f"  95th percentile genetic score (non-zero): {threshold_95:.4f}")

missed = df[
    (df["label"] == 0) &
    (df["genetic_association"] >= threshold_90)
].copy()

missed["az_area"] = missed["ot_therapeutic_area"].fillna("Other")
missed = missed.sort_values("genetic_association", ascending=False)

print(f"\n  Found {len(missed):,} stalled target-disease pairs with "
      f"high genetic evidence (top 10%)")
print(f"\n  Top 25 missed opportunity target-disease pairs:")
print(f"  {'Target':<15} {'Disease':<35} {'Genetic':>8} {'Lit':>6} "
      f"{'Area':<15}")
print(f"  {'-'*15} {'-'*35} {'-'*8} {'-'*6} {'-'*15}")

for _, row in missed.head(25).iterrows():
    dn = str(row.get("disease_name", ""))[:34]
    tn = str(row.get("target_name", ""))[:14]
    print(f"  {tn:<15} {dn:<35} "
          f"{row['genetic_association']:>8.3f} "
          f"{row['literature']:>6.3f} "
          f"{row.get('az_area', 'Other'):<15}")

missed.to_csv("results/missed_opportunity_targets.csv", index=False)
print(f"\n[+] Saved results/missed_opportunity_targets.csv "
      f"({len(missed):,} targets)")

# ── Missed opportunities by AZ area ──────────────────────────────────────────

print(f"\n  Missed opportunities by AZ therapeutic area:")
area_counts = missed["az_area"].value_counts()
for area, count in area_counts.items():
    pct = count / len(missed) * 100
    print(f"    {area:<20} {count:>4,}  ({pct:.1f}%)")

# Plot missed opportunity distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter: genetic vs literature for missed opportunities
axes[0].scatter(
    missed["literature"],
    missed["genetic_association"],
    c=missed["az_area"].map({
        "Oncology": "#e74c3c", "CVRM": "#3498db",
        "Respiratory": "#2ecc71", "Rare Disease": "#9b59b6",
        "Immunology": "#f39c12", "Other": "#95a5a6",
    }).fillna("#95a5a6"),
    alpha=0.5, s=15, linewidths=0,
)
axes[0].set_xlabel("Literature Mining Score", fontsize=11)
axes[0].set_ylabel("Genetic Association Score", fontsize=11)
axes[0].set_title("Missed Opportunity Targets\n(Stalled Phase 1/2, High Genetic Evidence)",
                  fontsize=11, fontweight="bold")

import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color="#e74c3c", label="Oncology"),
    mpatches.Patch(color="#3498db", label="CVRM"),
    mpatches.Patch(color="#2ecc71", label="Respiratory"),
    mpatches.Patch(color="#9b59b6", label="Rare Disease"),
    mpatches.Patch(color="#f39c12", label="Immunology"),
    mpatches.Patch(color="#95a5a6", label="Other"),
]
axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right")
axes[0].spines[["top", "right"]].set_visible(False)

# Bar: count by area
area_counts.plot.barh(ax=axes[1], color="#DD8452", edgecolor="white")
axes[1].set_xlabel("Number of Missed Opportunity Pairs", fontsize=11)
axes[1].set_title("Missed Opportunities by\nAZ Therapeutic Area",
                  fontsize=11, fontweight="bold")
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("Genetically-Supported Targets Stalled in Phase 1/2",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/missed_opportunities.png", dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved results/missed_opportunities.png")

# ── Save full summary ─────────────────────────────────────────────────────────

summary_lines = [
    "CLINICAL TRIAL SUCCESS PREDICTION v2 — FULL SUMMARY",
    "=" * 60,
    "",
    f"Dataset: {len(df):,} target-disease pairs",
    f"  Approved (Phase 4):    {df['label'].sum():,}",
    f"  Stalled (Phase 1/2):   {(df['label']==0).sum():,}",
    f"  Unique targets:        {df['ensembl_id'].nunique():,}",
    f"  Unique diseases:       {df['efo_id_norm'].nunique():,}",
    "",
    "MODEL PERFORMANCE (5-fold CV):",
    *[f"  {k}: AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}"
      for k, v in results.items()],
    "",
    "SHAP FEATURE IMPORTANCE (XGBoost, full dataset):",
    *[f"  {r['Feature']:<25} mean|SHAP|={r['mean_SHAP']:.4f}  "
      f"coverage={r['pct_nonzero']:.1f}%"
      for _, r in shap_df.iterrows()],
    "",
    "CORE HYPOTHESIS — GLOBAL:",
    f"  With genetic evidence:    {has_genetic.sum():,} pairs  "
    f"approval rate {rate_with:.1%}",
    f"  Without genetic evidence: {(~has_genetic).sum():,} pairs  "
    f"approval rate {rate_without:.1%}",
    f"  Fisher's exact: OR={odds_ratio:.2f}, p={p_value:.2e}",
    "",
    "GENETIC ENRICHMENT BY AZ AREA:",
    *[f"  {r['Area']:<15} OR={r['Odds_Ratio']:.2f}  "
      f"p={r['p_value']:.2e}  sig={r['Significant']}"
      for _, r in genetic_area_df.iterrows()],
    "",
    "SOMATIC MUTATION PARADOX:",
    f"  With somatic mutation evidence:    "
    f"approval rate {rate_s_with:.1%}",
    f"  Without somatic mutation evidence: "
    f"approval rate {rate_s_wo:.1%}",
    f"  OR={or_som:.2f}  p={p_som:.2e}",
    "",
    "MISSED OPPORTUNITIES (stalled, high genetic evidence):",
    f"  Total: {len(missed):,} pairs",
    *[f"  {area}: {count}"
      for area, count in area_counts.items()],
    "",
    "OUTPUT FILES:",
    "  results/roc_prc_curves.png",
    "  results/shap_importance.png",
    "  results/shap_beeswarm.png",
    "  results/az_therapeutic_areas.png",
    "  results/genetic_enrichment_by_area.png",
    "  results/missed_opportunities.png",
    "  results/missed_opportunity_targets.csv",
]

with open("results/model_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\n[+] Saved results/model_summary.txt")
print(f"\n{'='*60}")
print("DONE — all outputs in results/")
print(f"{'='*60}")