"""
threshold_sensitivity.py
========================
Genetic evidence threshold sensitivity analysis.
Runs Fisher's exact test at genetic_association thresholds > 0, > 0.5, > 0.8
globally and by therapeutic area. Saves Supplementary Table S2.
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# Load data
df = pd.read_csv("data/final_dataset.csv").fillna(0)
ta_df = pd.read_csv("data/disease_therapeutic_areas.csv")
df = df.merge(ta_df[["efo_id_norm", "ot_therapeutic_area"]], on="efo_id_norm", how="left")
df["ot_therapeutic_area"] = df["ot_therapeutic_area"].fillna("Other")

approved = df["label"] == 1

# Describe genetic_association score distribution
nonzero = df.loc[df["genetic_association"] > 0, "genetic_association"]
print("Genetic association score distribution (non-zero only):")
print(f"  N non-zero: {len(nonzero)} / {len(df)} ({len(nonzero)/len(df)*100:.1f}%)")
print(f"  Mean: {nonzero.mean():.3f}")
print(f"  Median: {nonzero.median():.3f}")
print(f"  25th percentile: {nonzero.quantile(0.25):.3f}")
print(f"  75th percentile: {nonzero.quantile(0.75):.3f}")
print(f"  90th percentile: {nonzero.quantile(0.90):.3f}")
print(f"  N with score > 0.5: {(df['genetic_association'] > 0.5).sum()}")
print(f"  N with score > 0.8: {(df['genetic_association'] > 0.8).sum()}")

# Threshold sensitivity
thresholds = [0, 0.5, 0.8]
results = []

for thresh in thresholds:
    has_genetic = df["genetic_association"] > thresh
    n_with = has_genetic.sum()
    n_without = (~has_genetic).sum()

    if n_with < 5 or n_without < 5:
        print(f"\nThreshold > {thresh}: insufficient data (n_with={n_with})")
        continue

    ct = [
        [(has_genetic & approved).sum(), (has_genetic & ~approved).sum()],
        [(~has_genetic & approved).sum(), (~has_genetic & ~approved).sum()],
    ]
    odds_ratio, p_value = fisher_exact(ct)
    rate_with = (has_genetic & approved).sum() / n_with
    rate_without = (~has_genetic & approved).sum() / n_without

    results.append({
        "Threshold": f"> {thresh}",
        "N_with_genetic": int(n_with),
        "N_without_genetic": int(n_without),
        "Approval_rate_with": round(rate_with, 4),
        "Approval_rate_without": round(rate_without, 4),
        "Odds_Ratio": round(odds_ratio, 2),
        "p_value": p_value,
        "Significant": p_value < 0.05,
    })

    print(f"\nThreshold > {thresh}:")
    print(f"  With genetic (n={n_with:,}): approval rate {rate_with:.1%}")
    print(f"  Without (n={n_without:,}): approval rate {rate_without:.1%}")
    print(f"  OR = {odds_ratio:.2f}, p = {p_value:.2e}")

# By therapeutic area at each threshold
print("\n" + "=" * 60)
print("BY THERAPEUTIC AREA")
print("=" * 60)

areas = ["Oncology", "CVRM", "Respiratory", "Immunology", "Rare Disease", "Other"]
area_results = []

for area in areas:
    subset = df[df["ot_therapeutic_area"] == area]
    appr_area = subset["label"] == 1

    for thresh in thresholds:
        has_gen = subset["genetic_association"] > thresh
        n_with = has_gen.sum()
        n_without = (~has_gen).sum()

        if n_with < 2 or n_without < 2:
            continue

        ct = [
            [(has_gen & appr_area).sum(), (has_gen & ~appr_area).sum()],
            [(~has_gen & appr_area).sum(), (~has_gen & ~appr_area).sum()],
        ]
        odds_ratio, p_value = fisher_exact(ct)

        area_results.append({
            "Area": area,
            "Threshold": f"> {thresh}",
            "N_with_genetic": int(n_with),
            "N_without_genetic": int(n_without),
            "Odds_Ratio": round(odds_ratio, 2),
            "p_value": p_value,
            "Significant": p_value < 0.05,
        })

        print(f"  {area} (threshold > {thresh}): n_with={n_with}, OR={odds_ratio:.2f}, p={p_value:.2e}")

# Save as Supplementary Table S2
results_df = pd.DataFrame(results)
results_df.to_csv("results/threshold_sensitivity_global.csv", index=False)

area_results_df = pd.DataFrame(area_results)
area_results_df.to_csv("results/threshold_sensitivity_by_area.csv", index=False)

# Combined table for Supplementary Table S2
print("\n" + "=" * 60)
print("SUPPLEMENTARY TABLE S2")
print("=" * 60)
print("\nGlobal:")
print(results_df.to_string(index=False))
print("\nBy therapeutic area:")
print(area_results_df.to_string(index=False))

print("\n[+] Saved results/threshold_sensitivity_global.csv")
print("[+] Saved results/threshold_sensitivity_by_area.csv")
