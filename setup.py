"""
build_trial_dataset_v5.py
=========================
Clinical trial success prediction — dataset builder.

All field names verified against live Open Targets API schema:
  - mechanismsOfAction.rows[].targets  (not linkedTargets)
  - maxClinicalStage as "PHASE_1".."PHASE_4"  (not maxPhaseForIndication)
  - maximumClinicalStage  (not maximumClinicalTrialPhase)

Run:
    delete data/ot_drug_data.csv
    delete data/target_disease_labels.csv
    delete data/ot_evidence_scores.csv
    delete data/final_dataset.csv
    python build_trial_dataset_v5.py

Steps 1a/1b load from cache. All other steps run fresh.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

OT_API    = "https://api.platform.opentargets.org/api/v4/graphql"
SLEEP     = 0.1
TIMEOUT   = 15
PAGE_SIZE = 500

FEATURE_COLS = [
    "genetic_association",
    "somatic_mutation",
    "literature",
    "rna_expression",
    "animal_model",
    "affected_pathway",
]

# maxClinicalStage string → integer phase
PHASE_MAP = {
    "EARLY_PHASE_1": 1,
    "PHASE_1":       1, "PHASE_I":   1,
    "PHASE_1_2":     1,
    "IND":           1,
    "PHASE_2":       2, "PHASE_II":  2,
    "PHASE_3":       3, "PHASE_III": 3,
    "PHASE_4":       4, "PHASE_IV":  4,
    "APPROVAL":      4,  # ← this was the bug
    "APPROVED":      4,  # fallback
}
def parse_phase(stage_str):
    if not stage_str:
        return None
    return PHASE_MAP.get(str(stage_str).upper().strip(), None)

def load_csv_if_valid(path):
    if os.path.exists(path) and os.path.getsize(path) > 100:
        df = pd.read_csv(path)
        if len(df) > 0:
            return df
    return None

def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"   Deleted stale cache: {path}")

# ── Steps 1a/b: Load cached ChEMBL molecule IDs ───────────────────────────────

def load_molecule_ids():
    print("[Steps 1a/b] Loading cached ChEMBL CSVs...")
    ind_df  = pd.read_csv("data/chembl_indications.csv")
    mech_df = pd.read_csv("data/chembl_mechanisms.csv")

    mol_with_ind  = set(ind_df["molecule_chembl_id"].dropna())
    mol_with_mech = set(mech_df["molecule_chembl_id"].dropna())
    molecules     = list(mol_with_ind & mol_with_mech)

    print(f"   Molecules with indication data:  {len(mol_with_ind):,}")
    print(f"   Molecules with mechanism data:   {len(mol_with_mech):,}")
    print(f"   Intersection (both):             {len(molecules):,}")
    return molecules

# ── Step 2: Pull drug-target-disease-phase from Open Targets ─────────────────

DRUG_QUERY = """
query DrugData($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targets {
          id
          approvedSymbol
        }
      }
    }
    indications {
      rows {
        disease {
          id
          name
        }
        maxClinicalStage
      }
    }
  }
}
"""


def fetch_drug(chembl_id):
    """Query OT drug endpoint. Returns list of (target, disease, phase) records."""
    try:
        r = requests.post(
            OT_API,
            json={"query": DRUG_QUERY, "variables": {"chemblId": chembl_id}},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return []

        drug_data = r.json().get("data", {}).get("drug")
        if not drug_data:
            return []   # Drug not in Open Targets

        mech_rows = (drug_data.get("mechanismsOfAction") or {}).get("rows") or []
        ind_rows  = (drug_data.get("indications")        or {}).get("rows") or []

        if not mech_rows or not ind_rows:
            return []

        records = []
        for mech in mech_rows:
            for target in (mech.get("targets") or []):
                ensembl_id = target.get("id")
                symbol     = target.get("approvedSymbol")
                if not ensembl_id:
                    continue
                for indication in ind_rows:
                    disease  = indication.get("disease") or {}
                    efo_id   = disease.get("id")
                    phase    = parse_phase(indication.get("maxClinicalStage"))
                    if not efo_id or phase is None:
                        continue
                    records.append({
                        "ensembl_id":   ensembl_id,
                        "target_name":  symbol,
                        "efo_id_norm":  efo_id,
                        "disease_name": disease.get("name"),
                        "phase":        phase,
                        "drug_id":      chembl_id,
                    })
        return records

    except Exception:
        return []


def pull_ot_drug_data(molecules):
    path = "data/ot_drug_data.csv"

    cached = load_csv_if_valid(path)
    if cached is not None:
        print(f"\n[Step 2] Loading ot_drug_data.csv from cache ({len(cached):,} rows)...")
        return cached

    print(f"\n[Step 2] Querying Open Targets for {len(molecules):,} molecules...")
    print("   Using corrected field names: mechanismsOfAction, maxClinicalStage")

    all_records = []
    found       = 0
    not_found   = 0

    for mol_id in tqdm(molecules, desc="   Drugs"):
        rows = fetch_drug(mol_id)
        if rows:
            all_records.extend(rows)
            found += 1
        else:
            not_found += 1
        time.sleep(SLEEP)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["ensembl_id", "target_name", "efo_id_norm",
                 "disease_name", "phase", "drug_id"]
    )
    df.to_csv(path, index=False)

    pct = found / len(molecules) * 100 if molecules else 0
    print(f"\n   Drugs found in Open Targets: {found:,} / {len(molecules):,} ({pct:.0f}%)")
    print(f"   Drugs not found:             {not_found:,}")
    print(f"   Raw target-disease rows:     {len(df):,}")

    if found == 0:
        print("\n   ✗ Still 0 drugs found. Trying spot check on CHEMBL941...")
        r = requests.post(
            OT_API,
            json={"query": DRUG_QUERY, "variables": {"chemblId": "CHEMBL941"}},
            timeout=15,
        )
        import json
        print(f"   CHEMBL941 status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)[:800]}")

    return df

# ── Step 3: Aggregate to target-disease level and label ───────────────────────

def build_labels(drug_df):
    path = "data/target_disease_labels.csv"

    print("\n[Step 3] Aggregating and labelling...")

    if len(drug_df) == 0:
        print("   ✗ Drug data is empty — cannot build labels.")
        return pd.DataFrame()

    # Max phase per unique target-disease pair across all drugs
    agg = (
        drug_df
        .groupby(["ensembl_id", "efo_id_norm", "target_name", "disease_name"])
        ["phase"]
        .max()
        .reset_index()
    )

    print(f"\n   Phase distribution:")
    print(agg["phase"].value_counts().sort_index().to_string())

    # Exclude Phase 3 (ambiguous — ongoing or failed)
    before = len(agg)
    agg    = agg[agg["phase"] != 3].copy()
    agg["label"] = (agg["phase"] == 4).astype(int)

    pos   = agg["label"].sum()
    neg   = (agg["label"] == 0).sum()
    ratio = neg // max(pos, 1)

    print(f"\n   Excluded {before - len(agg):,} Phase 3 pairs")
    print(f"   Positives (Phase 4):    {pos:,}")
    print(f"   Negatives (Phase 1/2):  {neg:,}")
    print(f"   Ratio:                  1:{ratio}")
    print(f"   Unique targets:         {agg['ensembl_id'].nunique():,}")
    print(f"   Unique diseases:        {agg['efo_id_norm'].nunique():,}")

    agg.to_csv(path, index=False)
    return agg

# ── Step 4: Fetch evidence scores via disease.associatedTargets ───────────────

DISEASE_ASSOC_QUERY = """
query DiseaseTargetAssociations($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""


def fetch_disease_associations(efo_id):
    records = []
    index   = 0
    while True:
        try:
            r = requests.post(
                OT_API,
                json={
                    "query": DISEASE_ASSOC_QUERY,
                    "variables": {"efoId": efo_id, "index": index, "size": PAGE_SIZE},
                },
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception:
            return []

        assoc = (
            data.get("data", {})
                .get("disease", {})
                .get("associatedTargets", {})
        )
        if not assoc:
            break

        rows  = assoc.get("rows", [])
        total = assoc.get("count", 0)

        for row in rows:
            rec = {
                "ensembl_id":    row["target"]["id"],
                "efo_id_norm":   efo_id,
                "overall_score": row["score"],
            }
            for ds in row.get("datatypeScores", []):
                rec[ds["id"]] = ds["score"]
            records.append(rec)

        index += 1
        if len(records) >= total:
            break

    return records


def fetch_all_evidence_scores(labels_df):
    path = "data/ot_evidence_scores.csv"

    cached = load_csv_if_valid(path)
    if cached is not None:
        print(f"\n[Step 4] Loading ot_evidence_scores.csv from cache ({len(cached):,} rows)...")
        return cached

    delete_if_exists(path)

    print("\n[Step 4] Fetching evidence scores (disease.associatedTargets)...")

    unique_diseases = labels_df["efo_id_norm"].unique()
    needed_pairs    = set(zip(labels_df["ensembl_id"], labels_df["efo_id_norm"]))
    all_records     = []
    found = not_found = 0

    for efo_id in tqdm(unique_diseases, desc="   Diseases"):
        rows = fetch_disease_associations(efo_id)
        if rows:
            found += 1
            for row in rows:
                if (row["ensembl_id"], efo_id) in needed_pairs:
                    all_records.append(row)
        else:
            not_found += 1
        time.sleep(SLEEP)

    pct = found / len(unique_diseases) * 100 if len(unique_diseases) else 0
    print(f"\n   Diseases found:        {found:,} / {len(unique_diseases):,} ({pct:.0f}%)")
    print(f"   Diseases not found:    {not_found:,}")
    print(f"   Matching pairs:        {len(all_records):,}")

    evidence_df = pd.DataFrame(all_records).fillna(0)
    evidence_df.to_csv(path, index=False)
    return evidence_df

# ── Step 5: Build final dataset ───────────────────────────────────────────────

def build_final_dataset(labels_df, evidence_df):
    print("\n[Step 5] Building final dataset...")

    if len(labels_df) == 0 or len(evidence_df) == 0:
        print("   ✗ Labels or evidence is empty.")
        return None

    dataset = labels_df.merge(
        evidence_df, on=["ensembl_id", "efo_id_norm"], how="inner"
    ).fillna(0)

    for col in FEATURE_COLS:
        if col not in dataset.columns:
            dataset[col] = 0.0

    dataset.to_csv("data/final_dataset.csv", index=False)

    pos = dataset["label"].sum()
    neg = (dataset["label"] == 0).sum()

    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs:           {len(dataset):,}")
    print(f"Positives (Phase 4):   {pos:,}")
    print(f"Negatives (Phase 1/2): {neg:,}")
    print(f"Ratio:                 1:{neg//max(pos,1)}")
    print(f"Unique targets:        {dataset['ensembl_id'].nunique():,}")
    print(f"Unique diseases:       {dataset['efo_id_norm'].nunique():,}")
    print(f"\nFeature coverage (% non-zero):")
    for col in FEATURE_COLS:
        print(f"   {col:<25} {(dataset[col]>0).mean()*100:.1f}%")
    print(f"\n✓ Saved → data/final_dataset.csv")
    return dataset

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CLINICAL TRIAL SUCCESS — DATASET BUILDER v5")
    print("=" * 60)

    molecules   = load_molecule_ids()
    drug_df     = pull_ot_drug_data(molecules)
    labels_df   = build_labels(drug_df)
    evidence_df = fetch_all_evidence_scores(labels_df)
    dataset     = build_final_dataset(labels_df, evidence_df)

    if dataset is not None:
        print("\n✓ Done. Run modelling script on data/final_dataset.csv")
    else:
        print("\n✗ Pipeline incomplete — see diagnostics above.")