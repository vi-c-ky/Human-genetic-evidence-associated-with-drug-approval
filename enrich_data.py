"""
enrich_data.py
==============
Fetches additional metadata for temporal validation and ontology-based
therapeutic area classification:
  1. First approval year for each drug (from ChEMBL)
  2. Therapeutic areas for each disease (from Open Targets ontology)

Run:
    python enrich_data.py
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
SLEEP = 0.05
TIMEOUT = 15

# ── 1. Fetch first_approval year from ChEMBL ─────────────────────────────────

def fetch_approval_years():
    path = "data/drug_approval_years.csv"
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"[1] Loading cached {path}")
        return pd.read_csv(path)

    print("[1] Fetching first_approval years from ChEMBL...")
    from chembl_webresource_client.new_client import new_client
    mol_client = new_client.molecule

    drug_df = pd.read_csv("data/ot_drug_data.csv")
    unique_drugs = drug_df["drug_id"].unique()
    print(f"    {len(unique_drugs)} unique drugs to query")

    records = []
    batch_size = 50
    for i in tqdm(range(0, len(unique_drugs), batch_size), desc="    ChEMBL batches"):
        batch = list(unique_drugs[i:i + batch_size])
        try:
            results = mol_client.filter(
                molecule_chembl_id__in=batch
            ).only(['molecule_chembl_id', 'first_approval'])
            for r in results:
                if r.get('first_approval'):
                    records.append({
                        'drug_id': r['molecule_chembl_id'],
                        'first_approval': int(r['first_approval']),
                    })
        except Exception as e:
            pass  # skip failed batches

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"    Found approval years for {len(df)} drugs")
    return df


# ── 2. Fetch therapeutic areas from Open Targets ontology ─────────────────────

DISEASE_TA_QUERY = """
query DiseaseInfo($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    therapeuticAreas {
      id
      name
    }
  }
}
"""

# Mapping from OT therapeutic area names to simplified categories
TA_MAPPING = {
    "cancer or benign tumor": "Oncology",
    "cardiovascular disease": "CVRM",
    "metabolic disease": "CVRM",
    "endocrine system disease": "CVRM",
    "kidney disease": "CVRM",
    "respiratory or thoracic disease": "Respiratory",
    "immune system disease": "Immunology",
    "genetic, familial or congenital disease": "Rare Disease",
}


def fetch_therapeutic_areas():
    path = "data/disease_therapeutic_areas.csv"
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"[2] Loading cached {path}")
        return pd.read_csv(path)

    print("[2] Fetching therapeutic areas from Open Targets ontology...")
    final_df = pd.read_csv("data/final_dataset.csv")
    unique_diseases = final_df["efo_id_norm"].unique()
    print(f"    {len(unique_diseases)} unique diseases to query")

    records = []
    for efo_id in tqdm(unique_diseases, desc="    Diseases"):
        try:
            r = requests.post(
                OT_API,
                json={"query": DISEASE_TA_QUERY, "variables": {"efoId": efo_id}},
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                disease = r.json().get("data", {}).get("disease")
                if disease and disease.get("therapeuticAreas"):
                    ta_names = [ta["name"] for ta in disease["therapeuticAreas"]]
                    # Map to simplified category (take first match in priority order)
                    assigned = "Other"
                    for ta_name in ta_names:
                        if ta_name in TA_MAPPING:
                            assigned = TA_MAPPING[ta_name]
                            break
                    records.append({
                        "efo_id_norm": efo_id,
                        "ot_therapeutic_area": assigned,
                        "ot_ta_raw": "|".join(ta_names),
                    })
                else:
                    records.append({
                        "efo_id_norm": efo_id,
                        "ot_therapeutic_area": "Other",
                        "ot_ta_raw": "",
                    })
        except Exception:
            records.append({
                "efo_id_norm": efo_id,
                "ot_therapeutic_area": "Other",
                "ot_ta_raw": "",
            })
        time.sleep(SLEEP)

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"    Classified {len(df)} diseases")
    print(f"    Distribution:")
    print(df["ot_therapeutic_area"].value_counts().to_string())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("DATA ENRICHMENT")
    print("=" * 60)
    approval_df = fetch_approval_years()
    ta_df = fetch_therapeutic_areas()
    print("\nDone. Files saved to data/")
