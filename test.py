import requests

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"

# Test with EGFR — known good target
TEST_QUERY = """
query {
  target(ensemblId: "ENSG00000146648") {
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 5}) {
      count
      rows {
        disease { id name }
        score
      }
    }
  }
}
"""

r = requests.post(OT_API, json={"query": TEST_QUERY}, timeout=15)
print("Status:", r.status_code)
print(r.json())