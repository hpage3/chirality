#!/usr/bin/env python3
"""
verify_homochirality_from_rcsb.py

Reads ids.tsv (pdb_id \t chain) and queries RCSB Data API to confirm whether each
chain belongs to a polymer entity with entity_poly.type == polypeptide(L).

Outputs a TSV summary and prints a short report.

Requires: requests
"""

import csv
import sys
from collections import defaultdict
import requests

RCSB = "https://data.rcsb.org/rest/v1/core"

def get_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def main(ids_tsv: str, out_tsv: str):
    # Read ids.tsv
    pairs = []
    with open(ids_tsv, "r", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pdb_id, chain = parts[0].lower(), parts[1]
            pairs.append((pdb_id, chain))

    # For each PDB, map chain -> entity_id using polymer_entity_instance (usually works with auth_asym_id,
    # but if it fails you may need to use label_asym_id instead).
    results = []
    for pdb_id, chain in pairs:
        inst_url = f"{RCSB}/polymer_entity_instance/{pdb_id}/{chain}"
        try:
            inst = get_json(inst_url)
        except Exception as e:
            results.append({
                "pdb_id": pdb_id, "chain": chain, "status": "ERROR_instance_lookup",
                "entity_id": "", "entity_poly_type": "", "nstd_monomer": "", "nstd_linkage": "",
                "rcsb_non_std_monomer_count": "", "error": str(e),
            })
            continue

        entity_id = inst.get("rcsb_polymer_entity_instance_container_identifiers", {}).get("entity_id", "")
        if not entity_id:
            results.append({
                "pdb_id": pdb_id, "chain": chain, "status": "ERROR_no_entity_id",
                "entity_id": "", "entity_poly_type": "", "nstd_monomer": "", "nstd_linkage": "",
                "rcsb_non_std_monomer_count": "", "error": "No entity_id in instance record",
            })
            continue

        ent_url = f"{RCSB}/polymer_entity/{pdb_id}/{entity_id}"
        try:
            ent = get_json(ent_url)
        except Exception as e:
            results.append({
                "pdb_id": pdb_id, "chain": chain, "status": "ERROR_entity_lookup",
                "entity_id": entity_id, "entity_poly_type": "", "nstd_monomer": "", "nstd_linkage": "",
                "rcsb_non_std_monomer_count": "", "error": str(e),
            })
            continue

        entity_poly = ent.get("entity_poly", {}) or {}
        entity_poly_type = entity_poly.get("type", "")
        nstd_monomer = entity_poly.get("nstd_monomer", "")
        nstd_linkage = entity_poly.get("nstd_linkage", "")
        nonstd_count = entity_poly.get("rcsb_non_std_monomer_count", "")

        status = "OK_polypeptide(L)" if entity_poly_type == "polypeptide(L)" else "FLAG_not_polypeptide(L)"

        results.append({
            "pdb_id": pdb_id,
            "chain": chain,
            "status": status,
            "entity_id": entity_id,
            "entity_poly_type": entity_poly_type,
            "nstd_monomer": nstd_monomer,
            "nstd_linkage": nstd_linkage,
            "rcsb_non_std_monomer_count": nonstd_count,
            "error": "",
        })

    # Write TSV
    fields = [
        "pdb_id","chain","status","entity_id","entity_poly_type",
        "nstd_monomer","nstd_linkage","rcsb_non_std_monomer_count","error"
    ]
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in results:
            w.writerow(row)

    # Print summary
    counts = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1

    print("Summary:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"\nWrote: {out_tsv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} ids.tsv out.tsv", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])

