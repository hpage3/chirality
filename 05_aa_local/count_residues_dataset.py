#!/usr/bin/env python3

import os
import csv
from collections import Counter, defaultdict
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fingerprint-dir",
        default="fingerprints/whole_protein",
        help="Directory containing *_fingerprint.csv files"
    )
    ap.add_argument(
        "--out",
        default="output/aa_counts_from_fingerprints.tsv",
        help="Output TSV file"
    )
    ap.add_argument(
        "--per-structure",
        action="store_true",
        help="Also write per-structure AA counts"
    )
    args = ap.parse_args()

    dataset_counts = Counter()
    per_structure = defaultdict(Counter)

    fp_files = sorted(
        f for f in os.listdir(args.fingerprint_dir)
        if f.endswith("_fingerprint.csv")
    )

    if not fp_files:
        raise RuntimeError("No fingerprint CSV files found")

    for fn in fp_files:
        structure_id = fn.replace("_fingerprint.csv", "")
        path = os.path.join(args.fingerprint_dir, fn)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "aa_i" not in reader.fieldnames:
                raise RuntimeError(f"{fn} missing aa_i column")

            for row in reader:
                aa = row["aa_i"].strip()
                if not aa:
                    continue

                dataset_counts[aa] += 1
                per_structure[structure_id][aa] += 1

    # ---- write dataset-level counts ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total = sum(dataset_counts.values())

    with open(args.out, "w", newline="") as out:
        out.write("aa\tcount\tfraction\n")
        for aa in sorted(dataset_counts):
            count = dataset_counts[aa]
            frac = count / total
            out.write(f"{aa}\t{count}\t{frac:.6f}\n")

    print(f"✓ Wrote dataset AA counts → {args.out}")

    # ---- optional per-structure output ----
    if args.per_structure:
        per_struct_out = args.out.replace(".tsv", "_per_structure.tsv")
        aas = sorted(dataset_counts.keys())

        with open(per_struct_out, "w", newline="") as out:
            out.write("structure\t" + "\t".join(aas) + "\n")
            for struct in sorted(per_structure):
                row = [struct] + [str(per_structure[struct].get(aa, 0)) for aa in aas]
                out.write("\t".join(row) + "\n")

        print(f"✓ Wrote per-structure AA counts → {per_struct_out}")


if __name__ == "__main__":
    main()

