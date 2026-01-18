#!/usr/bin/env python3
"""
qc_collect_sb_bin_costs.py

Minimal QC analyzer for SB bin-resolved costs.

Uses SB-produced:
  - cost_by_bin.tsv
  - cost_by_feature.tsv (optional)

Implements ONLY:
  Steps A–C (QC)
"""

from pathlib import Path
import pandas as pd
import argparse
import re


def parse_run_meta(run_name):
    """
    Extract feature mode and seed from run directory name.
    """
    mode = None
    if "_theta_" in run_name:
        mode = "theta"
    elif "_rms_" in run_name:
        mode = "rms"
    elif "_both_" in run_name:
        mode = "both"

    m = re.search(r"_s(\d+)$", run_name)
    seed = int(m.group(1)) if m else None

    return mode, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True,
                    help="04_sb/output")
    ap.add_argument("--out", required=True,
                    help="QC output TSV")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    rows = []

    for run in sorted(d for d in runs_root.iterdir() if d.is_dir()):
        bin_file = run / "cost_by_bin.tsv"
        if not bin_file.exists():
            continue

        mode, seed = parse_run_meta(run.name)

        df = pd.read_csv(bin_file, sep="\t")
        df["run"] = run.name
        df["feature_mode"] = mode
        df["seed"] = seed

        rows.append(df)

    if not rows:
        raise RuntimeError("No cost_by_bin.tsv files found")

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(args.out, sep="\t", index=False)

    print(f"[OK] Wrote QC bin-cost table: {args.out}")


if __name__ == "__main__":
    main()
