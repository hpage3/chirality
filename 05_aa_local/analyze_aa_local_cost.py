#!/usr/bin/env python3
"""
analyze_aa_cost_from_boxes.py

Derive amino-acid-specific Schrödinger Bridge costs
from *_boxes_normals.csv files.

AA attribution:
- Each plane contributes TWO amino acids (res_i and res_j)
- Both are assigned the SB transport cost of the run
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from math import sqrt

AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

AA20 = set(AA3_TO_1.values())


def load_cost(run_dir: Path) -> float | None:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return d.get("transport_cost", None)


def bootstrap_mean(x, n_boot=5000, seed=1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return x.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="chirality/output directory")
    ap.add_argument("--runs_root", required=True,
                    help="directory containing sb_* run folders")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------
    # Load SB costs
    # --------------------------------------------------
    run_cost = {}
    for d in runs_root.iterdir():
        if d.is_dir() and d.name.startswith("sb_"):
            c = load_cost(d)
            if c is not None:
                run_cost[d.name] = c

    print(f"[INFO] Loaded {len(run_cost)} SB runs")



    # --------------------------------------------------
    # Collect AA observations (FAST, VECTORIZED)
    # --------------------------------------------------
    rows = []

    run_df = (
        pd.DataFrame.from_dict(run_cost, orient="index", columns=["transport_cost"])
        .reset_index()
        .rename(columns={"index": "run"})
    )

    csv_files = list(output_dir.glob("*_boxes_normals.csv"))
    print(f"[INFO] Processing {len(csv_files)} boxes files")

    for csv in tqdm(csv_files, desc="Processing boxes"):
        df = pd.read_csv(csv, usecols=["resname_i", "resname_j"])

        # Convert 3-letter → 1-letter
        aa_i = df["resname_i"].map(AA3_TO_1)
        aa_j = df["resname_j"].map(AA3_TO_1)

        aa = pd.concat([aa_i, aa_j], ignore_index=True)
        aa = aa[aa.isin(AA20)]

        # Build AA dataframe for this structure
        aa_df = pd.DataFrame({"aa": aa})

        # Cartesian join with runs (fast)
        aa_run = aa_df.merge(run_df, how="cross")

        rows.append(aa_run)

    data = pd.concat(rows, ignore_index=True)
    print(f"[INFO] Total AA–cost observations: {len(data)}")

    data.to_csv(out_dir / "aa_cost_raw.tsv", sep="\t", index=False)
   
    # --------------------------------------------------
    # Run-normalized AA cost
    # --------------------------------------------------
    # Step 1: mean AA cost within each run
    per_run = (
        data
        .groupby(["run", "aa"], as_index=False)
        .agg(mean_cost=("transport_cost", "mean"))
    )

    per_run.to_csv(
        out_dir / "aa_cost_per_run.tsv",
        sep="\t",
        index=False
    )

    # Step 2: average those means across runs
    run_norm = (
        per_run
        .groupby("aa", as_index=False)
        .agg(
            mean_cost_across_runs=("mean_cost", "mean"),
            std_across_runs=("mean_cost", "std"),
            n_runs=("mean_cost", "count")
        )
    )

    global_mean = run_norm["mean_cost_across_runs"].mean()
    run_norm["delta_vs_global"] = (
        run_norm["mean_cost_across_runs"] - global_mean
    )

    run_norm = run_norm.sort_values(
        "mean_cost_across_runs", ascending=False
    )

    run_norm.to_csv(
        out_dir / "aa_cost_run_normalized.tsv",
        sep="\t",
        index=False
    )

    print("[OK] wrote aa_cost_per_run.tsv")
    print("[OK] wrote aa_cost_run_normalized.tsv")

   
    # --------------------------------------------------
    # AA summary (analytic CI — FAST)
    # --------------------------------------------------
    global_mean = data["transport_cost"].mean()
    summary = []

    grouped = data.groupby("aa", sort=False)["transport_cost"]

    for aa, s in grouped:
        x = s.values
        n = x.size
        mean = x.mean()
        std = x.std(ddof=1)
        se = std / sqrt(n)

        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se

        summary.append({
            "aa": aa,
            "n": n,
            "mean_cost": mean,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "delta_vs_global": mean - global_mean
        })

    out = pd.DataFrame(summary).sort_values("mean_cost", ascending=False)
    out.to_csv(out_dir / "aa_cost_summary.tsv", sep="\t", index=False)
    print("[OK] wrote aa_cost_summary.tsv")

if __name__ == "__main__":
    main()
