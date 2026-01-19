#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def load_table(path: Path, model: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing AA table: {path}")

    df = pd.read_csv(path, sep="\t")

    if "aa" not in df.columns:
        raise ValueError(f"{path} missing 'aa' column")

    df = df.copy()

    # Normalize column names
    if "mean_cost" in df.columns:
        pass
    elif "mean_cost_across_runs" in df.columns:
        df["mean_cost"] = df["mean_cost_across_runs"]
    else:
        raise ValueError(f"{path} missing mean cost column")

    df["model"] = model
    df["aa"] = df["aa"].astype(str)

    return df


def detect_normalized(df: pd.DataFrame, tol=1e-3) -> bool:
    if "run" not in df.columns:
        return abs(df["mean_cost"].sum() - 1.0) < tol

    sums = df.groupby("run")["mean_cost"].sum()
    return all(abs(s - 1.0) < tol for s in sums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True, help="Theta-only aa_cost_per_run.tsv")
    ap.add_argument("--rms",   required=True, help="RMS-only aa_cost_per_run.tsv")
    ap.add_argument("--both",  required=True, help="Theta+RMS aa_cost_per_run.tsv")
    ap.add_argument("--out",   required=True)

    args = ap.parse_args()

    dfs = [
        load_table(Path(args.theta), "theta"),
        load_table(Path(args.rms),   "rms"),
        load_table(Path(args.both),  "both"),
    ]

    df = pd.concat(dfs, ignore_index=True)

    is_normalized = detect_normalized(df)

    # 🔑 CRITICAL FIX: average over runs BEFORE plotting
    df_plot = (
        df
        .groupby(["model", "aa"], as_index=False)
        .agg(mean_cost=("mean_cost", "mean"))
    )

    pivot = (
        df_plot
        .pivot(index="aa", columns="model", values="mean_cost")
    )

    # Sort amino acids by theta-only cost (ascending)
    pivot = pivot.sort_values(by="theta", ascending=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Second panel: delta vs mean (contrast view)
    # --------------------------------------------------
    # --------------------------------------------------
    # Compute delta-from-mean
    # --------------------------------------------------
    pivot_delta = pivot.subtract(pivot.mean(axis=0), axis=1)

    # ==================================================
    # FIGURE 1: Raw / normalized cost
    # ==================================================
    plt.figure(figsize=(12, 5))
    plt.plot(pivot.index, pivot["theta"], marker="o", label="θ only")
    plt.plot(pivot.index, pivot["rms"],   marker="o", label="RMS only")
    plt.plot(pivot.index, pivot["both"],  marker="o", label="θ + RMS")

    plt.axhline(0, color="gray", lw=1, linestyle="--")

    if is_normalized:
        ylabel = "Relative AA cost (fraction of total SB cost)"
        suffix = "normalized"
    else:
        ylabel = "SB-attributed AA cost"
        suffix = "raw"

    plt.ylabel(ylabel)
    plt.xlabel("Amino acid (sorted by θ-only cost)")
    plt.title("Amino-acid–specific chirality-repair cost")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_out = out.with_name(out.stem + "_raw.png")
    plt.savefig(raw_out, dpi=200)
    plt.close()
    print(f"[OK] wrote {raw_out}")

    # ==================================================
    # FIGURE 2: Delta vs mean
    # ==================================================
    plt.figure(figsize=(12, 4))
    plt.plot(pivot_delta.index, pivot_delta["theta"], marker="o", label="θ only")
    plt.plot(pivot_delta.index, pivot_delta["rms"],   marker="o", label="RMS only")
    plt.plot(pivot_delta.index, pivot_delta["both"],  marker="o", label="θ + RMS")

    plt.axhline(0, color="black", lw=1)
    plt.ylabel("Δ cost vs mean")
    plt.xlabel("Amino acid (sorted by θ-only cost)")
    plt.title("Amino-acid cost deviation from global mean")
    plt.legend()
    plt.tight_layout()

    delta_out = out.with_name(out.stem + "_delta.png")
    plt.savefig(delta_out, dpi=200)
    plt.close()
    print(f"[OK] wrote {delta_out}")
    
if __name__ == "__main__":
    main()
