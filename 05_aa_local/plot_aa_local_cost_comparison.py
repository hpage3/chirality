#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


AA_CANONICAL_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    p = argparse.ArgumentParser(
        description="Amino-acid–specific chirality-repair cost comparison "
                    "(reads aa_cost_run_normalized.tsv)"
    )
    p.add_argument(
        "--base",
        required=True,
        help="Directory containing aa_cost_run_normalized.tsv"
    )
    p.add_argument(
        "--metric",
        choices=["delta", "mean"],
        default="delta",
        help="Plot delta_vs_global or mean_cost"
    )
    p.add_argument(
        "--aa-order",
        choices=["canonical", "alphabetical", "by_theta", "by_rms", "by_both"],
        default="canonical",
        help="Ordering of amino acids on x-axis"
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to save PNG instead of showing interactively"
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base)

    table = base / "aa_cost_run_normalized.tsv"
    if not table.exists():
        raise FileNotFoundError(f"Missing AA table: {table}")

    df = pd.read_csv(table, sep="\t")

    required_cols = {"aa", "feature_mode", "mean_cost", "delta_vs_global"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    metric_col = "delta_vs_global" if args.metric == "delta" else "mean_cost"

    # pivot: rows = aa, columns = feature_mode
    pivot = (
        df.pivot(index="aa", columns="feature_mode", values=metric_col)
        .reset_index()
    )

    for col in ["theta", "rms", "both"]:
        if col not in pivot.columns:
            raise RuntimeError(f"Missing feature_mode '{col}' in table")

    # ordering
    if args.aa_order == "canonical":
        pivot["aa"] = pd.Categorical(
            pivot["aa"], categories=AA_CANONICAL_ORDER, ordered=True
        )
        pivot = pivot.sort_values("aa")
    elif args.aa_order == "alphabetical":
        pivot = pivot.sort_values("aa")
    elif args.aa_order.startswith("by_"):
        key = args.aa_order.split("_")[1]
        pivot = pivot.sort_values(key, ascending=False)

    # plotting
    plt.figure(figsize=(12, 5))
    x = pivot["aa"]

    plt.plot(x, pivot["theta"], marker="o", label="θ-only")
    plt.plot(x, pivot["rms"], marker="o", label="RMS-only")
    plt.plot(x, pivot["both"], marker="o", label="θ + RMS")

    plt.axhline(0, color="gray", linestyle="--", lw=1)
    plt.xlabel("Amino acid")
    ylabel = "Δ local cost vs global mean" if args.metric == "delta" else "Mean local cost"
    plt.ylabel(ylabel)
    plt.title("Amino-acid–specific local chirality-repair cost")
    plt.legend()
    plt.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"[OK] Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
