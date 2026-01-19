#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize AA cost uncertainty across SB runs (CI-includes-zero)"
    )
    p.add_argument(
        "--aa-cost-per-run",
        required=True,
        help="Path to aa_cost_per_run.tsv"
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval level (default 0.95)"
    )
    p.add_argument(
        "--out",
        default="aa_cost_ci.tsv",
        help="Output TSV"
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.aa_cost_per_run, sep="\t")

    required = {"aa", "mean_cost", "run"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    lo_q = (1.0 - args.ci) / 2.0
    hi_q = 1.0 - lo_q

    summary = (
        df.groupby("aa", as_index=False)
        .agg(
            mean_cost=("mean_cost", "mean"),
            std_cost=("mean_cost", "std"),
            n_runs=("mean_cost", "count"),
            lo=("mean_cost", lambda x: np.quantile(x, lo_q)),
            hi=("mean_cost", lambda x: np.quantile(x, hi_q)),
        )
    )

    global_mean = df["mean_cost"].mean()
    summary["ci_includes_global_mean"] = (
        (summary["lo"] <= global_mean) & (summary["hi"] >= global_mean)
    )

    summary = summary.sort_values("mean_cost")

    summary.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
