#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Canonical AA order for readability
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot AA theta-only SB cost with CI-based near-zero coloring"
    )
    p.add_argument(
        "--aa-ci",
        required=True,
        help="aa_cost_ci.tsv (from theta-only analysis)"
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output PNG"
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.aa_ci, sep="\t")

    required = {
        "aa", "mean_cost",
        "ci_includes_global_mean"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.aa_ci}: {missing}")

    # Ensure AA ordering
    df["aa"] = df["aa"].str.strip().str.upper()
    df["aa"] = pd.Categorical(df["aa"], categories=AA_ORDER, ordered=True)
    df = df.sort_values("aa")

    # Two-color scheme
    color_near = "#bdbdbd"   # light gray
    color_non  = "#d73027"   # muted red

    colors = df["ci_includes_global_mean"].map(
        lambda x: color_near if x else color_non
    )

    plt.figure(figsize=(8, 4))

    plt.bar(
        df["aa"],
        df["mean_cost"],
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    # Labels and title (Loren-clean)
    plt.xlabel("Amino acid")
    plt.ylabel("Mean θ-only SB cost")
    plt.title("Residue-specific chirality cost (θ-only)")

    # Legend (manual, explicit)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_near, edgecolor="black",
              label="Near-zero (CI includes global mean)"),
        Patch(facecolor=color_non, edgecolor="black",
              label="Non-zero (CI excludes global mean)")
    ]
    plt.legend(handles=legend_handles, frameon=False, fontsize=9)

    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

