import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

AA_ORDER = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True,
                    help="aa_counts_from_boxes.tsv")
    ap.add_argument("--aa-cost", required=True,
                    help="aa_cost_per_run.tsv (for one feature mode)")
    ap.add_argument("--out", required=True,
                    help="output PNG")
    args = ap.parse_args()

    # ----------------------------
    # Load AA counts
    # ----------------------------
    counts = pd.read_csv(args.counts, sep="\t")
    counts = counts.set_index("aa")

    # ----------------------------
    # Load AA costs and average across runs
    # ----------------------------
    cost = pd.read_csv(args.aa_cost, sep="\t")

    cost_mean = (
        cost
        .groupby("aa", as_index=False)
        .agg(mean_cost=("mean_cost", "mean"))
        .set_index("aa")
    )

    # ----------------------------
    # Merge
    # ----------------------------
    df = counts.join(cost_mean, how="inner")

    # ----------------------------
    # Scatter plot
    # ----------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(df["fraction"], df["mean_cost"], s=60)

    for aa, row in df.iterrows():
        plt.text(row["fraction"], row["mean_cost"], aa,
                 fontsize=9, ha="right", va="bottom")

    plt.xlabel("Residue frequency (fraction)")
    plt.ylabel("Mean SB-attributed AA cost")
    plt.title("Amino-acid frequency vs chirality-repair cost")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()

