#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import re
from collections import defaultdict

import pandas as pd

AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
N_BINS = 10


def parse_seed(run_name: str):
    m = re.search(r"_s(\d+)$", run_name)
    return int(m.group(1)) if m else None


def load_bin_cost(run_dir: Path) -> dict:
    f = run_dir / "cost_by_bin.tsv"
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")
    df = pd.read_csv(f, sep="\t")
    return dict(zip(df["bin"], df["cost"]))


def select_runs(sb_root: Path, feature_mode: str):
    """
    Only SB runs for this mode: sb_theta_*, sb_rms_*, sb_both_*
    """
    prefix = f"sb_{feature_mode}_"
    runs = [d for d in sb_root.iterdir()
            if d.is_dir() and d.name.startswith(prefix)]
    return sorted(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_root", required=True, help="04_sb/output")
    ap.add_argument("--feature-mode", required=True, choices=["theta", "rms", "both"])
    ap.add_argument("--fingerprint_dir", required=True, help="03_fingerprints")
    ap.add_argument("--boxes_dir", required=True, help="02_boxes")
    ap.add_argument("--out_dir", required=True, help="05_aa_local/<mode>")
    ap.add_argument("--L", type=int, default=12, help="Target L for fingerprints (default 12)")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize AA costs per run so sum over AAs equals 1 (optional)")
    args = ap.parse_args()

    sb_root = Path(args.sb_root)
    fp_dir = Path(args.fingerprint_dir)
    boxes_dir = Path(args.boxes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = select_runs(sb_root, args.feature_mode)
    if not runs:
        raise RuntimeError(f"No runs found for mode={args.feature_mode} under {sb_root}")

    # Preload structure-level resources once
    # Map prefix -> (n_windows, n_planes, planes dataframe)
    structures = {}
    fp_glob = f"*_*_L{args.L}_fingerprint.csv"
    for fp_csv in sorted(fp_dir.glob(fp_glob)):
        prefix = fp_csv.name.replace(f"_L{args.L}_fingerprint.csv", "")
        box_csv = boxes_dir / f"{prefix}_boxes_normals.csv"
        if not box_csv.exists():
            continue

        fp = pd.read_csv(fp_csv)
        boxes = pd.read_csv(box_csv)
        n_windows = len(fp)
        n_planes = len(boxes)
        if n_windows == 0 or n_planes == 0:
            continue

        # Planes are indexed in file order; each plane touches resname_i and resname_j
        structures[prefix] = (n_windows, n_planes, boxes)

    if not structures:
        raise RuntimeError("No (fingerprint, boxes) pairs found. Check 03_fingerprints and 02_boxes.")

    per_run_rows = []

    for run in runs:
        seed = parse_seed(run.name)
        bin_cost = load_bin_cost(run)

        aa_cost_run = defaultdict(float)

        for prefix, (n_windows, n_planes, boxes) in structures.items():
            planes_per_window = math.ceil(n_planes / n_windows)
            windows_per_bin = math.ceil(n_windows / N_BINS)

            # window -> bin (by window index)
            def w2b(w):
                return min(w // windows_per_bin, N_BINS - 1)

            for w in range(n_windows):
                b = w2b(w)

                # distribute bin cost across windows in that bin
                cost_w = bin_cost.get(b, 0.0) / windows_per_bin

                p_lo = w * planes_per_window
                p_hi = min(p_lo + planes_per_window, n_planes)
                n_p = max(p_hi - p_lo, 1)

                cost_p = cost_w / n_p

                for _, row in boxes.iloc[p_lo:p_hi].iterrows():
                    aa_i = AA3_TO_1.get(str(row["resname_i"]).upper(), None)
                    aa_j = AA3_TO_1.get(str(row["resname_j"]).upper(), None)
                    if aa_i in AA20:
                        aa_cost_run[aa_i] += cost_p / 2.0
                    if aa_j in AA20:
                        aa_cost_run[aa_j] += cost_p / 2.0

        if args.normalize:
            total = sum(aa_cost_run.values())
            if total > 0:
                for aa in list(aa_cost_run.keys()):
                    aa_cost_run[aa] /= total

        for aa in AA20:
            per_run_rows.append({
                "run": run.name,
                "seed": seed,
                "aa": aa,
                "mean_cost": float(aa_cost_run.get(aa, 0.0))
            })

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(out_dir / "aa_cost_per_run.tsv", sep="\t", index=False)

    summary = (
        per_run_df.groupby("aa", as_index=False)
        .agg(
            mean_cost=("mean_cost", "mean"),
            std_cost=("mean_cost", "std"),
            n_runs=("mean_cost", "count")
        )
    )
    global_mean = summary["mean_cost"].mean()
    summary["delta_vs_global"] = summary["mean_cost"] - global_mean
    summary.to_csv(out_dir / "aa_cost_run_normalized.tsv", sep="\t", index=False)

    print(f"[OK] wrote {out_dir/'aa_cost_per_run.tsv'}")
    print(f"[OK] wrote {out_dir/'aa_cost_run_normalized.tsv'}")


if __name__ == "__main__":
    main()
