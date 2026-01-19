#!/usr/bin/env python3
"""Analyze sparse 'controller' amino acids as variance drivers (Prediction A).

LOCKED DESIGN (Option A):
  1) Assemble per-run table from:
       - 04_sb/output/*/summary.json (transport_cost + run metadata)
       - 05_aa_local/{theta,rms,both}/aa_cost_per_run.tsv (AA cost decomposition)
  2) Train/test split (70/30) stratified by feature_mode.
  3) Define controllers S_k on TRAIN: top-k AAs by mean absolute cost.
  4) Evaluate on TEST:
       - controller share s_r(k) = sum_{a in S_k} c_{r,a} / transport_cost
       - variance leverage R^2_ctrl(k) = corr(sum_{a in S_k} c_{r,a}, transport_cost)^2
         and R^2_non(k) = corr(transport_cost - controller_cost, transport_cost)^2
  5) Null model: random AA sets (size k), n=1000, compute same metrics.
  6) Report overall + per-feature_mode results; write TSVs + PNG plots.

Run from repo root, e.g.:
  python analyze_controller_variance.py \
    --sb-root 04_sb/output \
    --aa-root 05_aa_local \
    --out-dir 06_plots/output/controller_variance

Notes:
- This script intentionally uses AA absolute cost (aa_cost_per_run.tsv), not
  frequency-normalized tables.
- It performs a QC check: sum(AA costs) should match summary transport_cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_MODES = ("theta", "rms", "both")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def discover_sb_runs(sb_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sb_root.glob("*/summary.json"):
        run_dir = summary_path.parent
        run_id = run_dir.name
        js = _read_json(summary_path)
        rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "feature_mode": str(js.get("feature_mode", "")),
                "f0": _safe_float(js.get("f0")),
                "epsilon": _safe_float(js.get("epsilon")),
                "transport_cost": _safe_float(js.get("transport_cost")),
                "entropy": _safe_float(js.get("entropy")),
                "L0": _safe_float(js.get("L0")),
                "L1": _safe_float(js.get("L1")),
                "chirality0": js.get("chirality0"),
                "chirality1": js.get("chirality1"),
                "counterfactual": js.get("counterfactual"),
                "n_contexts": _safe_float(js.get("n_contexts")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No SB summaries found under: {sb_root}")

    # infer seed from run_id suffix '_s<seed>' if present
    def infer_seed(run_id: str) -> Optional[int]:
        # examples: sb_theta_L20_to_L12_s3
        if "_s" in run_id:
            try:
                return int(run_id.split("_s")[-1])
            except Exception:
                return None
        return None

    df["seed"] = df["run_id"].map(infer_seed)

    # basic validation
    bad_modes = sorted(set(df["feature_mode"]) - set(FEATURE_MODES) - {""})
    if bad_modes:
        print(f"[WARN] Unexpected feature_mode values in summaries: {bad_modes}")

    return df


def read_aa_cost_table(path: Path) -> pd.DataFrame:
    """Read aa_cost_per_run.tsv in a tolerant way.

    Expected shapes:
      - columns include run_id and AA columns
      - OR first column is run_id
      - OR run_id is index
    """
    df = pd.read_csv(path, sep="\t")

    # normalize run_id column
    if "run_id" in df.columns:
        pass
    elif "run" in df.columns:
        df = df.rename(columns={"run": "run_id"})
    else:
        # assume first column is run_id if it's not an AA letter
        first = df.columns[0]
        if first.lower() in ("run_id", "run"):
            df = df.rename(columns={first: "run_id"})
        else:
            # if the first column looks like run ids and others are AA letters
            df = df.rename(columns={first: "run_id"})

    # ensure run_id is str
    df["run_id"] = df["run_id"].astype(str)

    # keep AA columns that are single-letter amino acid codes
    aa_cols = [c for c in df.columns if len(c) == 1 and c.isalpha() and c.isupper()]
    if not aa_cols:
        raise ValueError(f"No AA columns detected in {path}. Columns: {list(df.columns)}")

    keep_cols = ["run_id"] + aa_cols
    return df[keep_cols].copy()


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (is_train, is_test) with stratification."""
    is_test = np.zeros(len(df), dtype=bool)

    for _, idx in df.groupby(stratify_col).indices.items():
        idx = np.array(list(idx), dtype=int)
        if len(idx) == 1:
            # put singletons in train by default
            continue
        n_test = max(1, int(round(len(idx) * test_frac)))
        pick = rng.choice(idx, size=n_test, replace=False)
        is_test[pick] = True

    is_train = ~is_test
    return is_train, is_test


def corr_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def compute_metrics_for_set(
    df_test: pd.DataFrame,
    aa_cols: List[str],
    controller_set: Sequence[str],
) -> Dict[str, float]:
    T = df_test["transport_cost"].to_numpy(float)
    C = df_test[list(controller_set)].sum(axis=1).to_numpy(float)
    N = (df_test[aa_cols].sum(axis=1).to_numpy(float) - C)

    share = np.divide(C, T, out=np.full_like(C, np.nan), where=(T != 0))
    share_mean = float(np.nanmean(share))
    share_std = float(np.nanstd(share))
    share_cv = float(share_std / share_mean) if share_mean and not np.isnan(share_mean) else float("nan")

    r2_ctrl = corr_r2(C, T)
    r2_non = corr_r2(N, T)

    return {
        "share_mean": share_mean,
        "share_std": share_std,
        "share_cv": share_cv,
        "r2_ctrl": r2_ctrl,
        "r2_non": r2_non,
    }


def plot_r2_vs_k(
    out_png: Path,
    k_values: List[int],
    r2_obs: List[float],
    null_r2: Dict[int, np.ndarray],
    title: str,
):
    plt.figure(figsize=(7.5, 4.8))
    # Null bands (5-95)
    ks = []
    lo = []
    hi = []
    med = []
    for k in k_values:
        vals = null_r2[k]
        ks.append(k)
        lo.append(np.nanpercentile(vals, 5))
        hi.append(np.nanpercentile(vals, 95))
        med.append(np.nanpercentile(vals, 50))

    plt.fill_between(ks, lo, hi, alpha=0.25, label="null 5–95%")
    plt.plot(ks, med, marker="o", linewidth=1.2, label="null median")
    plt.plot(k_values, r2_obs, marker="o", linewidth=2.0, label="observed controllers")

    plt.xlabel("k (size of controller set)")
    plt.ylabel("R²: corr(controller_cost, total_cost)²")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_share_boxplots(
    out_png: Path,
    df_test: pd.DataFrame,
    k: int,
    controller_set: Sequence[str],
    aa_cols: List[str],
    title: str,
):
    # one box per feature_mode
    data = []
    labels = []
    for mode in FEATURE_MODES:
        sub = df_test[df_test["feature_mode"] == mode]
        if sub.empty:
            continue
        T = sub["transport_cost"].to_numpy(float)
        C = sub[list(controller_set)].sum(axis=1).to_numpy(float)
        share = np.divide(C, T, out=np.full_like(C, np.nan), where=(T != 0))
        data.append(share)
        labels.append(mode)

    if not data:
        return

    plt.figure(figsize=(7.0, 4.8))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(f"Controller share s_r (k={k})")
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_null_histograms(
    out_png: Path,
    k: int,
    null_vals: np.ndarray,
    observed: float,
    title: str,
):
    plt.figure(figsize=(7.0, 4.8))
    plt.hist(null_vals[~np.isnan(null_vals)], bins=30, alpha=0.8)
    plt.axvline(observed, linewidth=2.0)
    plt.xlabel("Null R²")
    plt.ylabel("Count")
    plt.title(title + f" (k={k})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb-root", type=Path, default=Path("04_sb/output"))
    ap.add_argument("--aa-root", type=Path, default=Path("05_aa_local"))
    ap.add_argument("--out-dir", type=Path, default=Path("06_plots/output/controller_variance"))

    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--split-seed", type=int, default=20260117)

    ap.add_argument("--k", type=str, default="1,2,3,5,8", help="comma-separated k values")
    ap.add_argument("--n-null", type=int, default=1000)

    ap.add_argument("--include-frequency-matched-null", action="store_true")
    ap.add_argument("--aa-counts", type=Path, default=None,
                    help="Optional AA counts TSV (e.g., 05_aa_local/aa_counts_from_boxes.tsv)")

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover SB runs ---
    runs = discover_sb_runs(args.sb_root)

    # --- Load AA cost tables per feature mode ---
    aa_tables = []
    for mode in FEATURE_MODES:
        p = args.aa_root / mode / "aa_cost_per_run.tsv"
        if not p.exists():
            raise SystemExit(f"Missing AA cost table: {p}")
        t = read_aa_cost_table(p)
        t["feature_mode"] = mode
        aa_tables.append(t)

    aa_cost = pd.concat(aa_tables, ignore_index=True)

    # Merge
    df = runs.merge(aa_cost, on=["run_id", "feature_mode"], how="inner")
    if df.empty:
        raise SystemExit(
            "After merging SB summaries with AA costs, no rows remain. "
            "Check that run_id naming matches between 04_sb/output and 05_aa_local/*/aa_cost_per_run.tsv"
        )

    # AA columns
    aa_cols = [c for c in df.columns if len(c) == 1 and c.isalpha() and c.isupper()]
    aa_cols = sorted(aa_cols)

    # QC: AA sum vs transport_cost
    df["aa_cost_sum"] = df[aa_cols].sum(axis=1)
    df["qc_rel_error"] = np.where(
        df["transport_cost"].to_numpy(float) != 0,
        (df["aa_cost_sum"] - df["transport_cost"]) / df["transport_cost"],
        np.nan,
    )

    # Write merged table
    df.to_csv(out_dir / "run_table_merged.tsv", sep="\t", index=False)

    # --- Split ---
    rng = np.random.default_rng(args.split_seed)
    is_train, is_test = stratified_split(df, "feature_mode", args.test_frac, rng)
    df["split"] = np.where(is_test, "test", "train")
    df.to_csv(out_dir / "run_table_with_split.tsv", sep="\t", index=False)

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    # --- Controller selection (train only) ---
    mean_cost = df_train[aa_cols].mean(axis=0).sort_values(ascending=False)

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    controller_sets = {}
    controller_rows = []
    for k in k_values:
        S = list(mean_cost.index[:k])
        controller_sets[k] = S
        controller_rows.append({"k": k, "controllers": ",".join(S)})

    pd.DataFrame(controller_rows).to_csv(out_dir / "controller_sets.tsv", sep="\t", index=False)
    mean_cost.reset_index().rename(columns={"index": "aa", 0: "mean_cost_train"}).to_csv(
        out_dir / "aa_rank_by_train_mean_cost.tsv", sep="\t", index=False
    )

    # --- Observed metrics on TEST ---
    obs_rows = []
    for k in k_values:
        S = controller_sets[k]

        # overall
        m_over = compute_metrics_for_set(df_test, aa_cols, S)
        obs_rows.append({"scope": "overall", "k": k, **m_over})

        # per mode
        for mode in FEATURE_MODES:
            sub = df_test[df_test["feature_mode"] == mode]
            if len(sub) < 2:
                continue
            m = compute_metrics_for_set(sub, aa_cols, S)
            obs_rows.append({"scope": f"mode:{mode}", "k": k, **m})

    obs_df = pd.DataFrame(obs_rows)
    obs_df.to_csv(out_dir / "observed_metrics_test.tsv", sep="\t", index=False)

    # --- Null metrics (random AA sets) ---
    null_rows = []
    null_r2_by_k = {k: [] for k in k_values}

    aa_list = aa_cols.copy()

    # optional: frequency-matched null sampling
    freq_bins = None
    if args.include_frequency_matched_null:
        if args.aa_counts is None:
            # try common path
            guess = args.aa_root / "aa_counts_from_boxes.tsv"
            if guess.exists():
                args.aa_counts = guess
        if args.aa_counts is None or not args.aa_counts.exists():
            raise SystemExit("--include-frequency-matched-null requires --aa-counts (or 05_aa_local/aa_counts_from_boxes.tsv)")

        counts = pd.read_csv(args.aa_counts, sep="\t")
        if "aa" not in counts.columns:
            # tolerate alternative headers
            if "AA" in counts.columns:
                counts = counts.rename(columns={"AA": "aa"})
            else:
                counts = counts.rename(columns={counts.columns[0]: "aa"})
        if "count" not in counts.columns:
            # tolerate alternative
            for cand in ("counts", "n", "N"):
                if cand in counts.columns:
                    counts = counts.rename(columns={cand: "count"})
                    break
        counts = counts[["aa", "count"]].copy()
        counts["aa"] = counts["aa"].astype(str)
        counts = counts[counts["aa"].isin(aa_list)]

        # create 3 abundance bins by tertiles
        qs = np.quantile(counts["count"].to_numpy(float), [1/3, 2/3])
        def bin_of(c):
            if c <= qs[0]:
                return "low"
            if c <= qs[1]:
                return "mid"
            return "high"
        counts["freq_bin"] = counts["count"].map(bin_of)
        freq_bins = {b: counts[counts["freq_bin"] == b]["aa"].tolist() for b in ("low", "mid", "high")}

        # also record controller bin signature per k (to match)
        controller_bin_sig = {}
        for k in k_values:
            S = controller_sets[k]
            sig = []
            for a in S:
                b = counts.loc[counts["aa"] == a, "freq_bin"].iloc[0]
                sig.append(b)
            controller_bin_sig[k] = sig
    else:
        controller_bin_sig = None

    for k in k_values:
        S_obs = controller_sets[k]
        m_obs = compute_metrics_for_set(df_test, aa_cols, S_obs)
        obs_r2 = m_obs["r2_ctrl"]

        for i in range(args.n_null):
            if freq_bins is None:
                S = rng.choice(aa_list, size=k, replace=False).tolist()
            else:
                # sample with the same bin signature as observed controllers
                sig = controller_bin_sig[k]
                chosen = []
                used = set()
                for b in sig:
                    pool = [a for a in freq_bins[b] if a not in used]
                    if not pool:
                        # fallback to global if bin is exhausted
                        pool = [a for a in aa_list if a not in used]
                    pick = rng.choice(pool, size=1, replace=False)[0]
                    chosen.append(pick)
                    used.add(pick)
                S = chosen

            m = compute_metrics_for_set(df_test, aa_cols, S)
            null_rows.append({"k": k, "i": i, "share_mean": m["share_mean"], "share_cv": m["share_cv"], "r2_ctrl": m["r2_ctrl"]})
            null_r2_by_k[k].append(m["r2_ctrl"])

        null_r2_by_k[k] = np.array(null_r2_by_k[k], dtype=float)

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(out_dir / "null_metrics_random_sets.tsv", sep="\t", index=False)

    # --- Empirical p-values for R2 ---
    p_rows = []
    for k in k_values:
        obs_r2 = obs_df[(obs_df["scope"] == "overall") & (obs_df["k"] == k)]["r2_ctrl"].iloc[0]
        null_vals = null_r2_by_k[k]
        # p = fraction of null >= observed
        p = float(np.mean(null_vals >= obs_r2))
        z = float((obs_r2 - np.nanmean(null_vals)) / (np.nanstd(null_vals) + 1e-12))
        p_rows.append({"k": k, "obs_r2": obs_r2, "null_mean": float(np.nanmean(null_vals)), "null_std": float(np.nanstd(null_vals)), "p_empirical": p, "z": z})

    pd.DataFrame(p_rows).to_csv(out_dir / "r2_empirical_pvalues.tsv", sep="\t", index=False)

    # --- Plots ---
    r2_obs = [obs_df[(obs_df["scope"] == "overall") & (obs_df["k"] == k)]["r2_ctrl"].iloc[0] for k in k_values]

    plot_r2_vs_k(
        out_dir / "r2_vs_k.png",
        k_values,
        r2_obs,
        null_r2_by_k,
        title="Prediction A: controller set explains between-run variance (TEST only)",
    )

    # boxplots for a representative k (use 5 if present else max)
    k_box = 5 if 5 in k_values else max(k_values)
    plot_share_boxplots(
        out_dir / "controller_share_boxplots.png",
        df_test,
        k_box,
        controller_sets[k_box],
        aa_cols,
        title="Controller share s_r by feature mode (TEST only)",
    )

    # null histogram for representative k
    plot_null_histograms(
        out_dir / "null_r2_hist_k.png",
        k_box,
        null_r2_by_k[k_box],
        r2_obs[k_values.index(k_box)],
        title="Null distribution of R² (random AA sets)",
    )

    # --- README ---
    readme = out_dir / "README.md"
    readme.write_text(
        """# Controller variance test (Prediction A, Option A)

This folder contains outputs from the locked experiment:

- Build per-run dataset by joining:
  - `04_sb/output/*/summary.json` (total transport cost + metadata)
  - `05_aa_local/{theta,rms,both}/aa_cost_per_run.tsv` (AA cost decomposition)

- Stratified train/test split by `feature_mode` (70/30).
- Define controller sets `S_k` using TRAIN only: top-k amino acids by mean absolute cost.
- Evaluate on TEST only:
  - Controller share per run: `s_r = sum_{a in S_k} c_{r,a} / transport_cost`
  - Variance leverage: `R²_ctrl = corr(sum_{a in S_k} c_{r,a}, transport_cost)^2`
  - Compare to `R²_non` from the complement set.
- Null model: 1000 random AA sets per k.

## Key files
- `run_table_merged.tsv`: merged run table (before split) + QC columns.
- `run_table_with_split.tsv`: same table with `split=train|test`.
- `controller_sets.tsv`: controllers chosen from TRAIN for each k.
- `observed_metrics_test.tsv`: observed share/CV/R² metrics on TEST.
- `null_metrics_random_sets.tsv`: null metrics for random AA sets.
- `r2_empirical_pvalues.tsv`: empirical p-values + z-scores for observed R².

## Plots
- `r2_vs_k.png`: observed R² vs k with null 5–95% band.
- `controller_share_boxplots.png`: distribution of controller share across test runs by feature mode.
- `null_r2_hist_k.png`: histogram of null R² for representative k with observed marked.
""",
        encoding="utf-8",
    )

    # --- Print quick summary ---
    print("\n[OK] Wrote outputs to:", out_dir)
    print("\n[QC] Relative error (aa_cost_sum vs transport_cost):")
    print(df["qc_rel_error"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    print("\n[TEST] Observed metrics (overall):")
    print(obs_df[obs_df["scope"] == "overall"].sort_values("k"))
    print("\n[P] Empirical p-values for R²:")
    print(pd.DataFrame(p_rows).sort_values("k"))


if __name__ == "__main__":
    main()
