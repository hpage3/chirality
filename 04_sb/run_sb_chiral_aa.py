import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# I/O helpers (matches your ids_extract format)
# ----------------------------
class Context:
    def __init__(self, pdb_id: str, chain: str, segment_start: int):
        self.pdb_id = pdb_id.lower()
        self.chain = chain
        self.segment_start = int(segment_start)

    @property
    def prefix(self) -> str:
        return f"{self.pdb_id}_{self.chain}_start{self.segment_start}"


def load_ids_extract(path: Path) -> list[Context]:
    contexts: list[Context] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Bad line in {path}: {raw!r} (expected 3 columns)")
        pdb_id, chain, start = parts
        contexts.append(Context(pdb_id, chain, int(start)))
    return contexts


# ----------------------------
# Fingerprint construction
# x_L = [theta_bin_means (10), chi_bin_means (10)] in R^20
# ----------------------------
def make_fp_vector(
    fp_path: Path,
    n_bins: int,
    chirality_mode: str,
    mixed_frac: float,
    rng: np.random.Generator | None,
    feature_mode: str,
    theta_randomize: bool = False,
    rng_cf: np.random.Generator | None = None,
) -> np.ndarray:

    df = pd.read_csv(fp_path)

    if "theta_pp_deg" not in df.columns:
        raise KeyError(f"{fp_path} missing 'theta_pp_deg'")
    if feature_mode in ("rms", "both") and "box_rms" not in df.columns:
        raise KeyError(f"{fp_path} missing 'box_rms'")

    theta = df["theta_pp_deg"].to_numpy(dtype=float)

    # Chirality perturbation applies ONLY to theta
    if chirality_mode == "mirror":
        theta = -theta
    elif chirality_mode == "mixed":
        if rng is None:
            raise ValueError("rng must be provided for mixed chirality")
        mask = rng.random(len(theta)) < mixed_frac
        theta = theta.copy()
        theta[mask] *= -1
    elif chirality_mode == "L":
        pass
    else:
        raise ValueError(f"Unknown chirality_mode: {chirality_mode}")

    # --- Counterfactual: randomize theta signs ---
    if theta_randomize:
        if rng_cf is None:
            raise ValueError("theta_randomize=True requires rng_cf")
        flips = rng_cf.choice([-1.0, 1.0], size=len(theta))
        theta = theta * flips


    # Chirality coherence channel
    chi = np.sign(theta)
    chi[chi == 0] = 1.0

    # RMS channel (chirality-even)
    if feature_mode in ("rms", "both"):
        rms = df["box_rms"].to_numpy(dtype=float)

    n = len(theta)
    edges = np.linspace(0, n, n_bins + 1).astype(int)

    theta_bins = np.full(n_bins, np.nan)
    chi_bins   = np.full(n_bins, np.nan)
    rms_bins   = np.full(n_bins, np.nan)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if lo == hi:
            continue
        theta_bins[i] = np.nanmean(theta[lo:hi])
        chi_bins[i]   = np.nanmean(chi[lo:hi])
        if feature_mode in ("rms", "both"):
            rms_bins[i] = np.nanmean(rms[lo:hi])

    # --- Feature selection (tau-style) ---
    if feature_mode == "theta":
        return np.concatenate([theta_bins, chi_bins])        # 20D
    elif feature_mode == "rms":
        return rms_bins.copy()                                # 10D
    elif feature_mode == "both":
        return np.concatenate([theta_bins, chi_bins, rms_bins])  # 30D
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_ensemble(
    contexts,
    fingerprint_dir,
    L,
    n_bins,
    chirality_mode,
    mixed_frac,
    seed,
    feature_mode,
    theta_randomize: bool = False,
    rng_cf: np.random.Generator | None = None,
):


    rng = None
    if chirality_mode == "mixed":
        if seed is None:
            raise ValueError("seed is required for mixed chirality")
        rng = np.random.default_rng(seed)

    X = []
    labels = []
    print("Fingerprint Directory: ", fingerprint_dir)
    for ctx in contexts:
        fp_file = fingerprint_dir / f"{ctx.prefix}_L{L}_fingerprint.csv"
        if not fp_file.exists():
            continue
        x = make_fp_vector(
            fp_file,
            n_bins,
            chirality_mode,
            mixed_frac,
            rng,
            feature_mode,
            theta_randomize=theta_randomize,
            rng_cf=rng_cf,
        )


        if np.all(np.isnan(x)):
            continue
        X.append(x)
        labels.append(ctx.prefix)

    if len(X) < 2:
        raise RuntimeError(f"Not enough fingerprints found for L={L}")
    return np.vstack(X), labels


# ----------------------------
# Sinkhorn (discrete Schrödinger problem / entropic OT)
# ----------------------------
def rms_cost_matrix(X0, X1, rms_slice):
    """
    Compute pairwise squared distance using only RMS bins.
    """
    A = X0[:, rms_slice]
    B = X1[:, rms_slice]

    A0 = np.nan_to_num(A, nan=0.0)
    B0 = np.nan_to_num(B, nan=0.0)

    Am = ~np.isnan(A)
    Bm = ~np.isnan(B)

    diff = A0[:, None, :] - B0[None, :, :]
    sq = diff * diff

    overlap = (Am[:, None, :] & Bm[None, :, :]).sum(axis=2).astype(float)
    overlap[overlap < 1.0] = 1.0

    return sq.sum(axis=2) / overlap

def pairwise_sqeuclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # handles NaNs by masking per-feature (simple but OK for our use)
    # Convert NaNs to 0 and weight by observed dims count
    A0 = np.nan_to_num(A, nan=0.0)
    B0 = np.nan_to_num(B, nan=0.0)
    # mask counts
    Am = ~np.isnan(A)
    Bm = ~np.isnan(B)
    # For each (i,j), use dims observed in both
    # Compute squared distances and divide by overlap count to normalize
    # This keeps scale stable when some bins are NaN
    # d_ij = sum_k (A-B)^2 / max(1, overlap_k)
    diff = A0[:, None, :] - B0[None, :, :]
    sq = diff * diff
    overlap = (Am[:, None, :] & Bm[None, :, :]).sum(axis=2).astype(float)
    overlap[overlap < 1.0] = 1.0
    return sq.sum(axis=2) / overlap


def sinkhorn(a, b, K, n_iter=2000, tol=1e-9):
    # a, b are probability vectors; K is Gibbs kernel exp(-C/eps)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for it in range(n_iter):
        u_prev = u
        Ku = K @ v
        Ku[Ku == 0] = 1e-300
        u = a / Ku

        Kv = K.T @ u
        Kv[Kv == 0] = 1e-300
        v = b / Kv

        if it % 50 == 0:
            if np.linalg.norm(u - u_prev, ord=1) < tol:
                break

    P = (u[:, None] * K) * v[None, :]
    return P


def barycentric_interpolation(X0, X1, P, t: float) -> np.ndarray:
    """
    Produce 'bridge samples' at time t by coupling pairs and interpolating:
        x_t = (1-t)*x0 + t*x1
    We sample index pairs proportional to P.
    """
    n0, n1 = P.shape
    flat = P.ravel()
    flat = flat / flat.sum()
    m = min(n0, n1, 4000)  # cap for speed/plotting

    idx = np.random.choice(n0 * n1, size=m, replace=True, p=flat)
    i = idx // n1
    j = idx % n1

    Xt = (1 - t) * X0[i] + t * X1[j]
    return Xt


def pca2(X: np.ndarray):
    # Remove columns with NaNs everywhere or zero variance
    mask = ~np.all(np.isnan(X), axis=0)
    X = X[:, mask]

    # Replace remaining NaNs with column means
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    # Remove zero-variance columns
    var = X.var(axis=0)
    keep = var > 1e-8
    X = X[:, keep]

    # Center
    X0 = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    comps = Vt[:2].T
    Z = X0 @ comps
    return Z, comps, X.mean(axis=0)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Discrete Schrödinger bridge on θ-pp+chirality fingerprints")

    p.add_argument("--ids", default="ids_extract.txt")
    p.add_argument("--fingerprint-dir", default="fingerprint")

    p.add_argument("--n-bins", type=int, default=10)

    # Endpoint definitions (grounded in your crossover)
    p.add_argument("--L0", type=int, default=20)
    p.add_argument("--f0", type=float, default=0.6)
    p.add_argument("--L1", type=int, default=12)
    p.add_argument("--f1", type=float, default=0.0)

    p.add_argument("--chirality0", choices=["L", "mirror", "mixed"], default="mixed")
    p.add_argument("--chirality1", choices=["L", "mirror", "mixed"], default="mixed")

    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--seed1", type=int, default=42)

    # Sinkhorn regularization (epsilon)
    p.add_argument("--epsilon", type=float, default=1.0, help="Entropic regularization strength")

    p.add_argument("--out-prefix", default="sb_chiral")
    
    p.add_argument(
        "--feature-mode",
        choices=["theta", "rms", "both"],
        default="theta",
        help="Which fingerprint features to include in SB state vector"
    )

    p.add_argument(
        "--counterfactual",
        choices=["none", "rms_shuffle", "theta_randomize"],
        default="none",
        help="Apply counterfactual perturbation to test constraint vs correlation"
    )
    p.add_argument(
        "--cf_seed",
        type=int,
        default=123,
        help="Random seed for counterfactual perturbations"
    )
    p.add_argument(
        "--theta_randomize_where",
        choices=["p0", "p1", "both"],
        default="p0",
        help="Which endpoint ensemble to theta-randomize (used when counterfactual=theta_randomize)"
    )
    p.add_argument(
        "--out-dir",
        default="output",
        help="Directory to write SB outputs (one subfolder per run)"
    )


    return p.parse_args()

def load_contexts_from_fingerprints(fp_dir: Path):
    """
    Load contexts directly from fingerprint filenames.

    Expects files like:
        <prefix>_L20_fingerprint.csv
        <prefix>_L12_fingerprint.csv
    """
    prefixes = set()

    for fp in fp_dir.glob("*_L*_fingerprint.csv"):
        name = fp.name
        prefix = name.split("_L")[0]
        prefixes.add(prefix)

    if not prefixes:
        raise RuntimeError(f"No fingerprint files found in {fp_dir}")

    class Context:
        def __init__(self, prefix):
            self.prefix = prefix

    return [Context(p) for p in sorted(prefixes)]

def main():
    args = parse_args()
    ids_path = Path(args.ids)
    fp_dir = Path(args.fingerprint_dir)
    summary = {}

    
    rng_cf = np.random.default_rng(args.cf_seed)

    contexts = load_contexts_from_fingerprints(fp_dir)
    print(f"Loaded {len(contexts)} contexts from fingerprint directory")

    theta_rand_p0 = (
        args.counterfactual == "theta_randomize"
        and args.theta_randomize_where in ("p0", "both")
    )
    theta_rand_p1 = (
        args.counterfactual == "theta_randomize"
        and args.theta_randomize_where in ("p1", "both")
    )

    if args.counterfactual == "theta_randomize":
        print(
            f"⚠️ Counterfactual applied: θ signs randomized in "
            f"{args.theta_randomize_where}"
        )

    # Build endpoint ensembles
    X0, lab0 = build_ensemble(
        contexts, fp_dir,
        args.L0, args.n_bins,
        args.chirality0, args.f0, args.seed0,
        args.feature_mode,
        theta_randomize=theta_rand_p0,
        rng_cf=rng_cf,
    )

    X1, lab1 = build_ensemble(
        contexts, fp_dir,
        args.L1, args.n_bins,
        args.chirality1, args.f1, args.seed1,
        args.feature_mode,
        theta_randomize=theta_rand_p1,
        rng_cf=rng_cf,
    )

    rng_cf = np.random.default_rng(args.cf_seed)

    if args.counterfactual == "rms_shuffle":
        if args.feature_mode not in ("both", "rms"):
            raise ValueError("counterfactual=rms_shuffle requires --feature_mode rms or both")

        # Identify RMS slice (depends on feature_mode)
        if args.feature_mode == "rms":
            rms_slice = slice(0, args.n_bins)  # 10D
        else:
            # both: [theta(10), chi(10), rms(10)]
            rms_slice = slice(2 * args.n_bins, 3 * args.n_bins)

        perm0 = rng_cf.permutation(X0.shape[0])
        perm1 = rng_cf.permutation(X1.shape[0])

        X0 = X0.copy()
        X1 = X1.copy()
        X0[:, rms_slice] = X0[perm0, rms_slice]
        X1[:, rms_slice] = X1[perm1, rms_slice]

        print("⚠️ Counterfactual applied: RMS bins shuffled across contexts (p0 and p1)")


    print(f"p0: n={X0.shape[0]}  (L0={args.L0}, mode={args.chirality0}, f0={args.f0})")
    print(f"p1: n={X1.shape[0]}  (L1={args.L1}, mode={args.chirality1}, f1={args.f1})")

    # Normalize features so theta channel doesn't dwarf chi channel
    # theta bins in degrees; chi bins in [-1,1]
    # Use robust scaling per feature (MAD)
    def robust_scale(X):
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        mad[mad < 1e-6] = 1.0
        return (X - med) / mad, med, mad

    X0s, med0, mad0 = robust_scale(X0)
    X1s = (X1 - med0) / mad0  # scale to p0 stats for comparability

    # Cost matrix
    C = pairwise_sqeuclidean(X0s, X1s)

    # --- Precision fix: ensure RMS contributes in feature_mode=both ---
    if args.feature_mode == "both":
        rms_slice = slice(2 * args.n_bins, 3 * args.n_bins)  # RMS bins

        C_rms = rms_cost_matrix(X0s, X1s, rms_slice)

        # Scale RMS cost to comparable magnitude (robust)
        s_theta = np.median(C[C > 0])
        s_rms = np.median(C_rms[C_rms > 0])
        if s_rms > 0:
            C = C + (s_theta / s_rms) * C_rms

    eps = float(args.epsilon)
    K = np.exp(-C / eps)

    a = np.full(X0.shape[0], 1.0 / X0.shape[0])
    b = np.full(X1.shape[0], 1.0 / X1.shape[0])

    P = sinkhorn(a, b, K, n_iter=3000, tol=1e-10)

    # Summary metrics
    transport_cost = float(np.sum(P * C))
    entropy = float(-np.sum(P * np.log(P + 1e-300)))  # coupling entropy
    print(f"SB discrete coupling:  cost={transport_cost:.4f}  entropy={entropy:.2f}")

    # --------------------------------------------------
    # Decompose transport cost by feature / bin
    # --------------------------------------------------
    # X0s, X1s : (n0, d), (n1, d)
    # P        : (n0, n1)
    # d        : number of feature dimensions

    n0, d = X0s.shape
    n1 = X1s.shape[0]

    # Mask for valid (non-NaN) overlaps, matching pairwise_sqeuclidean
    A0 = np.nan_to_num(X0s, nan=0.0)
    B0 = np.nan_to_num(X1s, nan=0.0)
    Am = ~np.isnan(X0s)
    Bm = ~np.isnan(X1s)

    # overlap counts per (i,j)
    overlap = (Am[:, None, :] & Bm[None, :, :]).sum(axis=2).astype(float)
    overlap[overlap < 1.0] = 1.0

    # cost contribution per feature dimension
    cost_by_feature = np.zeros(d, dtype=float)

    for k in range(d):
        diff_k = A0[:, [k]] - B0[:, [k]].T          # (n0, n1)
        c_k = (diff_k * diff_k) / overlap
        cost_by_feature[k] = float(np.sum(P * c_k))

    # Sanity check (should match total cost)
    if args.feature_mode != "both":
        assert np.allclose(cost_by_feature.sum(), transport_cost, rtol=1e-6)


    # --------------------------------------------------
    # Aggregate feature cost → bin cost
    # --------------------------------------------------
    n_bins = args.n_bins

    if args.feature_mode == "theta":
        # [theta(10), chi(10)]
        cost_by_bin = cost_by_feature[:n_bins] + cost_by_feature[n_bins:2*n_bins]

    elif args.feature_mode == "rms":
        # [rms(10)]
        cost_by_bin = cost_by_feature[:n_bins]

    elif args.feature_mode == "both":
        # [theta(10), chi(10), rms(10)]
        cost_by_bin = (
            cost_by_feature[:n_bins] +
            cost_by_feature[n_bins:2*n_bins] +
            cost_by_feature[2*n_bins:3*n_bins]
        )

    else:
        raise ValueError("Unknown feature_mode")



    # --- Output directory handling ---
    out_root = Path(args.out_dir)
    out_run = out_root / args.out_prefix
    out_run.mkdir(parents=True, exist_ok=True)

    np.save(out_run / "coupling.npy", P)
    np.save(out_run / "C.npy", C)
    
    summary.update({
        "L0": args.L0,
        "f0": args.f0,
        "chirality0": args.chirality0,
        "L1": args.L1,
        "f1": args.f1,
        "chirality1": args.chirality1,
        "feature_mode": args.feature_mode,
        "counterfactual": args.counterfactual,
        "epsilon": args.epsilon,
        "transport_cost": transport_cost,
        "entropy": entropy,
        "n_contexts": X0.shape[0]
    })

    with open(out_run / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


    # Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))

    # Build bridge time-slices and visualize in PCA2
    rng = np.random.default_rng(0)
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Fit PCA on pooled scaled data for stable projection
    Xpool = np.vstack([X0s, X1s])
    Zpool, comps, meanpool = pca2(Xpool)

    # Project endpoints
    Z0 = (X0s - meanpool) @ comps
    Z1 = (X1s - meanpool) @ comps

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(Z0[:, 0], Z0[:, 1],
                s=12, alpha=0.45, label="p0 (heterochiral / long L)")
    plt.scatter(Z1[:, 0], Z1[:, 1],
                s=12, alpha=0.45, label="p1 (coherent / short L)")
    if args.counterfactual == "rms_shuffle":
        rms_label = "RMS shuffled across contexts (counterfactual)"
    else:
        rms_label = "real RMS preserved"

    plt.title(
        "Endpoint distributions (p0 → p1)\n"
        f"Feature space: θ + RMS ({rms_label})"
    )


    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    fname = "bridge_pca_endpoints_realRMS.png"
    if args.counterfactual == "rms_shuffle":
        fname = "bridge_pca_endpoints_rmsShuffle.png"

    plt.savefig(out_run / fname, dpi=200)

    # plt.show()

    print(f"✅ Wrote SB outputs to: {out_run}")

    # --------------------------------------------------
    # Write local cost decomposition
    # --------------------------------------------------
    df_bin = pd.DataFrame({
        "bin": np.arange(len(cost_by_bin)),
        "cost": cost_by_bin
    })
    df_bin.to_csv(out_run / "cost_by_bin.tsv", sep="\t", index=False)

    df_feat = pd.DataFrame({
        "feature_index": np.arange(len(cost_by_feature)),
        "cost": cost_by_feature
    })
    df_feat.to_csv(out_run / "cost_by_feature.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
