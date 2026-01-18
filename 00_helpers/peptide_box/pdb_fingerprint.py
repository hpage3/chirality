import os
import argparse
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np


AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}


# ------------------------------------------------------------
# Single source of truth: ids_extract.txt
# ------------------------------------------------------------
def load_chain_contexts(ids_path: Path):
    """
    Load unique (pdb_id, chain) pairs from ids_extract.txt.
    Segment start is ignored.
    """
    contexts = []
    seen = set()

    for raw in ids_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        pdb_id, chain, *_ = line.split()
        pdb_id = pdb_id.lower()

        key = (pdb_id, chain)
        if key in seen:
            continue
        seen.add(key)

        contexts.append((pdb_id, chain))

    return contexts

def bin_fingerprint(df, L):
    """
    Bin a per-residue fingerprint DataFrame to L bins.

    Expects columns:
        prefix, chain, res_i, aa_i, theta_pp_deg, box_rms
    """
    df = df.sort_values("res_i").reset_index(drop=True)
    n = len(df)

    # Assign each residue to a bin
    bin_ids = np.floor(np.linspace(0, L, n, endpoint=False)).astype(int)
    df["bin"] = bin_ids

    # Aggregate per bin
    binned = (
        df
        .groupby("bin", as_index=False)
        .agg(
            prefix=("prefix", "first"),
            chain=("chain", "first"),
            theta_pp_deg=("theta_pp_deg", "mean"),
            box_rms=("box_rms", "mean"),
            n_res=("res_i", "count"),
        )
    )

    binned["L"] = L
    return binned


def resolve_case_insensitive(directory: Path, filename: str):
    """
    Resolve a filename in directory ignoring case.
    """
    direct = directory / filename
    if direct.exists():
        return direct

    lname = filename.lower()
    for f in directory.iterdir():
        if f.name.lower() == lname:
            return f

    return None


def get_residue_map(pdb_file: Path):
    """
    Map (chain, resseq) -> one-letter AA for a full-chain PDB.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)

    mapping = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":
                    mapping[(chain.id, res.id[1])] = AA3_TO_1.get(
                        res.resname.strip().upper(), "X"
                    )
    return mapping


def process_chain(prefix, pdb_dir, angles_dir, boxes_dir, out_dir):
    """
    Build θpp + RMS fingerprint for a single chain PDB.
    prefix = pdb_id_chain (e.g. 1m14_A)
    """
    pdb_path = resolve_case_insensitive(pdb_dir, f"{prefix}.pdb")
    ang_csv  = resolve_case_insensitive(angles_dir, f"{prefix}_boxes_adjacent_angles.csv")
    box_csv  = resolve_case_insensitive(boxes_dir,  f"{prefix}_boxes_normals.csv")

    if not (pdb_path and ang_csv and box_csv):
        print(f"[skip] Missing files for {prefix}")
        return

    # θpp angles
    ang = pd.read_csv(ang_csv)
    ang["res_i"] = ang["res_i_A"]
    ang["res_j"] = ang["res_j_A"]
    ang = ang.rename(columns={"angle_signed_deg": "theta_pp_deg"})

    # Box RMS
    box = pd.read_csv(box_csv)
    rmap = {}
    for c in box.columns:
        cl = c.lower()
        if cl == "chain": rmap[c] = "chain"
        elif cl in ("res_i","resseq","i"): rmap[c] = "res_i"
        elif cl in ("res_j","j"): rmap[c] = "res_j"
        elif cl in ("rms","fit_rms","box_rms"): rmap[c] = "box_rms"

    box = box.rename(columns=rmap)
    box = box[["chain","res_i","res_j","box_rms"]].drop_duplicates()

    # Merge θpp + RMS
    ang = ang.merge(box, on=["chain","res_i","res_j"], how="left")

    # Residue identity
    aa_map = get_residue_map(pdb_path)
    ang["aa_i"] = [
        aa_map.get((c, i), "X")
        for c, i in zip(ang["chain"], ang["res_i"])
    ]

    # Per-residue fingerprint (anchor on res_i)
    out = ang[["chain","res_i","aa_i","theta_pp_deg","box_rms"]].copy()
    out.insert(0, "prefix", prefix)
    out = (
        out
        .drop_duplicates(subset=["prefix","chain","res_i"])
        .sort_values(["chain","res_i"])
    )

    out_dir.mkdir(exist_ok=True, parents=True)

    for L in (20, 12):
        binned = bin_fingerprint(out, L)
        out_path = out_dir / f"{prefix}_L{L}_fingerprint.csv"
        binned.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build θpp + RMS fingerprints (one per chain PDB)"
    )
    ap.add_argument("--ids", default="ids_extract.txt")
    ap.add_argument("--pdb-dir", default="windows_pdb")
    ap.add_argument("--angles-dir", default="output")
    ap.add_argument("--boxes-dir",  default="output")
    ap.add_argument("--outdir",     default="fingerprints")
    args = ap.parse_args()

    ids_path   = Path(args.ids)
    pdb_dir    = Path(args.pdb_dir)
    angles_dir = Path(args.angles_dir)
    boxes_dir  = Path(args.boxes_dir)
    out_dir    = Path(args.outdir)

    contexts = load_chain_contexts(ids_path)
    print(f"Loaded {len(contexts)} chain contexts from {ids_path}")

    for pdb_id, chain in contexts:
        prefix = f"{pdb_id}_{chain}"
        process_chain(prefix, pdb_dir, angles_dir, boxes_dir, out_dir)


if __name__ == "__main__":
    main()
