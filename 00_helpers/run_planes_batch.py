#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def load_chain_contexts(ids_path: Path):
    """
    Load unique (pdb_id, chain) pairs from ids.tsv.
    Expected format (tab-separated):
        pdb_id    chain
    """
    contexts = []
    seen = set()

    for raw in ids_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        pdb_id, chain = parts[0].lower(), parts[1]
        key = (pdb_id, chain)

        if key in seen:
            continue
        seen.add(key)

        contexts.append((pdb_id, chain))

    return contexts


def main():
    ap = argparse.ArgumentParser(
        description="Batch driver for planes_from_backbone_ortho_boxes.py"
    )
    ap.add_argument("--ids", required=True, help="ids.tsv (tab-separated: pdb_id, chain)")
    ap.add_argument("--pdb-dir", required=True, help="Directory of extracted chain PDBs")
    ap.add_argument("--outdir", required=True, help="Output directory for box CSVs")
    ap.add_argument(
        "--planes-script",
        required=True,
        help="Path to planes_from_backbone_ortho_boxes.py",
    )
    args = ap.parse_args()

    ids_path = Path(args.ids)
    pdb_dir = Path(args.pdb_dir)
    outdir = Path(args.outdir)
    planes_script = Path(args.planes_script)

    outdir.mkdir(parents=True, exist_ok=True)

    contexts = load_chain_contexts(ids_path)
    print(f"Loaded {len(contexts)} chain contexts")

    for pdb_id, chain in contexts:
        pdb_file = pdb_dir / f"{pdb_id}_{chain}.pdb"

        if not pdb_file.exists():
            print(f"[skip] Missing PDB: {pdb_file}")
            continue

        cmd = [
            "python",
            str(planes_script),
            str(pdb_file),
            "--outdir",
            str(outdir),
            "--csv",
            "--normal",
        ]

        print(f"Running planes on {pdb_file.name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] planes failed for {pdb_file.name}")
            print(e)


if __name__ == "__main__":
    main()
