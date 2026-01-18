#!/usr/bin/env python3
import argparse
import urllib.request
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # keep only standard residues
        return residue.id[0] == " "


def load_ids_tsv(path: Path):
    """
    Load (pdb_id, chain) pairs from a tab-separated ids.tsv file.
    """
    pairs = []
    seen = set()

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        pdb_id, chain = line.split("\t")[:2]
        pdb_id = pdb_id.lower()

        key = (pdb_id, chain)
        if key in seen:
            continue
        seen.add(key)

        pairs.append(key)

    return pairs


from urllib.error import HTTPError

def download_cif(pdb_id: str, cif_dir: Path) -> Path | None:
    cif_dir.mkdir(parents=True, exist_ok=True)
    out = cif_dir / f"{pdb_id}.cif"

    if out.exists():
        return out

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    print(f"[download] {url}")

    try:
        urllib.request.urlretrieve(url, out)
    except HTTPError as e:
        print(f"[skip] {pdb_id}: CIF download failed ({e.code})")
        return None

    return out


def load_structure(pdb_id, pdb_dir, cif_dir):
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    if pdb_path.exists():
        parser = PDBParser(QUIET=True)
        return parser.get_structure(pdb_id, pdb_path)

    cif_path = download_cif(pdb_id, cif_dir)
    if cif_path is None:
        return None

    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(pdb_id, cif_path)


def extract_chain(pdb_id, chain_id, structure, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}_{chain_id}.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), select=ChainSelect(chain_id))

    print(f"[ok] wrote {out_path}")
    return out_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Extract single chains from full PDB/CIF structures"
    )
    ap.add_argument("--ids", required=True, help="ids.tsv (tab-separated: pdb_id, chain)")
    ap.add_argument("--pdb-dir", default="01_extract/pdb_raw", help="Directory of full PDB files")
    ap.add_argument("--cif-dir", default="01_extract/cif_cache", help="CIF download cache")
    ap.add_argument("--outdir", default="01_extract/chains_pdb", help="Output directory for chain PDBs")
    args = ap.parse_args()

    ids_path = Path(args.ids)
    pdb_dir  = Path(args.pdb_dir)
    cif_dir  = Path(args.cif_dir)
    out_dir  = Path(args.outdir)

    pdb_dir.mkdir(parents=True, exist_ok=True)

    targets = load_ids_tsv(ids_path)
    print(f"Loaded {len(targets)} (pdb, chain) targets")

    for pdb_id, chain_id in targets:
        structure = load_structure(pdb_id, pdb_dir, cif_dir)
        if structure is None:
            continue
        extract_chain(pdb_id, chain_id, structure, out_dir)

if __name__ == "__main__":
    main()
