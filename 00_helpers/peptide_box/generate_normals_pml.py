#!/usr/bin/env python3
# ASCII-safe PyMOL overlay generator: compares boxer normals from boxes PDB
# to recomputed peptide-link plane normals from the original source PDB.
#
# Usage (Windows examples):
#   python generate_normals_pml.py ^
#     --boxes output\4jea_boxes.pdb ^
#     --src   input_data\4jea.pdb ^
#     --out   output\4jea_overlay.pml ^
#     --angle-thresh 10 ^
#     --scale 2.0
# Then in PyMOL:  @output/4jea_overlay.pml

import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# ----------------- vector helpers -----------------
def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vnorm(a):
    return math.sqrt(vdot(a, a))

def vunit(a):
    n = vnorm(a)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def vcross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

# ----------------- parse source PDB (backbone) -----------------
class Atom:
    __slots__ = ("name","resname","chain","resseq","icode","x","y","z")
    def __init__(self, name, resname, chain, resseq, icode, x, y, z):
        self.name = name
        self.resname = resname
        self.chain = chain
        self.resseq = resseq
        self.icode = icode
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

Key = Tuple[str, int, str]  # (chain, resseq, icode)

def parse_source_pdb(path: Path) -> Dict[Key, Dict[str, Atom]]:
    bb: Dict[Key, Dict[str, Atom]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not (ln.startswith('ATOM') or ln.startswith('HETATM')):
                continue
            name = ln[12:16].strip()
            resname = ln[17:20].strip()
            chain = (ln[21] or ' ').strip() or ' '
            try:
                resseq = int(ln[22:26])
            except Exception:
                continue
            icode = (ln[26] or '.').strip() or '.'
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except Exception:
                continue
            key = (chain, resseq, icode)
            if key not in bb:
                bb[key] = {}
            bb[key][name] = Atom(name, resname, chain, resseq, icode, x, y, z)
    return bb

# ----------------- parse boxes PDB (centers + boxer normals) -----------------
# For each peptide link (i -> j), we expect atoms with residue number set to i:
#   - PV1..PV4 (rectangle corners)
#   - PCEN (center)
#   - PN1/PN2 (points along the boxer normal) when --normal is used

class BoxRecord:
    __slots__ = ("chain","resi","center","pn1","pn2","v1","v2","v3","v4")
    def __init__(self, chain: str, resi: int):
        self.chain = chain
        self.resi = resi
        self.center = None
        self.pn1 = None
        self.pn2 = None
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.v4 = None

Key2 = Tuple[str, int]  # (chain, resi_i)

def parse_boxes_pdb(path: Path) -> Dict[Key2, BoxRecord]:
    def xyz(ln: str):
        return (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    recs: Dict[Key2, BoxRecord] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not (ln.startswith('ATOM') or ln.startswith('HETATM')):
                continue
            name = ln[12:16].strip()
            chain = (ln[21] or ' ').strip() or ' '
            try:
                resi = int(ln[22:26])
            except Exception:
                continue
            key = (chain, resi)
            if key not in recs:
                recs[key] = BoxRecord(chain, resi)
            if name == 'PCEN':
                recs[key].center = xyz(ln)
            elif name == 'PN1':
                recs[key].pn1 = xyz(ln)
            elif name == 'PN2':
                recs[key].pn2 = xyz(ln)
            elif name == 'PV1':
                recs[key].v1 = xyz(ln)
            elif name == 'PV2':
                recs[key].v2 = xyz(ln)
            elif name == 'PV3':
                recs[key].v3 = xyz(ln)
            elif name == 'PV4':
                recs[key].v4 = xyz(ln)
    return recs

# ----------------- recompute link normal from source PDB -----------------
# Estimate the peptide-link plane between residues i and i+1 using two in-plane
# vectors: (C_i -> O_i) and (N_{i+1} -> CA_{i+1}). Works without NumPy.

def recompute_link_normal(bb: Dict[Key, Dict[str, Atom]], chain: str, resi_i: int) -> Optional[Tuple[float, float, float]]:
    # find next residue j after i in the same chain
    keys_in_chain = sorted([k for k in bb.keys() if k[0] == chain], key=lambda k: (k[1], k[2]))
    idx = None
    for t, k in enumerate(keys_in_chain):
        if k[1] == resi_i:
            idx = t
            break
    if idx is None or idx + 1 >= len(keys_in_chain):
        return None
    ki = keys_in_chain[idx]
    kj = keys_in_chain[idx + 1]
    at_i = bb.get(ki, {})
    at_j = bb.get(kj, {})
    Ci = at_i.get('C')
    Oi = at_i.get('O')
    Nj = at_j.get('N')
    CAj = at_j.get('CA')
    if not (Ci and Oi and Nj and CAj):
        return None
    v1 = (Oi.x - Ci.x, Oi.y - Ci.y, Oi.z - Ci.z)
    v2 = (CAj.x - Nj.x, CAj.y - Nj.y, CAj.z - Nj.z)
    n = vcross(v1, v2)
    return vunit(n)

# ----------------- boxer normal from boxes PDB -----------------

def boxer_normal_for_record(rec: BoxRecord) -> Optional[Tuple[float, float, float]]:
    if rec.pn1 and rec.pn2:
        v = (rec.pn2[0] - rec.pn1[0], rec.pn2[1] - rec.pn1[1], rec.pn2[2] - rec.pn1[2])
        return vunit(v)
    # fallback: use rectangle edges PV1->PV2 and PV1->PV4
    if rec.v1 and rec.v2 and rec.v4:
        e1 = (rec.v2[0] - rec.v1[0], rec.v2[1] - rec.v1[1], rec.v2[2] - rec.v1[2])
        e2 = (rec.v4[0] - rec.v1[0], rec.v4[1] - rec.v1[1], rec.v4[2] - rec.v1[2])
        n = vcross(e1, e2)
        return vunit(n)
    return None

# ----------------- PML writer -----------------

def cgo_arrow_block(base, tip, color):
    r_cyl = 0.15
    r_cone = 0.35
    cone_len = 0.8
    vx = tip[0] - base[0]
    vy = tip[1] - base[1]
    vz = tip[2] - base[2]
    vlen = math.sqrt(vx*vx + vy*vy + vz*vz) or 1e-6
    cone_len_loc = cone_len if vlen >= cone_len*1.1 else max(0.2, vlen*0.3)
    cx = tip[0] - vx*(cone_len_loc/vlen)
    cy = tip[1] - vy*(cone_len_loc/vlen)
    cz = tip[2] - vz*(cone_len_loc/vlen)
    r, g, b = color
    return [
        'CYLINDER', base[0], base[1], base[2], cx, cy, cz, r_cyl, r, g, b, r, g, b,
        'CONE', cx, cy, cz, tip[0], tip[1], tip[2], r_cone, 0.0, r, g, b, r, g, b, 1.0, 0.0
    ]

def write_pml(out_pml: Path, src_pdb: str, boxes_pdb: str, arrows_calc, arrows_box, labels):
    with open(out_pml, 'w', encoding='utf-8') as fh:
        fh.write(f"load {src_pdb}\n")
        fh.write(f"load {boxes_pdb}\n")
        fh.write("from pymol.cgo import *\nfrom pymol import cmd\n\n")
        fh.write("hide everything\n")
        fh.write("show cartoon, not resn PLN\n")
        fh.write("show sticks, resn PLN\n")
        def dump(name_prefix, arrows):
            chunk = 400
            for i in range(0, len(arrows), chunk):
                block = []
                for a in arrows[i:i+chunk]:
                    block += cgo_arrow_block(a['base'], a['tip'], a['color'])
                oname = f"{name_prefix}_{i//chunk}"
                fh.write(f"cmd.load_cgo([{', '.join(map(str, block))}], '{oname}')\n")
        dump('normals_calc', arrows_calc)
        dump('normals_box', arrows_box)
        for i, (x, y, z, text, (r, g, b)) in enumerate(labels):
            fh.write(f"pseudoatom lbl{i}, pos=[{x:.3f},{y:.3f},{z:.3f}], label='{text}'\n")
            fh.write(f"set label_color, [ {r:.2f}, {g:.2f}, {b:.2f} ], lbl{i}\n")
        fh.write("bg_color white\n")
        fh.write("set ray_opaque_background, off\n")
        fh.write("set label_size, 14\n")
        fh.write("set antialias, 2\n")

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description='Overlay boxer normals (boxes PDB) vs recomputed normals (source PDB)')
    ap.add_argument('--boxes', required=True, help='Path to *_boxes.pdb produced by planes_from_backbone_ortho_boxes.py')
    ap.add_argument('--src', required=True, help='Original source PDB used to build the boxes')
    ap.add_argument('--out', required=True, help='Output .pml path')
    ap.add_argument('--chain', default=None, help='Optional chain ID to restrict analysis')
    ap.add_argument('--angle-thresh', type=float, default=10.0)
    ap.add_argument('--scale', type=float, default=2.0, help='Arrow length (A)')
    args = ap.parse_args()

    boxes_p = Path(args.boxes).resolve()
    src_p = Path(args.src).resolve()
    out_p = Path(args.out)

    # parse inputs
    bb = parse_source_pdb(src_p)
    recs = parse_boxes_pdb(boxes_p)

    # build arrows and labels
    arrows_calc: List[dict] = []
    arrows_box: List[dict] = []
    labels: List[tuple] = []

    keys = sorted(recs.keys(), key=lambda k: (k[0], k[1]))
    for (chain, resi_i) in keys:
        if args.chain and chain != args.chain:
            continue
        rec = recs[(chain, resi_i)]
        center = rec.center or rec.v1 or rec.v2 or rec.v3 or rec.v4
        if center is None:
            continue
        n_box = boxer_normal_for_record(rec)
        n_calc = recompute_link_normal(bb, chain, resi_i)

        if n_box is not None and n_calc is not None:
            # compare with +/- equivalence
            dot = abs(vdot(n_calc, n_box) / (vnorm(n_calc)*vnorm(n_box) + 1e-12))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

            if ang > args.angle_thresh:
                tip_box  = (center[0] + args.scale*n_box[0],  center[1] + args.scale*n_box[1],  center[2] + args.scale*n_box[2])
                tip_calc = (center[0] + args.scale*n_calc[0], center[1] + args.scale*n_calc[1], center[2] + args.scale*n_calc[2])

                arrows_box.append({'base': center, 'tip': tip_box,  'color': (0.7, 0.2, 0.9)})
                arrows_calc.append({'base': center, 'tip': tip_calc, 'color': (0.0, 0.7, 1.0)})
                labels.append((center[0], center[1], center[2], f"{chain}{resi_i}:{ang:.1f} deg", (1.0, 0.2, 0.2)))
        # else: if either normal is missing, skip arrows entirely


    write_pml(out_p, str(src_p), str(boxes_p), arrows_calc, arrows_box, labels)
    print("[ok] Wrote PyMOL overlay: %s" % out_p)

if __name__ == '__main__':
    main()
