#!/usr/bin/env python3
"""
planes_from_backbone_ortho_boxes.py

Generate rectangles that approximate peptide planes from a backbone:
{CA(i), C(i), O(i), N(i+1), CA(i+1), HN(i+1 if present)}. Fit a best plane (SVD),
build an orthogonal in-plane basis, then create an axis-aligned rectangle that
bounds anchors {CA(i), O(i), CA(i+1), N/HN proj} under range-based extents.

Features:
- Robust when HN is absent (always includes projected N in extents).
- Optional normal line through rectangle center (±1 Å) via --normal (default OFF).
- Interior "horizontal" lines count via --horiz-lines/-hl N (0..9, default 5), along the long axis.
- No cross-box diagonals in CONECT; perimeter + optional interior lines only.
- TER after each rectangle; LF line endings.
- CSV outputs (normals, adjacent angles) and optional SVG plot.
- --color-ss: emit <output>_color_ss.pml that colors rectangles by secondary structure
  parsed from HELIX/SHEET records in the input PDB (Helix=cyan, Sheet=magenta, Other=orange),
  and forces sticks representation in that PML.
- --as-sticks: emit <output>_as_sticks.pml that loads the boxes and forces sticks
  (optionally set stick radius with --stick-radius).
- --force-chain: stamp all PLN records with a specific chain ID and use it in PML selections.
- Output PDB name defaults to <input_basename>_boxes.pdb (override with --output).
- --from-input-data: process all *.pdb in ./input_data (batch mode).
- SVG plot annotates the input filename at the top-left.
- NEW: --outdir: write all outputs (PDB/CSV/SVG/PML) into a dedicated directory.
"""

import sys, os, math, argparse, csv, collections, glob
from typing import Dict, Tuple, List, Optional, Set
import numpy as np

# ----------------------------- PDB parsing ----------------------------------

class Atom:
    __slots__ = ("name","resname","chain","resseq","icode","x","y","z")
    def __init__(self, name, resname, chain, resseq, icode, x, y, z):
        self.name = name.strip()
        self.resname = resname.strip()
        self.chain = chain.strip() or ""
        self.resseq = int(resseq)
        self.icode = icode.strip()
        self.x, self.y, self.z = float(x), float(y), float(z)

def parse_pdb(path: str) -> Dict[Tuple[str,int,str,str], Dict[str, Atom]]:
    """Return {(chain, resseq, icode, resname) -> {atom_name -> Atom}}"""
    resmap: Dict[Tuple[str,int,str,str], Dict[str, Atom]] = {}
    with open(path, "r") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            name = ln[12:16]
            resname = ln[17:20]
            chain = ln[21:22]
            resseq = ln[22:26].strip() or "0"
            icode = ln[26:27]
            x = ln[30:38]; y = ln[38:46]; z = ln[46:54]
            key = (chain.strip(), int(resseq), icode.strip(), resname.strip())
            a = Atom(name, resname, chain, resseq, icode, x, y, z)
            resmap.setdefault(key, {})[a.name.strip()] = a
    return resmap

def sort_key_icode(icode: str) -> Tuple[int,str]:
    return (0, "") if (icode is None or icode.strip() == "") else (1, icode.strip())

def residues_by_chain(resmap):
    by_chain = collections.defaultdict(list)
    for key, atoms in resmap.items():
        chain, resseq, icode, resname = key
        by_chain[chain].append((key, atoms))
    for ch in sorted(by_chain.keys(), key=lambda s: s or ""):
        lst = by_chain[ch]
        lst.sort(key=lambda t: (t[0][1], sort_key_icode(t[0][2])))
        yield ch, lst

# ----------------------------- Math helpers ---------------------------------

def _v(a: Atom) -> np.ndarray:
    return np.array([a.x, a.y, a.z], dtype=float)

def _dist3(a: Atom, b: Atom) -> float:
    return float(np.linalg.norm(_v(a) - _v(b)))

# NEW: guard against gaps—only treat i/j as neighbors if C(i)–N(j) is a peptide bond
def is_peptide_link(at_i: Dict[str, Atom], at_j: Dict[str, Atom], max_cn: float = 1.7) -> bool:
    """
    Return True if residues i and j appear to be covalently linked by a peptide bond:
      - both have backbone C (i) and N (j)
      - distance C_i–N_{j} within peptide-bond range (default ≤ 1.7 Å)
    This gate prevents computing planes across gaps/TERs/missing residues.
    """
    Ci = at_i.get("C")
    Nj = at_j.get("N")
    if Ci is None or Nj is None:
        return False
    try:
        return _dist3(Ci, Nj) <= float(max_cn)
    except Exception:
        return False

def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n if n > 0 else 1.0)

def project_to_plane(P: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    return P - np.dot(P - origin, n) * n

def fit_plane(points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    M = np.vstack(points)
    c = M.mean(axis=0)
    X = M - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1, :]
    dists = X @ normalize(n)
    rms = float(np.sqrt((dists**2).mean())) if len(points) else 0.0
    return c, normalize(n), rms

def ortho_axes_in_plane(A: np.ndarray, B: np.ndarray, C: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    n = normalize(n)
    ACp = (C - A) - np.dot(C - A, n) * n
    u = normalize(ACp)
    ABp = (B - A) - np.dot(B - A, n) * n
    ABp -= np.dot(ABp, u) * u
    v = normalize(ABp)
    if np.linalg.norm(v) < 1e-8:
        v = normalize(np.cross(n, u))
    return u, v

# FIXED: Improved angle calculation functions for better stability
def ensure_consistent_normal_orientation(n1: np.ndarray, n2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure consistent orientation between two normal vectors.
    If they point in generally opposite directions, flip one to minimize the angle.
    """
    n1_norm = normalize(n1)
    n2_norm = normalize(n2)
    
    # If dot product is negative, they're pointing in opposite directions
    if np.dot(n1_norm, n2_norm) < 0:
        n2_norm = -n2_norm
    
    return n1_norm, n2_norm

def stable_angle_between_normals(n1: np.ndarray, n2: np.ndarray, reference_axis: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Calculate the angle between two normal vectors with improved numerical stability.
    
    Returns:
        - unsigned_angle: Always positive angle in degrees (0-180)
        - signed_angle: Signed angle in degrees (-180 to +180) if reference_axis provided
    """
    # Ensure unit vectors
    n1_unit = normalize(n1)
    n2_unit = normalize(n2)
    
    # Calculate dot product with clamping to handle numerical errors
    dot_product = np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0)
    
    # Calculate unsigned angle using arccos (always 0-180 degrees)
    unsigned_angle = np.degrees(np.arccos(np.abs(dot_product)))
    
    if reference_axis is None:
        return unsigned_angle, unsigned_angle
    
    # For signed angle, use the cross product and reference axis
    cross_product = np.cross(n1_unit, n2_unit)
    
    # Project cross product onto reference axis to determine sign
    reference_unit = normalize(reference_axis)
    sign_indicator = np.dot(cross_product, reference_unit)
    
    # Use atan2 for better numerical stability around 90 degrees
    sin_component = np.linalg.norm(cross_product)
    cos_component = dot_product
    
    # Determine sign: if original dot product was negative, we need to account for that
    if dot_product < 0:
        cos_component = -np.abs(cos_component)
    
    # Apply sign from cross product
    if sign_indicator < 0:
        sin_component = -sin_component
    
    signed_angle = np.degrees(np.arctan2(sin_component, cos_component))
    
    return unsigned_angle, signed_angle

def calculate_dihedral_angle(n1: np.ndarray, n2: np.ndarray, bond_vector: np.ndarray) -> float:
    """
    Calculate dihedral angle between two planes defined by their normals.
    Uses the bond vector as reference for sign determination.
    
    This is more appropriate for protein backbone dihedral angles.
    """
    n1_unit = normalize(n1)
    n2_unit = normalize(n2)
    bond_unit = normalize(bond_vector)
    
    # Calculate the cross product of the normals
    cross_product = np.cross(n1_unit, n2_unit)
    
    # Calculate components for atan2
    cos_component = np.dot(n1_unit, n2_unit)
    sin_component = np.dot(cross_product, bond_unit)
    
    # Use atan2 for numerical stability
    dihedral = np.degrees(np.arctan2(sin_component, cos_component))
    
    return dihedral

# ---------------------- Geometry & rectangle construction -------------------

def best_HN_for_resj(atoms_j: Dict[str,Atom], N_j: Optional[Atom]) -> Optional[Atom]:
    if atoms_j is None or N_j is None:
        return None
    if "HN" in atoms_j:
        return atoms_j["HN"]
    candidates = [a for nm,a in atoms_j.items() if nm.strip().upper().startswith("H")]
    best = None; best_d = 1e9
    for h in candidates:
        d = _dist3(h, N_j)
        if d < best_d:
            best, best_d = h, d
    if best is not None and best_d <= 1.35:
        return best
    return None

def build_plane_for_pair(at_i, at_j, pads) -> Optional[dict]:
    CAi = at_i.get("CA"); Ci = at_i.get("C"); Oi = at_i.get("O")
    Nj = at_j.get("N"); CAj = at_j.get("CA")
    if CAi is None or Ci is None or Oi is None or Nj is None or CAj is None:
        return None

    HNj = best_HN_for_resj(at_j, Nj)
    pts = [_v(CAi), _v(Ci), _v(Oi), _v(Nj), _v(CAj)]
    if HNj is not None:
        pts.append(_v(HNj))
    c, n, rms = fit_plane(pts)

    A = _v(CAi); B = _v(Oi); C = _v(CAj)
    u, v = ortho_axes_in_plane(A, B, C, n)

    # Right-handed consistency
    bn = normalize(np.cross(u, v))
    nn = normalize(n)
    if float(np.dot(nn, bn)) < 0.0:
        nn = -nn

    # Always include projected N; include projected HN if present
    anchors = [A, B, C, project_to_plane(_v(Nj), c, nn)]
    if HNj is not None:
        anchors.append(project_to_plane(_v(HNj), c, nn))

    u_coords = [float(np.dot(P - A, u)) for P in anchors]
    v_coords = [float(np.dot(P - A, v)) for P in anchors]
    umin, umax = min(u_coords), max(u_coords)
    vmin, vmax = min(v_coords), max(v_coords)

    # padding options
    pad_global = max(0.0, float(pads.get("pad", 0.0)))
    umin -= pad_global; umax += pad_global
    vmin -= pad_global; vmax += pad_global
    if pads.get("pad_u") is not None:
        pu = max(0.0, float(pads["pad_u"]))
        umin -= pu; umax += pu
    if pads.get("pad_v") is not None:
        pv = max(0.0, float(pads["pad_v"]))
        vmin -= pv; vmax += pv
    for key, sign, axis in [
        ("pad_u_min",-1,"u"), ("pad_u_max",+1,"u"),
        ("pad_v_min",-1,"v"), ("pad_v_max",+1,"v")
    ]:
        if pads.get(key) is not None:
            pd = max(0.0, float(pads[key]))
            if axis == "u":
                if sign < 0: umin -= pd
                else:        umax += pd
            else:
                if sign < 0: vmin -= pd
                else:        vmax += pd

    # rectangle
    m = A + ((umin + umax) * 0.5) * u + ((vmin + vmax) * 0.5) * v
    hx = (umax - umin) * 0.5 + 1e-6
    hy = (vmax - vmin) * 0.5 + 1e-6
    V1 = m + (-hx)*u + (-hy)*v
    V2 = m + ( +hx)*u + (-hy)*v
    V3 = m + ( +hx)*u + ( +hy)*v
    V4 = m + (-hx)*u + ( +hy)*v

    chain = at_i[next(iter(at_i))].chain.strip()
    resname_i = at_i[next(iter(at_i))].resname.strip()
    resname_j = at_j[next(iter(at_j))].resname.strip()
    any_i = next(iter(at_i.values())); any_j = next(iter(at_j.values()))
    
    # FIXED: Store the bond vector for more accurate dihedral calculations
    bond_vector = _v(Nj) - _v(Ci)  # C(i) to N(j) peptide bond direction
    
    plane = dict(
        chain=chain,
        resseq_i=any_i.resseq, icode_i=any_i.icode or "",
        resname_i=resname_i,
        resseq_j=any_j.resseq, icode_j=any_j.icode or "",
        resname_j=resname_j,
        normal=tuple(float(x) for x in nn),
        center=tuple(float(x) for x in m),
        corners=[V1,V2,V3,V4],
        u_axis=tuple(float(x) for x in u),
        v_axis=tuple(float(x) for x in v),
        bond_vector=tuple(float(x) for x in normalize(bond_vector)),  # Store normalized bond vector
        hx=float(hx), hy=float(hy),
        rms=float(rms),
    )
    return plane

# ------------------------------ CSV & plotting ------------------------------

def _to_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def write_normals_csv(base: str, planes: List[dict]) -> str:
    path = base + "_normals.csv"
    with open(path, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["plane_index","chain","res_i","icode_i","resname_i",
                    "res_j","icode_j","resname_j","nx","ny","nz","rms"])
        for k, pl in enumerate(planes, start=1):
            nx, ny, nz = [float(pl["normal"][0]), float(pl["normal"][1]), float(pl["normal"][2])]
            rms = float(pl["rms"])
            w.writerow([
                k,
                pl["chain"].strip(),
                str(pl["resseq_i"]).strip(), pl["icode_i"].strip(), pl["resname_i"].strip(),
                str(pl["resseq_j"]).strip(), pl["icode_j"].strip(), pl["resname_j"].strip(),
                f"{nx:.6f}", f"{ny:.6f}", f"{nz:.6f}", f"{rms:.6f}"
            ])
    return path

def write_adjacent_angles_csv(base: str, planes: List[dict], make_plots: bool, plot_label: Optional[str]=None):
    """
    FIXED: Improved angle calculation with better numerical stability and consistent sign handling.
    """
    path = base + "_adjacent_angles.csv"
    x_idx: List[float] = []
    y_signed: List[float] = []
    y_unsigned: List[float] = []

    with open(path, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["plane_index_A","plane_index_B","chain",
                    "res_i_A","res_j_A","res_i_B","res_j_B",
                    "dot","angle_unsigned_deg","angle_signed_deg","dihedral_deg"])
        
        for k in range(len(planes)-1):
            A = planes[k]; B = planes[k+1]
            if A["chain"] != B["chain"]:
                continue
            ai = _to_int(A["resseq_i"]); aj = _to_int(A["resseq_j"])
            bi = _to_int(B["resseq_i"]); bj = _to_int(B["resseq_j"])
            if aj is None or bi is None or (aj != bi):
                continue

            # Get normal vectors
            n1 = np.array(A["normal"], dtype=float)
            n2 = np.array(B["normal"], dtype=float)
            
            # FIXED: Use improved angle calculation
            n1_consistent, n2_consistent = ensure_consistent_normal_orientation(n1, n2)
            
            # Calculate dot product for reference
            dot = float(np.dot(n1_consistent, n2_consistent))
            dot = np.clip(dot, -1.0, 1.0)  # Clamp for numerical safety
            
            # Use u-axis as reference for signed angle
            uA = np.array(A.get("u_axis", (1.0, 0.0, 0.0)), dtype=float)
            unsigned_angle, signed_angle = stable_angle_between_normals(n1, n2, uA)
            
            # FIXED: Calculate dihedral angle using bond vector if available
            bond_vector_A = np.array(A.get("bond_vector", (0.0, 0.0, 1.0)), dtype=float)
            dihedral_angle = calculate_dihedral_angle(n1, n2, bond_vector_A)

            w.writerow([k, k+1, A["chain"].strip(),
                        str(A["resseq_i"]).strip(), str(A["resseq_j"]).strip(),
                        str(B["resseq_i"]).strip(), str(B["resseq_j"]).strip(),
                        f"{dot:.6f}", f"{unsigned_angle:.6f}", f"{signed_angle:.6f}", f"{dihedral_angle:.6f}"])

            xi = ai if ai is not None else k
            x_idx.append(xi)
            y_signed.append(signed_angle)
            y_unsigned.append(unsigned_angle)

    signed_svg = None
    unsigned_svg = None
    if make_plots and x_idx:
        try:
            import matplotlib
            matplotlib.rcParams["svg.fonttype"] = "none"
            import matplotlib.pyplot as plt
            
            # Signed angles plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_idx, y_signed, 'b-', linewidth=1.5, label='Signed angles')
            plt.xlabel("Residue index i")
            plt.ylabel("Signed angle (deg)")
            plt.title("Adjacent plane signed angles")
            plt.grid(True, alpha=0.3)
            ax = plt.gca()
            if plot_label:
                ax.text(0.01, 0.99, str(plot_label), transform=ax.transAxes,
                        ha="left", va="top")
            plt.tight_layout()
            signed_svg = base + "_angles_signed.svg"
            plt.savefig(signed_svg, format="svg", dpi=150)
            plt.close()
            
            # Unsigned angles plot  
            plt.figure(figsize=(10, 6))
            plt.plot(x_idx, y_unsigned, 'r-', linewidth=1.5, label='Unsigned angles')
            plt.xlabel("Residue index i")
            plt.ylabel("Unsigned angle (deg)")
            plt.title("Adjacent plane unsigned angles")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 180)
            ax = plt.gca()
            if plot_label:
                ax.text(0.01, 0.99, str(plot_label), transform=ax.transAxes,
                        ha="left", va="top")
            plt.tight_layout()
            unsigned_svg = base + "_angles_unsigned.svg"
            plt.savefig(unsigned_svg, format="svg", dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"[warn] Plotting failed: {e}")
    
    return path, signed_svg, unsigned_svg

def write_per_chain_csvs(base: str, planes: List[dict]) -> List[str]:
    """
    FIXED: Updated to use improved angle calculations in per-chain CSVs.
    """
    out_paths = []
    groups = collections.defaultdict(list)
    for k, pl in enumerate(planes, start=1):
        groups[pl["chain"]].append((k, pl))

    def safe_chain_id(s: str) -> str:
        s = (s or "").strip() or "blank"
        return "".join(ch if ch.isalnum() else "_" for ch in s)

    for ch, items in groups.items():
        ch_safe = safe_chain_id(ch)
        n_path = f"{base}_chain_{ch_safe}_normals.csv"
        a_path = f"{base}_chain_{ch_safe}_adjacent_angles.csv"
        
        with open(n_path, "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["plane_index","chain","res_i","icode_i","resname_i",
                        "res_j","icode_j","resname_j","nx","ny","nz","rms"])
            for k, pl in items:
                nx, ny, nz = [float(pl["normal"][0]), float(pl["normal"][1]), float(pl["normal"][2])]
                rms = float(pl["rms"])
                w.writerow([
                    k, pl["chain"].strip(),
                    str(pl["resseq_i"]).strip(), pl["icode_i"].strip(), pl["resname_i"].strip(),
                    str(pl["resseq_j"]).strip(), pl["icode_j"].strip(), pl["resname_j"].strip(),
                    f"{nx:.6f}", f"{ny:.6f}", f"{nz:.6f}", f"{rms:.6f}"
                ])
        out_paths.append(n_path)

        with open(a_path, "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["plane_index_A","plane_index_B","chain","res_i_A","res_j_A",
                        "res_i_B","res_j_B","dot","angle_unsigned_deg","angle_signed_deg","dihedral_deg"])
            for i in range(len(items)-1):
                (kA, A), (kB, B) = items[i], items[i+1]
                ai = _to_int(A["resseq_i"]); aj = _to_int(A["resseq_j"])
                bi = _to_int(B["resseq_i"]); bj = _to_int(B["resseq_j"])
                if aj is None or bi is None or aj != bi:
                    continue
                
                # FIXED: Use improved angle calculation
                n1 = np.array(A["normal"], dtype=float)
                n2 = np.array(B["normal"], dtype=float)
                
                n1_consistent, n2_consistent = ensure_consistent_normal_orientation(n1, n2)
                dot = float(np.dot(n1_consistent, n2_consistent))
                dot = np.clip(dot, -1.0, 1.0)
                
                uA = np.array(A.get("u_axis", (1.0, 0.0, 0.0)), dtype=float)
                unsigned_angle, signed_angle = stable_angle_between_normals(n1, n2, uA)
                
                bond_vector_A = np.array(A.get("bond_vector", (0.0, 0.0, 1.0)), dtype=float)
                dihedral_angle = calculate_dihedral_angle(n1, n2, bond_vector_A)
                
                w.writerow([kA, kB, A["chain"].strip(),
                           str(A["resseq_i"]).strip(), str(A["resseq_j"]).strip(),
                           str(B["resseq_i"]).strip(), str(B["resseq_j"]).strip(),
                           f"{dot:.6f}", f"{unsigned_angle:.6f}", f"{signed_angle:.6f}", f"{dihedral_angle:.6f}"])
        out_paths.append(a_path)
    return out_paths

# ---------------------- Secondary structure → PML ---------------------------

def _safe_int2(s, default=None):
    try:
        return int(str(s).strip())
    except Exception:
        return default

def parse_secondary_structure_from_pdb(pdb_path: str) -> Tuple[Set[Tuple[str,int]], Set[Tuple[str,int]]]:
    """
    Parse HELIX/SHEET ranges from the input PDB.
    Returns (helix_set, sheet_set) of (chain, resseq) pairs. (Ignores insertion codes.)
    """
    helix: Set[Tuple[str,int]] = set()
    sheet: Set[Tuple[str,int]] = set()
    with open(pdb_path, "r") as f:
        for ln in f:
            rec = ln[:6]
            if rec.startswith("HELIX "):
                ch_i = ln[19:20].strip() or ""
                res_i = _safe_int2(ln[21:25], None)
                ch_j = ln[31:32].strip() or ""
                res_j = _safe_int2(ln[33:37], None)
                if res_i is not None and res_j is not None and (ch_i == ch_j):
                    a, b = sorted((res_i, res_j))
                    for r in range(a, b+1):
                        helix.add((ch_i, r))
            elif rec.startswith("SHEET "):
                ch_i = ln[21:22].strip() or ""
                res_i = _safe_int2(ln[22:26], None)
                ch_j = ln[32:33].strip() or ""
                res_j = _safe_int2(ln[33:37], None)
                if res_i is not None and res_j is not None and (ch_i == ch_j):
                    a, b = sorted((res_i, res_j))
                    for r in range(a, b+1):
                        sheet.add((ch_i, r))
    return helix, sheet

def write_color_ss_pml(out_pdb_path: str, planes: List[dict],
                       helix_set: Set[Tuple[str,int]], sheet_set: Set[Tuple[str,int]],
                       helix_color: str = "cyan",
                       sheet_color: str = "magenta",
                       other_color: str = "orange",
                       force_chain: Optional[str] = None) -> str:
    """
    Creates <out_base>_color_ss.pml that:
      - loads the boxes PDB as object '<basename>'
      - forces sticks
      - colors all PLN rectangles 'other_color' first
      - recolors helix residues 'helix_color' and sheet residues 'sheet_color'
    """
    base, _ = os.path.splitext(out_pdb_path)
    pml_path = base + "_color_ss.pml"
    objname = os.path.basename(base)

    pdb_path_for_pml = os.path.abspath(out_pdb_path).replace("\\", "/")

    with open(pml_path, "w", newline="\n", encoding="ascii") as g:
        g.write(f'load "{pdb_path_for_pml}", {objname}\n')
        g.write(f"hide everything, {objname}\n")
        g.write(f"as sticks, {objname}\n")
        g.write(f"color {other_color}, {objname} and resn PLN\n")
        for pl in planes:
            ch_orig = (pl['chain'] or "").strip()
            ch_sel  = ((force_chain or ch_orig or "")[:1])
            ri = int(pl['resseq_i'])
            sel = f"{objname} and resn PLN and resi {ri}"
            if ch_sel:
                sel += f" and chain {ch_sel}"
            if (ch_orig, ri) in helix_set:
                g.write(f"color {helix_color}, {sel}\n")
            elif (ch_orig, ri) in sheet_set:
                g.write(f"color {sheet_color}, {sel}\n")
        g.write("\n")
    return pml_path

def write_as_sticks_pml(out_pdb_path: str, stick_radius: Optional[float] = None) -> str:
    """
    Creates <out_base>_as_sticks.pml that:
      - loads the boxes PDB as object '<basename>'
      - hides everything and shows sticks (optionally sets stick_radius)
    """
    base, _ = os.path.splitext(out_pdb_path)
    pml_path = base + "_as_sticks.pml"
    objname = os.path.basename(base)
    pdb_path_for_pml = os.path.abspath(out_pdb_path).replace("\\", "/")
    with open(pml_path, "w", encoding="ascii", newline="\n") as g:
        g.write(f'load "{pdb_path_for_pml}", {objname}\n')
        g.write(f"hide everything, {objname}\n")
        g.write(f"as sticks, {objname}\n")
        if stick_radius is not None:
            g.write(f"set stick_radius, {float(stick_radius):.3f}\n")
        g.write("\n")
    return pml_path

# ----------------------------- PDB rectangles -------------------------------

def format_pdb_hetatm(
    serial: int,
    name: str,
    resn: str,
    chain: str,
    resi: int,
    x: float,
    y: float,
    z: float,
    icode: str = "",
    element: str = "C",
    altloc: str = " "
) -> str:
    """
    Emit a PDB HETATM line with correct fixed columns.
    """
    return (
        "HETATM{serial:5d} "
        "{name:<4s}{altloc:1s}{resn:>3s} {chain:1s}{resi:4d}{icode:1s}   "
        "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}  "
    ).format(
        serial=serial,
        name=name[:4],
        altloc=(altloc or " ")[:1],
        resn=resn[:3],
        chain=(chain or " ")[:1],
        resi=int(resi),
        icode=(icode or " ")[:1],
        x=x, y=y, z=z,
        element=(element or "C")[:2].rjust(2),
    )

def _interior_offsets(span_half: float, n: int) -> List[float]:
    """Return n evenly spaced offsets in (-span_half, +span_half) (open interval)."""
    if n <= 0:
        return []
    return [(-span_half + (i+1) * (2.0*span_half) / (n+1)) for i in range(n)]

def write_rectangles_pdb(path: str, planes: List[dict], resname: str = "PLN",
                         draw_normal: bool = False, horiz_lines: int = 5,
                         force_chain: Optional[str] = None) -> None:
    """
    For each plane:
      - HETATM PV1..PV4 (rectangle corners)
      - (optional) HETATM PCEN, PN1, PN2 for normal line (±1 Å) if draw_normal=True
      - N interior horizontal lines along the long axis via pairs PHkA/PHkB (k=1..N)
      - TER after each group; CONECT for perimeter + interior lines; no diagonals
    """
    serial = 1
    adjacency = {}  # atom_serial -> set(neighbor_serials)

    def add_edge(a: int, b: int):
        if a == b: return
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    n_interior = max(0, min(9, int(horiz_lines)))

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("REMARK PLN rectangles with optional normals and interior lines\n")
        for k, pl in enumerate(planes, start=1):
            chain_out = ((force_chain or pl["chain"] or " ")[:1])
            resi = int(pl["resseq_i"])
            V1, V2, V3, V4 = pl["corners"]
            m = np.array(pl["center"], dtype=float)
            u = np.array(pl["u_axis"], dtype=float)
            v = np.array(pl["v_axis"], dtype=float)
            hx = float(pl.get("hx", 0.5*np.linalg.norm(np.array(V2)-np.array(V1))))
            hy = float(pl.get("hy", 0.5*np.linalg.norm(np.array(V4)-np.array(V1))))
            nx, ny, nz = pl.get("normal", (0.0, 0.0, 1.0))

            # Corner atoms
            s1 = serial; f.write(format_pdb_hetatm(s1, "PV1", resname, chain_out, resi, V1[0], V1[1], V1[2]) + "\n")
            s2 = serial+1; f.write(format_pdb_hetatm(s2, "PV2", resname, chain_out, resi, V2[0], V2[1], V2[2]) + "\n")
            s3 = serial+2; f.write(format_pdb_hetatm(s3, "PV3", resname, chain_out, resi, V3[0], V3[1], V3[2]) + "\n")
            s4 = serial+3; f.write(format_pdb_hetatm(s4, "PV4", resname, chain_out, resi, V4[0], V4[1], V4[2]) + "\n")
            serial += 4

            # Perimeter without diagonals
            add_edge(s1, s2); add_edge(s2, s3); add_edge(s3, s4); add_edge(s4, s1)

            # Interior "horizontal" lines: along the long axis, strictly inside
            long_is_u = hx >= hy
            if long_is_u:
                offsets = _interior_offsets(hy, n_interior)
                for idx, off in enumerate(offsets, start=1):
                    A = m + (-hx)*u + off*v
                    B = m + ( +hx)*u + off*v
                    sa = serial; f.write(format_pdb_hetatm(sa, f"PH{idx}A", resname, chain_out, resi, A[0], A[1], A[2]) + "\n")
                    sb = serial+1; f.write(format_pdb_hetatm(sb, f"PH{idx}B", resname, chain_out, resi, B[0], B[1], B[2]) + "\n")
                    add_edge(sa, sb)
                    serial += 2
            else:
                offsets = _interior_offsets(hx, n_interior)
                for idx, off in enumerate(offsets, start=1):
                    A = m + off*u + (-hy)*v
                    B = m + off*u + ( +hy)*v
                    sa = serial; f.write(format_pdb_hetatm(sa, f"PH{idx}A", resname, chain_out, resi, A[0], A[1], A[2]) + "\n")
                    sb = serial+1; f.write(format_pdb_hetatm(sb, f"PH{idx}B", resname, chain_out, resi, B[0], B[1], B[2]) + "\n")
                    add_edge(sa, sb)
                    serial += 2

            # Optional normal line through center (±1 Å along normal)
            if draw_normal:
                cx, cy, cz = m.tolist()
                Pcen = (cx, cy, cz)
                Pn1 = (cx - nx, cy - ny, cz - nz)
                Pn2 = (cx + nx, cy + ny, cz + nz)
                sc = serial;  f.write(format_pdb_hetatm(sc, "PCEN", resname, chain_out, resi, *Pcen) + "\n")
                sn1 = serial+1; f.write(format_pdb_hetatm(sn1, "PN1",  resname, chain_out, resi, *Pn1) + "\n")
                sn2 = serial+2; f.write(format_pdb_hetatm(sn2, "PN2",  resname, chain_out, resi, *Pn2) + "\n")
                add_edge(sn1, sc); add_edge(sc, sn2)
                serial += 3

            # TER record (after this rectangle's atoms)
            f.write(f"TER   {serial:5d}      {resname:>3s} {chain_out:1s}{resi:4d}\n")
            serial += 1

        # Emit CONECTs
        for a in sorted(adjacency.keys()):
            nbrs = sorted(adjacency[a])
            for i in range(0, len(nbrs), 4):
                chunk = nbrs[i:i+4]
                f.write("CONECT{a:5d}{b1}{b2}{b3}{b4}\n".format(
                    a=a,
                    b1=f"{chunk[0]:5d}" if len(chunk) > 0 else "     ",
                    b2=f"{chunk[1]:5d}" if len(chunk) > 1 else "     ",
                    b3=f"{chunk[2]:5d}" if len(chunk) > 2 else "     ",
                    b4=f"{chunk[3]:5d}" if len(chunk) > 3 else "     ",
                ))
        f.write("END\n")

# ----------------------------- Processing -----------------------------------

def derive_output_path(input_path: str, override_output: Optional[str], outdir: Optional[str]) -> str:
    """
    Returns the output PDB path. Priority:
      1) explicit override_output
      2) <outdir>/<input_stem>_boxes.pdb (creating outdir if needed)
      3) <input_dir>/<input_stem>_boxes.pdb
    """
    if override_output:
        d, _ = os.path.split(override_output)
        if d and (not os.path.isdir(d)):
            os.makedirs(d, exist_ok=True)
        return override_output
    d, b = os.path.split(input_path)
    stem, _ = os.path.splitext(b)
    out_name = f"{stem}_boxes.pdb"
    out_dir = outdir or d or "."
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_name)

def process_one(input_path: str, args) -> None:
    pads = dict(
        pad=args.pad, pad_u=args.pad_u, pad_v=args.pad_v,
        pad_u_min=args.pad_u_min, pad_u_max=args.pad_u_max,
        pad_v_min=args.pad_v_min, pad_v_max=args.pad_v_max
    )

    n_interior = max(0, min(9, int(args.horiz_lines)))
    if n_interior != args.horiz_lines:
        print(f"[info] --horiz-lines clipped to {n_interior} (allowed 0..9)")

    fc = (args.force_chain[:1] if args.force_chain else None)

    print(f"[1/4] Parsing PDB: {input_path}")
    resmap = parse_pdb(input_path)

    planes: List[dict] = []
    for chain, lst in residues_by_chain(resmap):
        if args.chain and chain != args.chain:
            continue
        for idx in range(len(lst)-1):
            (key_i, at_i) = lst[idx]
            (key_j, at_j) = lst[idx+1]

            # NEW: skip non-adjacent (gapped) pairs unless C(i)-N(j) is peptide-bond distance
            if not is_peptide_link(at_i, at_j, max_cn=args.cn_cutoff):
                continue

            plane = build_plane_for_pair(at_i, at_j, pads)
            if plane is not None:
                planes.append(plane)

    print(f"[2/4] Constructed {len(planes)} plane rectangles")

    out_pdb = derive_output_path(input_path, args.output, args.outdir)
    write_rectangles_pdb(out_pdb, planes, draw_normal=args.normal,
                         horiz_lines=n_interior, force_chain=fc)
    print(f"[3/4] Wrote rectangles-only PDB: {out_pdb}")

    base, _ = os.path.splitext(out_pdb)

    if args.csv:
        normals_csv = write_normals_csv(base, planes)
        angles_csv, signed_svg, unsigned_svg = write_adjacent_angles_csv(
            base, planes, args.plot, plot_label=os.path.basename(input_path)
        )
        print(f"[CSV] normals: {normals_csv}")
        print(f"[CSV] angles : {angles_csv}")
        if args.csv_per_chain:
            for pth in write_per_chain_csvs(base, planes):
                print(f"[CSV] per-chain: {pth}")
        if signed_svg:
            print(f"[PLOT] signed angles: {signed_svg}")
        if unsigned_svg:
            print(f"[PLOT] unsigned angles: {unsigned_svg}")

    if args.color_ss:
        helix_set, sheet_set = parse_secondary_structure_from_pdb(input_path)
        pml = write_color_ss_pml(out_pdb, planes, helix_set, sheet_set,
                                 helix_color="cyan", sheet_color="magenta",
                                 other_color="orange", force_chain=fc)
        print(f"[PML] secondary-structure coloring: {pml}")
    elif args.as_sticks:
        pml = write_as_sticks_pml(out_pdb, args.stick_radius)
        print(f"[PML] sticks style: {pml}")

    print("[4/4] Done.")

# ---------------------------------- CLI -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate orthogonalized rectangles from peptide planes.")
    # Single-file input is optional when using --from-input-data
    ap.add_argument("input", nargs="?", help="input PDB (ignored if --from-input-data)")
    ap.add_argument("--output", help="override output PDB path (single-file mode only)")
    ap.add_argument("--outdir", default=None,
                    help="directory to write all outputs; will be created if missing")
    ap.add_argument("--from-input-data", action="store_true",
                    help="process all *.pdb files in ./input_data directory")
    ap.add_argument("--chain", help="restrict to a single chain ID", default=None)
    ap.add_argument("--min-sep", type=int, default=0, help="minimum resseq increment for adjacency (reserved)")

    # padding flags
    ap.add_argument("--pad", type=float, default=0.0, help="extra in-plane padding (Å) added to both axes; 0 disables")
    ap.add_argument("--pad-u", type=float, default=None, help="symmetric padding along +/−u (Å), overrides --pad for u if set")
    ap.add_argument("--pad-v", type=float, default=None, help="symmetric padding along +/−v (Å), overrides --pad for v if set")
    ap.add_argument("--pad-u-min", type=float, default=None, help="padding toward u-min only (Å)")
    ap.add_argument("--pad-u-max", type=float, default=None, help="padding toward u-max only (Å)")
    ap.add_argument("--pad-v-min", type=float, default=None, help="padding toward v-min only (Å)")
    ap.add_argument("--pad-v-max", type=float, default=None, help="padding toward v-max only (Å)")

    # drawing toggles
    ap.add_argument("--normal", action="store_true", help="draw center point and ±1 Å normal line for each rectangle (default off)")
    ap.add_argument("--horiz-lines", "-hl", type=int, default=5, help="number of interior horizontal lines (0..9, default 5)")

    # CSV / plots
    ap.add_argument("--csv", action="store_true", help="write CSVs for plane normals and adjacent normal angles")
    ap.add_argument("--csv-per-chain", action="store_true", help="also write per-chain CSVs (one set per chain)")
    ap.add_argument("--plot", action="store_true", help="make line plots of index i vs signed angle (SVG, text kept as text)")

    # color-by-secondary-structure PML
    ap.add_argument("--color-ss", action="store_true",
                    help="emit a PML that colors rectangles by secondary structure (HELIX=cyan, SHEET=magenta, other=orange) and forces sticks")

    # force chain
    ap.add_argument("--force-chain", help="override chain ID used in PLN output (single letter, e.g., A)")

    # sticks-only PML
    ap.add_argument("--as-sticks", action="store_true",
                    help="emit a PML that loads the boxes and sets stick representation")
    ap.add_argument("--stick-radius", type=float, default=None,
                    help="optional stick radius to set (e.g., 0.18)")

    # NEW: peptide-link distance cutoff for gap guarding
    ap.add_argument("--cn-cutoff", type=float, default=1.7,
                    help="max C(i)–N(i+1) distance (Å) to treat as a peptide link (default 1.7 Å)")

    args = ap.parse_args()

    if args.from_input_data:
        # Batch mode
        input_dir = os.path.join(".", "input_data")
        patterns = [os.path.join(input_dir, "*.pdb"), os.path.join(input_dir, "*.PDB")]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        files = sorted(set(files))
        if not files:
            print(f"[error] --from-input-data specified but no PDB files found in {input_dir}/")
            sys.exit(2)
        for path in files:
            try:
                process_one(path, args)
            except SystemExit:
                raise
            except Exception as e:
                print(f"[warn] Skipping {path} due to error: {e}")
    else:
        # Single-file mode
        if not args.input:
            print("[error] Please provide an input PDB path, or use --from-input-data to batch process ./input_data/*.pdb")
            sys.exit(2)
        process_one(args.input, args)

if __name__ == "__main__":
    main()