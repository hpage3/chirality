# Amino-Acid–Specific Chirality Repair Costs via Schrödinger Bridge Transport

## Overview

This repository implements a pipeline for quantifying **amino-acid–specific local costs associated with chirality repair** in polypeptide backbones. The approach combines:

* local geometric descriptors (θ-plane orientation),
* local strain descriptors (RMS deviation),
* and global optimal transport (Schrödinger Bridge, SB)

to measure how difficult it is to transform a mixed-chirality ensemble into a homochiral target, both globally and at the level of individual amino acids.

The analysis is **exploratory** and intended to reveal physically interpretable structure–cost relationships, not to make definitive evolutionary claims.

---

## Conceptual Pipeline

The pipeline has four logically distinct stages:

1. **Fingerprint generation** (precomputed)
2. **Schrödinger Bridge transport** (global cost)
3. **Local cost attribution** (per residue / amino acid)
4. **Comparative visualization** (θ vs RMS vs both)

Each stage is modular and reproducible.

# Repository Structure

.chirality_clean/
├── 00_helpers/                # geometry / box helpers
├── 01_extract/                # raw structure extraction
├── 02_boxes/                  # backbone plane normals (CSV)
├── 03_fingerprints/           # θ / RMS fingerprints (CSV)
├── 04_sb/
│   └── output/                # Schrödinger Bridge runs
│       ├── sb_theta_*         # θ-only models
│       ├── sb_rms_*           # RMS-only models
│       └── sb_both_*          # θ + RMS models
├── 05_aa_local/               # amino-acid–level analyses
│   ├── theta/
│   ├── rms/
│   ├── both/
│   └── aa_counts_from_boxes.tsv
├── 06_plots/
│   └── output/                # figures
├── 99_qc/                     # sanity checks / diagnostics
└── scripts/
    ├── run_sb_bridge_chiral.py
    ├── analyze_aa_cost_from_bins.py
    ├── plot_aa_local_cost_comparison.py
    ├── count_residues_dataset.py
    └── plot_aa_cost_vs_frequency.py
	
Create the folder structure by running
	mkdir -p chirality_clean/{00_helpers,01_extract,02_boxes,03_fingerprints,04_sb,05_aa_local,06_plots,99_qc}


# Structure Preparation and Chain Extraction

All downstream analysis assumes a curated set of single protein chains, extracted from PDB or mmCIF files and aligned to a consistent residue numbering scheme.

This step is handled by the 01_extract/ stage.

Input identifiers (01_extract/ids.tsv)

The file 01_extract/ids.tsv defines which structures, chains, and residue offsets are analyzed.

Format
pdb_id    chain    segment_start_1based


Example:

1a2y    	A    		1
1ake   	 	A    		1
1b7f   		A   	 	1
3pgk    	A    		1


Where:

Column						Meaning
	pdb_id					4-character PDB identifier
	chain					Chain ID to extract
	segment_start_1based	First residue (1-based) to include

Why segment_start_1based matters

	Many PDB files:

		start residue numbering at values ≠ 1

		contain signal peptides, tags, or disordered N-termini

		include biologically irrelevant regions

		segment_start_1based allows you to:

		skip non-structural regions,

		align windows across proteins,

		avoid pathological short fragments at chain ends.

This value is critical for reproducible windowing.

1.2 Raw structure inputs

Raw files are placed in:

	01_extract/pdb_raw/


Accepted formats:

.pdb

.cif (mmCIF)

If only CIFs are available, the pipeline caches converted PDBs in:

01_extract/cif_cache/

Chain extraction (extract_chains.py)

The script extract_chains.py reads ids.tsv and produces clean, single-chain PDBs.

Example invocation
python scripts/extract_chains.py \
  --ids 01_extract/ids.tsv \
  --pdb-dir 01_extract/pdb_raw \
  --out-dir 01_extract/chains_pdb

Output
01_extract/chains_pdb/
├── 1a2y_A.pdb
├── 1ake_A.pdb
├── 3pgk_A.pdb
└── ...


Each output file contains:

	a single chain,

	residues starting at segment_start_1based,

	only backbone atoms required for plane construction.

Tuning extraction behavior 

Key tunable behaviors in extract_chains.py include:

(A) Residue filtering

	Skip residues with missing backbone atoms

	Exclude alternate locations (ALTLOC)

	Optionally exclude terminal residues

(B) Chain continuity

	Chains with large residue gaps can be:

		rejected,

		truncated,

		or passed through (configurable)

(C) Minimum length

Chains shorter than the largest window length (L) are automatically excluded downstream, but can also be filtered here for efficiency.

Relationship to downstream steps

01_extract/chains_pdb/*.pdb
⟶ input to plane extraction (02_boxes/)

segment_start_1based
⟶ determines plane indices and window alignment

Incorrect ids.tsv entries will silently produce:

	empty box files,

	short fingerprints,

	invalid SB runs

For this reason, validating ids.tsv is strongly recommended before running the full pipeline.

Recommended validation checks

Before proceeding to 02_boxes/, verify:

ls 01_extract/chains_pdb/*.pdb | wc -l


and visually inspect a few extracted chains:

head 01_extract/chains_pdb/1a2y_A.pdb


Chains should:

	start at the expected residue,

	have continuous backbone atoms,

	contain no extraneous chains or heteroatoms.

Summary

01_extract/ids.tsv is the contract between biological intent and computational analysis.

Careful curation of:

	chain identity,

	residue start,

	and structural completeness

is essential for meaningful chirality cost estimates.

# Plane Extraction

Plane Extraction (02_boxes/)

Each protein backbone is decomposed into local planes derived from backbone atoms.

Output files:

02_boxes/<pdb>_<chain>_boxes_normals.csv


Each row corresponds to a local backbone plane and includes:

residue identities (resname_i, resname_j)

plane normal components

local RMS deviation

These planes are the physical substrate for all downstream analysis.

# Fingerprints (03_fingerprints/)

Fingerprints encode local backbone geometry for fixed-length residue windows.

They are stored in a directory (typically `fingerprints/`) and are treated as **inputs** to all downstream analyses.

Each fingerprint supports three feature modes:

* `theta` — orientational (chiral geometry)
* `rms` — local strain magnitude
* `both` — concatenation of geometry + strain

### Script
python scripts/pdb_fingerprint.py \
  --boxes-dir 02_boxes \
  --out-dir 03_fingerprints \
  --L 12

Example fingerprint file

bin,prefix,chain,theta_pp_deg,box_rms,n_res,L
0,1a2y_A,A,-74.64,0.0069,9,12
1,1a2y_A,A,-114.13,0.0077,9,12
2,1a2y_A,A,-70.05,0.0075,9,12
...
Where:

Column			Meaning
bin	Window 		index along chain
theta_pp_deg	Mean plane orientation (chirality-sensitive)
box_rms			Mean plane RMS deviation
n_res			Number of residues contributing
L				Window length

> **Important:** Fingerprints must already exist before running any SB or AA analysis.
Fingerprints are treated as inputs and are not modified downstream.
Important invariants

Fingerprints do not encode amino acids

Amino-acid identity is reintroduced only during AA cost attribution

Changing fingerprint parameters (L, binning) changes SB behavior and must be documented

---

## Schrödinger Bridge Transport (04_sb/output/)

### Purpose

Compute the minimum-cost stochastic transport from a mixed-chirality ensemble to a homochiral target ensemble.

### Script

```
run_sb_chiral_aa.py
```

### Key parameters

* `--feature-mode {theta,rms,both}`
* `--fingerprint-dir fingerprints`
* `--L1 <int>` — target window length
* `--f0 <float>` — initial mixed-chirality fraction
* `--epsilon <float>` — entropic regularization
* `--out-dir output`
* `--out-prefix <string>`

### Typical usage (looped over seeds)

```
python scripts/run_sb_bridge_chiral.py \
  --feature-mode theta \
  --fingerprint-dir 03_fingerprints \
  --L0 20 --f0 0.5 \
  --L1 12 --f1 0.0 \
  --chirality0 mixed --chirality1 mixed \
  --epsilon 1.0 \
  --seed0 0 --seed1 0 \
  --out-dir 04_sb/output \
  --out-prefix sb_theta_L20_to_L12_s0

```

Repeat for multiple seeds (`s1 … s5`) and for each feature mode:

* `sb_theta_*`
* `sb_rms_*`
* `sb_both_*`

### Outputs (per run)

Each SB run produces a directory:

```
output/<run-prefix>/
```

containing:

* `summary.json` — global transport cost and metadata
* `coupling.npy` — optimal transport plan
* cost matrices used for local attribution

> **Determinism note:** For fixed marginals and cost matrix, SB solutions are deterministic. Multiple seeds are used only for robustness checks.

---

## 3. Amino-Acid Local Cost Attribution (05_aa_local/)

### Purpose

Attribute **global SB transport cost** back to individual residue planes and aggregate by amino acid identity.

This step is essential for residue-level interpretation and **must be run after SB**.

### Script

```
analyze_aa_local_cost.py
```

### Key parameters

* `--output-dir output`
* `--boxes-dir output`
* `--n-bins 10`
* `--run-prefix <prefix>` (critical)
* `--out-dir <output subdir>`

### Important rule

> **Run once per model configuration**, not per seed.

### Correct usage examples

Theta-only example:

python scripts/analyze_aa_cost_from_bins.py \
  --sb_root 04_sb/output \
  --fingerprint_dir 03_fingerprints \
  --boxes_dir 02_boxes \
  --out_dir 05_aa_local/theta \
  --L 12


Run separately for:

theta

rms

both


### Outputs

Each run produces:

* `aa_local_cost_per_run.tsv`
* `aa_local_cost_run_normalized.tsv`

The normalized file reports, per amino acid:

* mean local cost across runs
* deviation from model-wide mean (`delta_vs_global`)

---
### Visualization (Scripts)


##Amino Acid cost comparison

Compare amino-acid–specific local chirality repair costs across feature modes.

### Script

```
plot_aa_local_cost_comparison.py
```

### Expected inputs

The script expects the following directories to exist:

```
output/
  aa_local_cost_theta/
  aa_local_cost_rms/
  aa_local_cost_both/
```

Each must contain `aa_local_cost_run_normalized.tsv`.

### Output

A single comparative plot showing:

* θ-only (geometry-dominated)
* RMS-only (strain-dominated)
* combined (geometry + strain)

with diagnostics to detect accidental data overlap or pipeline errors.

---

## Interpretation Guidelines

* **θ dominates** chirality repair cost.
* **RMS modulates** cost in residue-specific ways.
* `both` should be similar to θ, but not identical.
* Identical θ and `both` curves indicate a pipeline error.

The analysis is **structural and physical**, not evolutionary. Any evolutionary interpretation must be framed cautiously.

---

## Known Limitations

* PDB sampling is limited and biased.
* Costs are relative, not absolute energies.
* Results are exploratory and hypothesis-generating.

---

## Summary

This pipeline provides a reproducible way to:

* quantify chirality repair cost,
* decompose it into geometric and strain contributions,
* and localize those costs to specific amino acids.

It is designed for **clarity, modularity, and interpretability**, not overfitting or speculative claims.
