import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Input data
# -------------------------
data = {
    "aa": ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
           "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"],
    "count": [1582,898,859,1173,335,684,1152,1453,422,1017,
              1620,1264,397,815,915,1144,1136,257,620,1437]
}

df = pd.DataFrame(data)

# -------------------------
# Hydrophobicity definition
# -------------------------
hydrophobic = {
    "ALA","VAL","ILE","LEU","MET","PHE","TRP","TYR","CYS","PRO"
}

df["class"] = df["aa"].apply(
    lambda x: "hydrophobic" if x in hydrophobic else "hydrophilic"
)

# Optional: sort by count (comment out if you want alphabetical)
df = df.sort_values("count", ascending=True)

# -------------------------
# Colors per residue
# -------------------------
colors = df["class"].map({
    "hydrophobic": "tab:blue",
    "hydrophilic": "tab:orange"
})

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 4))
plt.bar(df["aa"], df["count"], color=colors)

plt.xlabel("Amino acid")
plt.ylabel("Residue count")
plt.title("Residue counts by amino acid (colored by hydrophobicity)")

# Manual legend (since colors are per-bar)
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="tab:blue", label="Hydrophobic"),
    Patch(color="tab:orange", label="Hydrophilic"),
]
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.show()
