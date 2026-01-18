def bin_to_residue(bin_idx, n_bins, seg_start, L):
    frac = (bin_idx - 0.5) / n_bins
    offset = int(round(frac * (L - 1)))
    return seg_start + offset
