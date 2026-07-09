"""
split_structs.py — Split simulation results into train/test sets.

Performs a temporal split: the first `ratio` fraction of timesteps goes to
training and the remainder to testing.  Only the state array (x) and aTTC
metric are kept; all other fields are discarded.

Usage (imported):
    from split_structs import split_structs
    data_train, data_test = split_structs(data, ratio=0.8)

Usage (CLI):
    python split_structs.py --mat_file ../Results/attc_train_2D.mat --ratio 0.8
"""

import numpy as np
from auspice_ml_utils import normalize_results


def split_structs(data, ratio=0.8, shuffle=False, seed=42):
    """Split simulation results into training and testing sets.

    Parameters
    ----------
    data : dict
        Results struct from auspice_sim (Python dict) or loaded .mat file.
        Must contain keys: 'x', 'attc', 'params'.
        Arrays may be in MATLAB (HDF5-transposed) or Python layout.
    ratio : float
        Fraction of timesteps allocated to training (0 < ratio < 1).
    shuffle : bool
        If True, shuffle timesteps before splitting (random split).
        If False, perform a temporal split (first ratio fraction = train).
    seed : int
        Random seed for reproducible shuffling (only used if shuffle=True).

    Returns
    -------
    data_train : dict  with keys 'x', 'attc', 'params'
    data_test  : dict  with keys 'x', 'attc', 'params'
    """
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    for key in ("x", "attc", "params"):
        if key not in data:
            raise KeyError(f"data dict missing required key '{key}'")

    # Ensure canonical Python layout (time axis = 0)
    data = normalize_results(data)

    N = data["x"].shape[0]
    n_train = int(np.floor(N * ratio))

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        x_shuffled    = data["x"][idx]
        attc_shuffled = data["attc"][idx]
    else:
        x_shuffled    = data["x"]
        attc_shuffled = data["attc"]

    data_train = {
        "x":      x_shuffled[:n_train],
        "attc":   attc_shuffled[:n_train],
        "params": data["params"],
    }
    data_test = {
        "x":      x_shuffled[n_train:],
        "attc":   attc_shuffled[n_train:],
        "params": data["params"],
    }

    n_test = N - n_train
    split_type = "shuffled" if shuffle else "temporal"
    print(f"Split ({split_type}): {n_train} train + {n_test} test timesteps "
          f"(ratio={ratio}, N={N})")

    # Report finite aTTC counts
    n_fin_train = np.sum(np.isfinite(data_train["attc"]) & (data_train["attc"] > 0))
    n_fin_test  = np.sum(np.isfinite(data_test["attc"])  & (data_test["attc"]  > 0))
    print(f"  Train: {n_fin_train} finite aTTC samples")
    print(f"  Test:  {n_fin_test} finite aTTC samples")

    return data_train, data_test


def split_arrays(*arrays, ratio=0.8, shuffle=True, seed=42):
    """Shuffle and split multiple arrays along axis 0.

    All arrays must have the same length along axis 0.  Useful for
    splitting pre-extracted features, targets, and v_max arrays.

    Parameters
    ----------
    *arrays : ndarray
        One or more arrays to split (must share axis-0 length).
    ratio : float
        Fraction allocated to the first (train) split.
    shuffle : bool
        If True, shuffle before splitting.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    train_arrays : tuple of ndarrays
    test_arrays  : tuple of ndarrays
        Each tuple has the same number of elements as the input arrays.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    N = len(arrays[0])
    for i, a in enumerate(arrays):
        if len(a) != N:
            raise ValueError(f"Array {i} has length {len(a)}, expected {N}")

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        arrays = tuple(a[idx] for a in arrays)

    n_train = int(np.floor(N * ratio))
    train_arrays = tuple(a[:n_train] for a in arrays)
    test_arrays  = tuple(a[n_train:] for a in arrays)

    split_type = "shuffled" if shuffle else "sequential"
    print(f"Split ({split_type}): {n_train} train + {N - n_train} test samples "
          f"(ratio={ratio}, N={N})")

    return train_arrays, test_arrays


# ============================================================================
#                              CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    from auspice_ml_utils import load_mat

    parser = argparse.ArgumentParser(description="Split .mat results into train/test.")
    parser.add_argument("--mat_file", required=True, help="Path to .mat file")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train fraction (default 0.8)")
    args = parser.parse_args()

    data = load_mat(args.mat_file)
    data_train, data_test = split_structs(data, args.ratio)
    print("Done.")
