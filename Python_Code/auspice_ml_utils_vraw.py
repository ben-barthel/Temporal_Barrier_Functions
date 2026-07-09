"""
auspice_ml_utils_vraw.py — Feature extractor for the "raw velocities"
training pipeline.

Differences from auspice_ml_utils.py (cond1):
  - extract_pair_features_vraw / extract_multi_pair_features_vraw return
    9-dim features in 3D (6-dim in 2D) of the form
        [delta_p, vel_i, vel_j]
    where delta_p = pos_i - pos_j (i-j convention, matching MATLAB),
    and vel_i / vel_j are the i/j agents' Cartesian velocities.

The model architecture is unchanged from cond1: FiLM-conditioned MLP
with scalar v_max_j conditioning, built via auspice_ml_utils.build_model
with input_dim=9 (3D) / input_dim=6 (2D).  The cond1 train / test /
export routines work as-is on the wider features.
"""

import numpy as np

# Re-export unchanged helpers
from auspice_ml_utils import (
    load_mat, load_mat_v73, load_mat_scipy, load_keras_model,
    normalize_results, state_to_pos_vel, parse_pair_idx,
    _get_soft_cap_layer_class, _soft_cap_layer, _film_bias_init,
    compute_ow_weights, compute_iw_weights, parse_loss_name, get_loss,
    build_model,                # reused (input_dim auto-detected from data)
    RESULTS_DIR,
)

_get_soft_cap_layer_class()


# ============================================================================
#                  FEATURE EXTRACTION  ([dp, vel_i, vel_j])
# ============================================================================

def extract_pair_features_vraw(data, pair_idx=0, attc_cap=None,
                                attc_filter=None):
    """Extract ([delta_p, vel_i, vel_j], targets) for a single agent pair.

    Returns
    -------
    features : ndarray (N_valid, 3*dim)
        Columns: [delta_p (dim), vel_i (dim), vel_j (dim)]
    targets  : ndarray (N_valid,)
        Finite, positive aTTC values (s).
    """
    params = data["params"]
    dim = int(params["dim"])
    na  = int(params["num_agents"])
    x   = data["x"]       # (N, na, sd)
    attc = data["attc"]   # (N, num_pairs)

    N = x.shape[0]

    # Map pair_idx → (agent_i, agent_j)
    pair = 0
    agent_i, agent_j = 0, 1
    for i in range(na):
        for j in range(i + 1, na):
            if pair == pair_idx:
                agent_i, agent_j = i, j
            pair += 1

    pos_i, vel_i = state_to_pos_vel(x[:, agent_i, :], dim)
    pos_j, vel_j = state_to_pos_vel(x[:, agent_j, :], dim)

    delta_p = pos_i - pos_j   # (N, dim) — i-j convention

    # Concatenate raw velocities — the NN sees both speeds explicitly,
    # not just their difference.
    features = np.hstack([delta_p, vel_i, vel_j])  # (N, 3*dim)
    targets  = attc[:, pair_idx].copy()

    # --- Cap / clamp / filter (mirrors extract_pair_features) ---
    n_inf_capped = 0
    n_fin_capped = 0
    if attc_cap is not None:
        inf_mask = np.isinf(targets) & (targets > 0)
        n_inf_capped = int(inf_mask.sum())
        targets[inf_mask] = attc_cap
        fin_above_mask = np.isfinite(targets) & (targets > attc_cap)
        n_fin_capped = int(fin_above_mask.sum())
        targets[fin_above_mask] = attc_cap

    valid = np.isfinite(targets) & ~np.isnan(targets) & (targets > 0)

    n_filtered = 0
    if attc_filter is not None:
        above = valid & (targets > attc_filter)
        n_filtered = int(above.sum())
        valid = valid & (targets <= attc_filter)

    cap_str = ""
    if attc_cap is not None:
        cap_str = (f", {n_inf_capped} Inf→cap, "
                   f"{n_fin_capped} finite clamped to cap={attc_cap:.0f}s")
    filt_str = ""
    if attc_filter is not None:
        filt_str = f", {n_filtered} filtered (>{attc_filter:.0f}s)"
    print(f"  Pair ({agent_i+1},{agent_j+1}): {N} timesteps, "
          f"{int(valid.sum())} valid aTTC samples "
          f"({100*valid.mean():.1f}%){cap_str}{filt_str}")

    return (features[valid].astype(np.float32),
            targets[valid].astype(np.float32))


def extract_multi_pair_features_vraw(data, pair_indices, attc_cap=None,
                                      attc_filter=None):
    """Like extract_multi_pair_features but with the vraw 9-dim features."""
    all_feat, all_tgt = [], []
    for p in pair_indices:
        f, t = extract_pair_features_vraw(
            data, p, attc_cap=attc_cap, attc_filter=attc_filter)
        all_feat.append(f)
        all_tgt.append(t)

    return (np.concatenate(all_feat, axis=0),
            np.concatenate(all_tgt,  axis=0))


def extract_all_pair_features_vraw(data, attc_cap=None, attc_filter=None):
    """vraw analog of extract_all_pair_features."""
    params = data["params"]
    na = int(params["num_agents"])
    num_pairs = na * (na - 1) // 2
    return extract_multi_pair_features_vraw(
        data, list(range(num_pairs)),
        attc_cap=attc_cap, attc_filter=attc_filter)
