#!/usr/bin/env python3
"""
run_ml_training_vraw.py — Driver for the "raw velocities" FiLM aTTC NN.

Same training infrastructure as cond1 (1D-conditioning on v_max_j), but
the state features are
    [delta_p,  vel_i,  vel_j]      (9-dim in 3D, 6-dim in 2D)
instead of the original
    [delta_p,  delta_v]            (6-dim in 3D, 4-dim in 2D).

Hypothesis: giving the NN both individual velocities (instead of only
their difference) lets it directly model how the pursuer's reachable set
shrinks/grows relative to the evader's instantaneous speed — information
that's fundamentally underdetermined when only delta_v is provided.

Architecture is unchanged from cond1: FiLM-conditioned MLP with scalar
v_max_j conditioning.  Train / test / export functions are reused from
the cond1 codebase (auspice_ml_train, auspice_ml_test, export_attc_weights);
only the feature extraction and the post-training paper figures + Inf-test
are vraw-specific.

Usage:
    python run_ml_training_vraw.py
    python run_ml_training_vraw.py --loss iw_huber
    python run_ml_training_vraw.py --attc_cap 60 --attc_filter None
    python run_ml_training_vraw.py --tag my_experiment
"""

import argparse
import sys
import numpy as np
import tensorflow as tf

from auspice_ml_utils import load_mat, normalize_results, parse_pair_idx
from auspice_ml_utils_vraw import (
    extract_pair_features_vraw, extract_multi_pair_features_vraw,
)
from auspice_ml_split import split_arrays
from auspice_ml_train import auspice_ml_train     # cond1 routine — input_dim agnostic
from auspice_ml_test  import auspice_ml_test      # cond1 routine — input_dim agnostic

#%% Default Inputs
training_ratio = 0.75
batch_size     = 128
net_layers     = "128,128,64"
learning_rate  = 1e-3
pair_index     = 'all'
attc_cap       = None
attc_filter    = 120.0
log_transform  = False
huber_delta    = 1.0
loss_fun       = 'iw_huber3'
file_name      = None

# v_desired x v_max grid (same as cond2): 4 v_desired × 5 v_max = 20 files.
# Required for vraw — the [vel_i, vel_j] feature parameterization needs
# meaningful variation in v_i across the training set.  Files at fixed
# v_desired (cond1 data) would clamp v_i ~ const and mute the gain.
v_desired_list = [0.15, 0.25, 0.35, 0.50]
v_max_list     = [0.10, 0.30, 0.50, 0.70, 0.90]
T_sim          = 50000


#%% CLI
def _none_or_float(s):
    if s is None or str(s).lower() in ('none', 'null', ''):
        return None
    return float(s)


_parser = argparse.ArgumentParser(description='Train vraw FiLM aTTC NN.')
_parser.add_argument('--attc_cap',    type=_none_or_float, default=None,
                     help='Soft-cap output at this value (None = no cap).')
_parser.add_argument('--attc_filter', type=_none_or_float, default=None,
                     help='Drop training samples with aTTC > this.  '
                          'None = no filter.')
_parser.add_argument('--tag',         type=str, default=None,
                     help='Override auto-generated output dir name.')
_parser.add_argument('--tag_suffix',  type=str, default=None,
                     help='Suffix appended to auto-generated tag.')
_parser.add_argument('--loss',        type=str, default=None,
                     help='Loss function (mse, mae, huber, iw_mse, iw_huber, ...)')
_parser.add_argument('--data_tag',    type=str, default='data1',
                     help=('Filename infix between "training_data_" and '
                           '"3D_2A_..." (e.g. "data1" reads training_data_data1_3D_2A_*.mat). '
                           'Pass an empty string "" to read the legacy '
                           'training_data_3D_2A_* files.'))
_args, _ = _parser.parse_known_args()

if _args.attc_cap is not None:    attc_cap    = _args.attc_cap
if _args.attc_filter is not None: attc_filter = _args.attc_filter
if _args.loss is not None:        loss_fun    = _args.loss

# Resolve data_tag (controls both training-data filename infix AND output
# folder suffix — single source of truth so they can never desync).
#   data_tag = ''    → loads training_data_3D_2A_*           , saves to ..._iw_huber
#   data_tag = 'X'   → loads training_data_X_3D_2A_*         , saves to ..._iw_huber_X
data_tag = (_args.data_tag or "").strip().strip("_")
infix    = f"{data_tag}_" if data_tag else ""

# Auto-tag: nn_bs{BS}_h{HIDDEN}_vraw_{LOSS}[_cap{C}][_{DATA_TAG}][_{SUFFIX}]
if _args.tag is not None:
    file_name = _args.tag
else:
    hidden_part = net_layers.replace(',', '-')
    file_name = f"nn_bs{batch_size}_h{hidden_part}_vraw_{loss_fun}"
    if attc_cap is not None:
        file_name += f"_cap{int(attc_cap)}"
    if data_tag:
        file_name += f"_{data_tag}"
    if _args.tag_suffix:
        file_name += f"_{_args.tag_suffix.lstrip('_')}"

print(f"[run_ml_training_vraw] tag={file_name}   data_tag='{data_tag}'")
print(f"  attc_cap={attc_cap}   attc_filter={attc_filter}   loss={loss_fun}")
print(f"  v_desired_list={v_desired_list}")
print(f"  v_max_list    ={v_max_list}")


#%% Build mat_files list — vdes×vmax grid, using the data_tag infix above
mat_files = []
for v_des in v_desired_list:
    for v_m in v_max_list:
        mat_files.append(
            f"../Results/training_data_{infix}3D_2A_T{T_sim}_"
            f"vdes{v_des:.2f}_vmax{v_m:.2f}.mat"
        )
print(f"  Expecting {len(mat_files)} mat files "
      f"({len(v_desired_list)} v_des x {len(v_max_list)} v_max, data_tag='{data_tag}')")


#%% Load + extract vraw features (9-dim in 3D)
all_feat, all_tgt, all_v_max_j = [], [], []
dim = None
n_loaded = 0

for path in mat_files:
    print(f"\nLoading {path} ...")
    try:
        data = load_mat(str(path))
        data = normalize_results(data)
    except Exception as e:
        print(f"  SKIP (failed to load): {e}")
        continue

    params = data["params"]
    d = int(params["dim"])
    if dim is None:
        dim = d
    na = int(params["num_agents"])
    v_max = float(params.get("v_max_attc", params.get("v_max", 0)))
    v_des = float(params.get("v_desired", 0.25))
    print(f"  {d}D, {na} agents, {data['x'].shape[0]} timesteps, "
          f"v_des={v_des:.3f}, v_max={v_max:.3f}")

    pair_indices = parse_pair_idx(pair_index, na)
    if len(pair_indices) == 1:
        feat, tgt = extract_pair_features_vraw(
            data, pair_indices[0],
            attc_cap=attc_cap, attc_filter=attc_filter)
    else:
        feat, tgt = extract_multi_pair_features_vraw(
            data, pair_indices,
            attc_cap=attc_cap, attc_filter=attc_filter)

    all_feat.append(feat)
    all_tgt.append(tgt)
    all_v_max_j.append(np.full(feat.shape[0], v_max, dtype=np.float32))
    n_loaded += 1

if n_loaded == 0:
    print("\nERROR: no training files loaded.  Aborting.")
    sys.exit(1)

all_features = np.concatenate(all_feat,    axis=0)
all_targets  = np.concatenate(all_tgt,     axis=0)
all_v_max_j  = np.concatenate(all_v_max_j, axis=0)

print(f"\nTotal samples: {len(all_targets)}  (from {n_loaded} files)")
print(f"  features.shape = {all_features.shape}    "
      f"(expected {len(all_targets)}x{3*dim} for vraw)")
print(f"  v_max_j: min={all_v_max_j.min():.3f}  "
      f"max={all_v_max_j.max():.3f}  mean={all_v_max_j.mean():.3f}")


#%% Train/test split
(train_features, train_targets, train_vmax), \
(test_features,  test_targets,  test_vmax) = \
    split_arrays(all_features, all_targets, all_v_max_j,
                 ratio=training_ratio, shuffle=True, seed=42)


#%% Training (uses cond1's auspice_ml_train — input_dim auto-detected)
print("\n" + "─" * 50)
print("  TRAIN vraw FiLM MODEL")
print("─" * 50)

model, history, config = auspice_ml_train(
    data=None,
    loss=loss_fun,
    epochs=400,
    val_split=0.1,
    batch_size=batch_size,
    hidden=net_layers,
    lr=learning_rate,
    patience=20,
    pair_idx=pair_index,
    shuffle=True,
    tag=file_name,
    film=True,
    v_max_array=train_vmax,
    features_precomputed=train_features,
    targets_precomputed=train_targets,
    attc_cap=attc_cap,
    attc_filter=attc_filter,
    log_transform=log_transform,
    huber_delta=huber_delta,
    mat_files=mat_files,
)


#%% Test on held-out split
print("\n" + "─" * 50)
print("  TEST vraw MODEL")
print("─" * 50)

save_dir = config.get("save_dir", None)
test_results = auspice_ml_test(
    data=None,
    model=model,
    config=config,
    save_dir=save_dir,
    v_max_array=test_vmax,
    features_precomputed=test_features,
    targets_precomputed=test_targets,
    cap_pred=attc_filter,
)


#%% Fix config.json: cond1's training routine sets `dim = features.shape[1]//2`
#   which is wrong for vraw (9//2=4 instead of 3).  Overwrite before export
#   so the saved .mat carries the correct simulation dim.
import json, os
cfg_path = os.path.join(save_dir, 'config.json')
with open(cfg_path) as _f:
    _cfg = json.load(_f)
if _cfg.get('dim') != dim:
    print(f"\n[run_ml_training_vraw] correcting config.json: dim "
          f"{_cfg.get('dim')} → {dim}  (was set to features.shape[1]//2 by cond1 train)")
    _cfg['dim'] = int(dim)
    with open(cfg_path, 'w') as _f:
        json.dump(_cfg, _f, indent=2)


#%% Export MATLAB weights (uses cond1's exporter — same architecture)
print("\n" + "─" * 50)
print("  EXPORT MATLAB WEIGHTS (vraw)")
print("─" * 50)

from export_attc_weights import export_weights
export_weights(save_dir)


#%% Paper figures + Inf-prediction test
print("\n" + "─" * 50)
print("  PAPER ML FIGURES (vraw)")
print("─" * 50)

try:
    from plot_paper_ml_results_vraw import make_paper_plots_vraw
    make_paper_plots_vraw(
        save_dir,
        mat_files=mat_files,
        attc_filter=attc_filter if attc_filter is not None else 120.0,
        training_ratio=training_ratio,
        seed=42,
    )
except Exception as e:
    print(f"  WARNING: paper plots failed: {e}")

print("\n" + "─" * 50)
print("  INF-aTTC PREDICTION TEST (vraw)")
print("─" * 50)

try:
    from test_nn_inf_predictions_vraw import run_inf_test_vraw
    run_inf_test_vraw(
        save_dir,
        mat_files=mat_files,
        attc_filter_ref=attc_filter if attc_filter is not None else 120.0,
    )
except Exception as e:
    print(f"  WARNING: inf-prediction test failed: {e}")
