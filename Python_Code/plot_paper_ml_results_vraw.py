#!/usr/bin/env python3
"""
plot_paper_ml_results_vraw.py — Paper ML figures for the "raw velocities"
FiLM model (features = [delta_p, vel_i, vel_j], scalar v_max_j conditioning).

Mirrors plot_paper_ml_results_cond2.py but with the simpler 1D-cond
architecture and the vraw feature extractor.

Wraps everything in a callable function `make_paper_plots_vraw(model_dir,
save_dir=None)` so it can be invoked at the end of run_ml_training_vraw.

Usage:
    python plot_paper_ml_results_vraw.py
    python plot_paper_ml_results_vraw.py --model_dir ../Results/nn_bs128_h128-128-64_vraw_iw_huber

    # programmatic
    from plot_paper_ml_results_vraw import make_paper_plots_vraw
    make_paper_plots_vraw('../Results/nn_bs128_h128-128-64_vraw_iw_huber')
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3d projection)
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from auspice_ml_utils import (load_mat, normalize_results, parse_pair_idx,
                               _get_soft_cap_layer_class)
from auspice_ml_utils_vraw import extract_multi_pair_features_vraw
from auspice_ml_split import split_arrays

_get_soft_cap_layer_class()


def make_paper_plots_vraw(model_dir, save_dir=None,
                           mat_files=None,
                           attc_filter=120.0,
                           training_ratio=0.75,
                           seed=42):
    """Generate the 5 ML-section paper figures for a vraw model.

    Parameters
    ----------
    model_dir : str
        Directory containing model.h5 + config.json.
    save_dir : str, optional
        Where to write figure files.  Default: <model_dir>/../paper_ml_figures_vraw/
    mat_files : list of str, optional
        Training data files.  Default: 4 v_des × 5 v_max grid.
    attc_filter : float
        Drop training samples with aTTC > this value (matches training).
    training_ratio : float
        Reproduces training split (must match value used during training).
    seed : int
        RNG seed for split (must match training).
    """
    if save_dir is None:
        save_dir = os.path.join(model_dir, 'paper_ml_figures')
    os.makedirs(save_dir, exist_ok=True)

    if mat_files is None:
        # cond2 vdes×vmax grid — must match training data for split reproducibility.
        v_desired_list = [0.15, 0.25, 0.35, 0.50]
        v_max_list     = [0.10, 0.30, 0.50, 0.70, 0.90]
        T_sim          = 50000
        mat_files = [
            f"../Results/training_data_3D_2A_T{T_sim}_vdes{v:.2f}_vmax{m:.2f}.mat"
            for v in v_desired_list for m in v_max_list
        ]

    pair_index = 'all'

    # ---- Plot conventions ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
    })
    fn = 'Times New Roman'
    fs_label, fs_title, fs_tick, fs_legend, lw = 13, 14, 11, 11, 2

    COLOR_MAE = [0.00, 0.45, 0.74]
    COLOR_MPE = [0.85, 0.33, 0.10]

    print(f"[plot_paper_ml_results_vraw] model_dir={model_dir}")
    print(f"  save_dir={save_dir}")

    # ---- Load model + 1D-cond norm_stats ----
    model_path = os.path.join(model_dir, 'model.h5')
    cfg_path   = os.path.join(model_dir, 'config.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    with open(cfg_path) as f:
        cfg = json.load(f)
    ns = cfg['norm_stats']
    feat_mean = np.array(ns['feat_mean'], dtype=np.float32)
    feat_std  = np.array(ns['feat_std'],  dtype=np.float32)
    v_max_min = float(ns['v_max_min'])
    v_max_max = float(ns['v_max_max'])

    rng_v = max(v_max_max - v_max_min, 1e-10)

    def _normalize_vmax(v_raw):
        if v_max_max - v_max_min < 1e-10:
            return np.full((len(v_raw), 1), 0.5, dtype=np.float32)
        return ((v_raw - v_max_min) / rng_v).reshape(-1, 1).astype(np.float32)

    def _predict(features, v_raw):
        X  = ((features - feat_mean) / feat_std).astype(np.float32)
        Xv = _normalize_vmax(v_raw)
        return model.predict([X, Xv], batch_size=4096, verbose=0).ravel()

    # ---- Load filtered data (Inf dropped, true ≤ attc_filter) for Figs 1, 3-5 ----
    print("\nLoading filtered training data ...")
    all_feat, all_tgt, all_vmj = [], [], []
    for path in mat_files:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        data = load_mat(str(path))
        data = normalize_results(data)
        params = data['params']
        na = int(params['num_agents'])
        v_max = float(params.get('v_max_attc', params.get('v_max', 0)))
        pair_indices = parse_pair_idx(pair_index, na)
        feat, tgt = extract_multi_pair_features_vraw(
            data, pair_indices, attc_cap=None, attc_filter=attc_filter)
        all_feat.append(feat)
        all_tgt.append(tgt)
        all_vmj.append(np.full(feat.shape[0], v_max, dtype=np.float32))

    all_features = np.concatenate(all_feat, axis=0)
    all_targets  = np.concatenate(all_tgt,  axis=0)
    all_vmj      = np.concatenate(all_vmj,  axis=0)

    (train_feat, train_tgt, train_vmax), \
    (test_feat,  test_tgt,  test_vmax) = split_arrays(
        all_features, all_targets, all_vmj,
        ratio=training_ratio, shuffle=True, seed=seed)
    print(f"Train samples: {len(train_tgt)}   Test samples: {len(test_tgt)}")

    # ---- Load viz data (Inf samples capped at 120 s) for Fig 2 ----
    VIZ_CAP = 120.0
    print(f"\nLoading viz data (attc_cap={VIZ_CAP}, attc_filter=None) for Fig 2 ...")
    viz_feat_list, viz_tgt_list, viz_vmj_list = [], [], []
    for path in mat_files:
        if not os.path.exists(path):
            continue
        data = load_mat(str(path))
        data = normalize_results(data)
        params = data['params']
        na = int(params['num_agents'])
        v_max = float(params.get('v_max_attc', params.get('v_max', 0)))
        pair_indices = parse_pair_idx(pair_index, na)
        feat_v, tgt_v = extract_multi_pair_features_vraw(
            data, pair_indices, attc_cap=VIZ_CAP, attc_filter=None)
        viz_feat_list.append(feat_v)
        viz_tgt_list.append(tgt_v)
        viz_vmj_list.append(np.full(feat_v.shape[0], v_max, dtype=np.float32))
    viz_features = np.concatenate(viz_feat_list, axis=0)
    viz_targets  = np.concatenate(viz_tgt_list,  axis=0)
    viz_vmax     = np.concatenate(viz_vmj_list,  axis=0)

    # ---- Predictions ----
    print("\nPredicting on train + test + viz ...")
    pred_train = _predict(train_feat,    train_vmax)
    pred_test  = _predict(test_feat,     test_vmax)
    pred_viz   = _predict(viz_features,  viz_vmax)

    # ---- Metrics ----
    T_WIDE  = np.arange(1, 111, 1, dtype=float)
    MPE_EPS = 0.5

    def _cond_mae(true, pred, thresholds):
        out = np.full(len(thresholds), np.nan)
        for j, T in enumerate(thresholds):
            m = true < T
            if m.sum() > 0:
                out[j] = float(np.mean(np.abs(pred[m] - true[m])))
        return out

    def _cond_mpe(true, pred, thresholds, eps=MPE_EPS):
        out = np.full(len(thresholds), np.nan)
        for j, T in enumerate(thresholds):
            m = (true < T) & (true > eps)
            if m.sum() > 0:
                out[j] = float(100.0 * np.mean(np.abs(pred[m] - true[m]) / true[m]))
        return out

    mae_train = _cond_mae(train_tgt, pred_train, T_WIDE)
    mae_test  = _cond_mae(test_tgt,  pred_test,  T_WIDE)
    mpe_train = _cond_mpe(train_tgt, pred_train, T_WIDE)
    mpe_test  = _cond_mpe(test_tgt,  pred_test,  T_WIDE)
    print(f"\nMAE (all):  train={np.mean(np.abs(pred_train-train_tgt)):.3f}  "
          f"test={np.mean(np.abs(pred_test-test_tgt)):.3f}")

    # =========================================================================
    #  Fig 1: Conditional MAE + MPE vs threshold
    # =========================================================================
    print("\nFig 1: conditional MAE + MPE ...")
    fig1, ax_mae = plt.subplots(figsize=(7.5, 5.0))
    ax_mpe = ax_mae.twinx()

    h_mae_test,  = ax_mae.plot(T_WIDE, mae_test,  '-',  color=COLOR_MAE,
                                linewidth=lw, label='MAE (test)')
    h_mae_train, = ax_mae.plot(T_WIDE, mae_train, '--o', color=COLOR_MAE,
                                linewidth=lw, markersize=5, markevery=10,
                                markerfacecolor='white', label='MAE (train)')
    h_mpe_test,  = ax_mpe.plot(T_WIDE, mpe_test,  '-',  color=COLOR_MPE,
                                linewidth=lw, label='MPE (test)')
    h_mpe_train, = ax_mpe.plot(T_WIDE, mpe_train, '--o', color=COLOR_MPE,
                                linewidth=lw, markersize=5, markevery=10,
                                markerfacecolor='white', label='MPE (train)')

    ax_mae.set_xlabel(r'Threshold $T$  [s]   (samples with true $\tau^{*} < T$)',
                      fontname=fn, fontsize=fs_label)
    ax_mae.set_ylabel('Mean Absolute Error [s]',
                      fontname=fn, fontsize=fs_label, color=COLOR_MAE)
    ax_mae.tick_params(axis='y', labelsize=fs_tick, colors=COLOR_MAE)
    ax_mae.tick_params(axis='x', labelsize=fs_tick)
    ax_mpe.set_ylabel('Mean Percentage Error [%]',
                      fontname=fn, fontsize=fs_label, color=COLOR_MPE)
    ax_mpe.tick_params(axis='y', labelsize=fs_tick, colors=COLOR_MPE)
    ax_mae.grid(True, alpha=0.3)

    lines = [h_mae_test, h_mae_train, h_mpe_test, h_mpe_train]
    ax_mae.legend(lines, [ln.get_label() for ln in lines],
                  fontsize=fs_legend, loc='lower left', framealpha=0.95)

    fig1.tight_layout()
    p = os.path.join(save_dir, 'fig1_conditional_mae_mpe.png')
    fig1.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    fig1.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig1);  print(f"  Saved: {p}")

    # =========================================================================
    #  Fig 2: 3D ridgeline (true & predicted) over (aTTC, v_max)
    # =========================================================================
    print("\nFig 2: 3D ridgeline ...")
    attc_edges   = np.linspace(0, 60, 121)
    attc_centers = 0.5 * (attc_edges[:-1] + attc_edges[1:])
    viz_vmax_r   = np.round(viz_vmax, decimals=3)
    vmax_levels  = np.array(sorted(np.unique(viz_vmax_r)))

    def _pdf_matrix(targets, vmax_round, vmax_bins):
        M = np.zeros((len(vmax_bins), len(attc_centers)))
        for i, v in enumerate(vmax_bins):
            m = vmax_round == v
            if m.sum() == 0:
                continue
            M[i], _ = np.histogram(targets[m], bins=attc_edges, density=True)
        return M

    pred_viz_capped = np.minimum(pred_viz, VIZ_CAP)
    P_true = _pdf_matrix(viz_targets,     viz_vmax_r, vmax_levels)
    P_pred = _pdf_matrix(pred_viz_capped, viz_vmax_r, vmax_levels)
    pdf_zmax = float(max(P_true.max(), P_pred.max()))
    v_lo, v_hi = float(vmax_levels.min()), float(vmax_levels.max())
    def _vmax_color(v):
        t = (v - v_lo) / max(v_hi - v_lo, 1e-12)
        return plt.cm.cool(t)

    fig2 = plt.figure(figsize=(10, 7.5))
    ax = fig2.add_subplot(111, projection='3d')
    h_true = h_pred = None
    for i, v in enumerate(vmax_levels):
        x = attc_centers
        y = np.full_like(x, v)
        c = _vmax_color(v)
        ln_pred, = ax.plot(x, y, P_pred[i], color=c, linewidth=2.0)
        ln_true, = ax.plot(x, y, P_true[i], color='black', linewidth=2.0, alpha=0.85)
        if h_true is None:
            h_true, h_pred = ln_true, ln_pred
    ax.set_xlim(0, 60); ax.set_ylim(v_lo - 0.05, v_hi + 0.05); ax.set_zlim(0, pdf_zmax * 1.05)
    ax.set_xlabel(r'aTTC  $\tau^{*}$  [s]', fontname=fn, fontsize=fs_label, labelpad=8)
    ax.set_ylabel(r'$v_{\mathrm{max},j}$  [km/s]', fontname=fn, fontsize=fs_label, labelpad=8)
    ax.set_zlabel('PDF', fontname=fn, fontsize=fs_label, labelpad=6)
    ax.set_yticks(vmax_levels)
    ax.set_yticklabels([f'{v:.2f}' for v in vmax_levels], fontname=fn, fontsize=fs_tick - 1)
    ax.tick_params(axis='x', labelsize=fs_tick - 1)
    ax.tick_params(axis='z', labelsize=fs_tick - 1)
    # journal figures: caption supplied externally, no in-figure title
    # ax.set_title(r'$p(\tau^{*}\,|\,v_{\max,j})$:  true (black) vs NN (colored)',
    #              fontname=fn, fontsize=fs_title)
    ax.legend([h_true, h_pred],
              [r'True  $\tau^{*}$', r'NN  $\hat{\tau}^{*}$'],
              fontsize=fs_legend, loc='upper right')
    ax.view_init(elev=22, azim=-65)
    fig2.tight_layout()
    p = os.path.join(save_dir, 'fig2_pdf_ridgeline_vmax.png')
    fig2.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    fig2.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig2);  print(f"  Saved: {p}")

    # =========================================================================
    #  Fig 3: scatter true vs predicted, color = v_max
    # =========================================================================
    print("\nFig 3: scatter ...")
    fig3, ax3 = plt.subplots(figsize=(7, 6.0))
    norm = Normalize(vmin=test_vmax.min(), vmax=test_vmax.max())
    sc = ax3.scatter(test_tgt, pred_test, c=test_vmax, cmap='cool',
                     s=3, alpha=0.25, norm=norm, rasterized=True)
    lo, hi = 0, 60
    ax3.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, alpha=0.8,
             label=r'$\hat{\tau}^{*} = \tau^{*}$')
    ax3.set_xlabel(r'True aTTC  $\tau^{*}$  [s]', fontname=fn, fontsize=fs_label)
    ax3.set_ylabel(r'NN prediction  $\hat{\tau}^{*}$  [s]', fontname=fn, fontsize=fs_label)
    # journal figures: caption supplied externally, no in-figure title
    # ax3.set_title(r'True vs predicted aTTC — colored by $v_{\mathrm{max},j}$',
    #               fontname=fn, fontsize=fs_title)
    ax3.set_xlim(lo, hi); ax3.set_ylim(lo, hi); ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=fs_legend, loc='upper left')
    ax3.tick_params(labelsize=fs_tick)
    cb = fig3.colorbar(sc, ax=ax3, shrink=0.85, pad=0.02)
    cb.set_label(r'$v_{\mathrm{max},j}$  [km/s]', fontname=fn, fontsize=fs_label)
    cb.ax.tick_params(labelsize=fs_tick)
    cb.solids.set_alpha(1.0)
    fig3.tight_layout()
    p = os.path.join(save_dir, 'fig3_true_vs_pred_scatter.png')
    fig3.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    fig3.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig3);  print(f"  Saved: {p}")

    # =========================================================================
    #  Fig 4: signed error PDF by v_max
    # =========================================================================
    print("\nFig 4: signed error PDF ...")
    VMAX_FIG4 = [0.30, 0.50, 0.70, 0.90]
    err_test_signed = pred_test - test_tgt

    def _skewness(x):
        mu = float(np.mean(x));  sigma = float(np.std(x))
        if sigma < 1e-12: return 0.0
        return float(np.mean(((x - mu) / sigma) ** 3))

    v_lo4, v_hi4 = min(VMAX_FIG4), max(VMAX_FIG4)
    def _vc4(v):
        t = (v - v_lo4) / max(v_hi4 - v_lo4, 1e-12)
        return plt.cm.cool(t)

    half_width = 0.0
    for v in VMAX_FIG4:
        m = np.abs(test_vmax - v) < 1e-3
        if m.sum() > 0:
            half_width = max(half_width,
                             float(np.percentile(np.abs(err_test_signed[m]), 95)))
    half_width = max(half_width, 1.0)

    n_bins = 80
    edges  = np.linspace(-half_width, half_width, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig4, ax4 = plt.subplots(figsize=(7.5, 5.0))
    for v in VMAX_FIG4:
        m = np.abs(test_vmax - v) < 1e-3
        if m.sum() == 0:
            print(f"  skipping v_max={v:.2f}: no test samples")
            continue
        e = err_test_signed[m]
        counts, _ = np.histogram(e, bins=edges, density=True)
        bias_v = float(np.mean(e));  skew_v = _skewness(e)
        ax4.plot(centers, counts, '-', color=_vc4(v), linewidth=lw,
                 label=(fr'$v_{{\mathrm{{max}},j}} = {v:.2f}$:  '
                        fr'bias $= {bias_v:+.2f}$ s,  '
                        fr'skew $= {skew_v:+.2f}$'))
    ax4.axvline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$\hat{\tau}^{*} - \tau^{*}$  [s]',
                   fontname=fn, fontsize=fs_label)
    ax4.set_ylabel('PDF', fontname=fn, fontsize=fs_label)
    # journal figures: caption supplied externally, no in-figure title
    # ax4.set_title(r'Signed prediction error by $v_{\mathrm{max},j}$',
    #               fontname=fn, fontsize=fs_title)
    ax4.set_xlim(-half_width, half_width)
    ax4.grid(True, which='both', alpha=0.3)
    ax4.legend(fontsize=fs_legend, loc='lower left', framealpha=0.95)
    ax4.tick_params(labelsize=fs_tick)
    fig4.tight_layout()
    p = os.path.join(save_dir, 'fig4_signed_error_pdf_by_vmax.png')
    fig4.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    fig4.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig4);  print(f"  Saved: {p}")

    # =========================================================================
    #  Fig 5: signed percentage error PDF by v_max
    # =========================================================================
    print("\nFig 5: signed percentage error PDF ...")
    valid_pct = test_tgt > MPE_EPS
    pct_err = np.full_like(test_tgt, np.nan, dtype=np.float32)
    pct_err[valid_pct] = 100.0 * (pred_test[valid_pct] - test_tgt[valid_pct]) / test_tgt[valid_pct]

    half_width_pct = 0.0
    for v in VMAX_FIG4:
        m = (np.abs(test_vmax - v) < 1e-3) & valid_pct
        if m.sum() > 0:
            half_width_pct = max(half_width_pct,
                                  float(np.percentile(np.abs(pct_err[m]), 95)))
    half_width_pct = max(half_width_pct, 10.0)

    edges_pct  = np.linspace(-half_width_pct, half_width_pct, 81)
    centers_pct = 0.5 * (edges_pct[:-1] + edges_pct[1:])

    fig5, ax5 = plt.subplots(figsize=(7.5, 5.0))
    for v in VMAX_FIG4:
        m = (np.abs(test_vmax - v) < 1e-3) & valid_pct
        if m.sum() == 0:
            continue
        e = pct_err[m]
        counts, _ = np.histogram(e, bins=edges_pct, density=True)
        bias_v = float(np.mean(e));  skew_v = _skewness(e)
        ax5.plot(centers_pct, counts, '-', color=_vc4(v), linewidth=lw,
                 label=(fr'$v_{{\mathrm{{max}},j}} = {v:.2f}$:  '
                        fr'bias $= {bias_v:+.1f}$\%,  '
                        fr'skew $= {skew_v:+.2f}$'))
    ax5.axvline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
    ax5.set_yscale('log')
    ax5.set_xlabel(r'$100 \times (\hat{\tau}^{*} - \tau^{*}) / \tau^{*}$  [\%]',
                   fontname=fn, fontsize=fs_label)
    ax5.set_ylabel('PDF', fontname=fn, fontsize=fs_label)
    # journal figures: caption supplied externally, no in-figure title
    # ax5.set_title(r'Signed percentage error by $v_{\mathrm{max},j}$',
    #               fontname=fn, fontsize=fs_title)
    ax5.set_xlim(-half_width_pct, half_width_pct)
    ax5.grid(True, which='both', alpha=0.3)
    ax5.legend(fontsize=fs_legend, loc='lower left', framealpha=0.95)
    ax5.tick_params(labelsize=fs_tick)
    fig5.tight_layout()
    p = os.path.join(save_dir, 'fig5_signed_pct_error_pdf_by_vmax.png')
    fig5.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    fig5.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig5);  print(f"  Saved: {p}")

    # =========================================================================
    #  Figs 6 & 7: same as fig 4 & 5 but restricted to deployment-critical
    #              small-τ samples  (true τ* < TAU_SMALL_THRESH s).  Heavy
    #              tail samples are dropped before computing bias / skew.
    # =========================================================================
    TAU_SMALL_THRESH = 30.0
    small_mask = test_tgt < TAU_SMALL_THRESH
    n_small = int(small_mask.sum())
    print(f"\nFigs 6 & 7: restricting to τ_true < {TAU_SMALL_THRESH} s "
          f"({n_small} / {len(test_tgt)} test samples = {100*n_small/max(len(test_tgt),1):.1f}%)")

    if n_small < 50:
        print(f"  [skip] only {n_small} small-τ test samples — too few for fig 6 / 7")
    else:
        tgt_s   = test_tgt[small_mask]
        pred_s  = pred_test[small_mask]
        vmax_s  = test_vmax[small_mask]
        err_s   = pred_s - tgt_s

        # ---- Fig 6: signed error PDF by v_max, conditional on τ < 30 s ----
        print(f"\nFig 6: signed error PDF | τ_true < {TAU_SMALL_THRESH:.0f} s ...")
        half_width6 = 0.0
        for v in VMAX_FIG4:
            m = np.abs(vmax_s - v) < 1e-3
            if m.sum() > 0:
                half_width6 = max(half_width6,
                                  float(np.percentile(np.abs(err_s[m]), 95)))
        half_width6 = max(half_width6, 1.0)

        edges6  = np.linspace(-half_width6, half_width6, n_bins + 1)
        centers6 = 0.5 * (edges6[:-1] + edges6[1:])

        fig6, ax6 = plt.subplots(figsize=(7.5, 5.0))
        for v in VMAX_FIG4:
            m = np.abs(vmax_s - v) < 1e-3
            if m.sum() == 0:
                print(f"  skipping v_max={v:.2f}: no small-τ test samples")
                continue
            e = err_s[m]
            counts, _ = np.histogram(e, bins=edges6, density=True)
            bias_v = float(np.mean(e));  skew_v = _skewness(e)
            ax6.plot(centers6, counts, '-', color=_vc4(v), linewidth=lw,
                     label=(fr'$v_{{\mathrm{{max}},j}} = {v:.2f}$:  '
                            fr'bias $= {bias_v:+.2f}$ s,  '
                            fr'skew $= {skew_v:+.2f}$  '
                            fr'($n={int(m.sum())}$)'))
        ax6.axvline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
        ax6.set_yscale('log')
        ax6.set_xlabel(r'$\hat{\tau}^{*} - \tau^{*}$  [s]',
                       fontname=fn, fontsize=fs_label)
        ax6.set_ylabel('PDF', fontname=fn, fontsize=fs_label)
        # journal figures: caption supplied externally, no in-figure title
        # ax6.set_title(rf'Signed prediction error by $v_{{\mathrm{{max}},j}}$  '
        #               rf'($\tau^{{*}} < {int(TAU_SMALL_THRESH)}$ s)',
        #               fontname=fn, fontsize=fs_title)
        ax6.set_xlim(-half_width6, half_width6)
        ax6.grid(True, which='both', alpha=0.3)
        ax6.legend(fontsize=fs_legend, loc='lower left', framealpha=0.95)
        ax6.tick_params(labelsize=fs_tick)
        fig6.tight_layout()
        p = os.path.join(save_dir, 'fig6_signed_error_pdf_by_vmax_tau_lt_30.png')
        fig6.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        fig6.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig6);  print(f"  Saved: {p}")

        # ---- Fig 7: signed % error PDF by v_max, conditional on τ < 30 s ---
        print(f"\nFig 7: signed % error PDF | τ_true < {TAU_SMALL_THRESH:.0f} s ...")
        valid_pct_s = tgt_s > MPE_EPS
        pct_err_s = np.full_like(tgt_s, np.nan, dtype=np.float32)
        pct_err_s[valid_pct_s] = 100.0 * (pred_s[valid_pct_s] - tgt_s[valid_pct_s]) / tgt_s[valid_pct_s]

        half_width_pct7 = 0.0
        for v in VMAX_FIG4:
            m = (np.abs(vmax_s - v) < 1e-3) & valid_pct_s
            if m.sum() > 0:
                half_width_pct7 = max(half_width_pct7,
                                      float(np.percentile(np.abs(pct_err_s[m]), 95)))
        half_width_pct7 = max(half_width_pct7, 10.0)

        edges_pct7  = np.linspace(-half_width_pct7, half_width_pct7, 81)
        centers_pct7 = 0.5 * (edges_pct7[:-1] + edges_pct7[1:])

        fig7, ax7 = plt.subplots(figsize=(7.5, 5.0))
        for v in VMAX_FIG4:
            m = (np.abs(vmax_s - v) < 1e-3) & valid_pct_s
            if m.sum() == 0:
                continue
            e = pct_err_s[m]
            counts, _ = np.histogram(e, bins=edges_pct7, density=True)
            bias_v = float(np.mean(e));  skew_v = _skewness(e)
            ax7.plot(centers_pct7, counts, '-', color=_vc4(v), linewidth=lw,
                     label=(fr'$v_{{\mathrm{{max}},j}} = {v:.2f}$:  '
                            fr'bias $= {bias_v:+.1f}$\%,  '
                            fr'skew $= {skew_v:+.2f}$  '
                            fr'($n={int(m.sum())}$)'))
        ax7.axvline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
        ax7.set_yscale('log')
        ax7.set_xlabel(r'$100 \times (\hat{\tau}^{*} - \tau^{*}) / \tau^{*}$  [\%]',
                       fontname=fn, fontsize=fs_label)
        ax7.set_ylabel('PDF', fontname=fn, fontsize=fs_label)
        # journal figures: caption supplied externally, no in-figure title
        # ax7.set_title(rf'Signed percentage error by $v_{{\mathrm{{max}},j}}$  '
        #               rf'($\tau^{{*}} < {int(TAU_SMALL_THRESH)}$ s)',
        #               fontname=fn, fontsize=fs_title)
        ax7.set_xlim(-half_width_pct7, half_width_pct7)
        ax7.grid(True, which='both', alpha=0.3)
        ax7.legend(fontsize=fs_legend, loc='lower left', framealpha=0.95)
        ax7.tick_params(labelsize=fs_tick)
        fig7.tight_layout()
        p = os.path.join(save_dir, 'fig7_signed_pct_error_pdf_by_vmax_tau_lt_30.png')
        fig7.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        fig7.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig7);  print(f"  Saved: {p}")

    print("\nAll paper figures saved to", save_dir)


# =============================================================================
#  CLI
# =============================================================================
if __name__ == '__main__':
    _p = argparse.ArgumentParser()
    _p.add_argument('--model_dir', type=str,
                     default='../Results/nn_bs128_h128-128-64_vraw_iw_huber3_data1',
                     help='Path to vraw model directory.')
    _p.add_argument('--save_dir', type=str, default=None,
                     help='Output directory (default: <model_dir>/../paper_ml_figures_vraw/)')
    _args = _p.parse_args()
    make_paper_plots_vraw(_args.model_dir, save_dir=_args.save_dir)
