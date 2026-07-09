"""
auspice_ml_test.py — Apply a trained aTTC surrogate model to test data.

Generates diagnostic plots:
  1. Time histories  — true vs predicted aTTC, prediction error, scatter plot
  2. PDFs            — distribution of true vs predicted aTTC, error distribution

Usage (imported):
    from auspice_ml_test import auspice_ml_test
    results = auspice_ml_test(data_test, model, config)

Usage (CLI):
    python auspice_ml_test.py --mat_file ../Results/attc_train_2D.mat \
        --model_dir ../Results/attc_2D_mse_test/ --ratio_skip 0.8
"""

import json
import numpy as np
from pathlib import Path

from auspice_ml_utils import (
    extract_pair_features, extract_multi_pair_features, parse_pair_idx,
    normalize_results, load_keras_model,
)


def auspice_ml_test(data, model, config, pair_idx=0, save_dir=None,
                    features_precomputed=None, targets_precomputed=None,
                    v_max_array=None, attc_type=None, cap_pred=None):
    """Apply trained surrogate model to test data and generate plots.

    Parameters
    ----------
    data : dict or None
        Test data with keys 'x', 'attc', 'params'.  Can be None if
        features_precomputed and targets_precomputed are provided.
    model : keras.Model or str
        Trained model, or path to model.h5.
    config : dict or str
        Training config (with 'norm_stats'), or path to config.json.
    pair_idx : int, str, or list
        Agent pair selection (0-based).  Accepts a single int, a
        comma-separated string (e.g. "0,2,5"), a list of ints, or
        "all" to test on every pair.
    save_dir : str or None
        Directory to save plots.  If None, plots are shown interactively.
    features_precomputed : ndarray or None
        Pre-extracted features (N, 2*dim).
    targets_precomputed : ndarray or None
        Pre-extracted targets (N,).
    v_max_array : ndarray or None
        Per-sample v_max values (for FiLM models).
    attc_type : str or None
        Override aTTC type for prediction mode.  If None, reads from
        data['params']['attc_type'].  When 'pursuit_ego', prediction
        is f(x).  When 'pursuit_min', prediction is min(f(x), f(-x)).
    cap_pred : float or None
        If set, clamp predictions to min(pred, cap_pred).  Useful when
        test targets were capped at a horizon so metrics compare fairly.

    Returns
    -------
    results : dict
        'attc_true', 'attc_pred', 'mae', 'rmse', 'mape'
    """
    import tensorflow as tf
    from tensorflow import keras

    ##### Load model / config if paths #####
    if isinstance(model, str):
        model = load_keras_model(model)
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    ##### Create save directory if needed #####
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    norm_stats = config["norm_stats"]
    feat_mean  = np.array(norm_stats["feat_mean"], dtype=np.float32)
    feat_std   = np.array(norm_stats["feat_std"],  dtype=np.float32)
    film = config.get("film", False)

    ##### Feature extraction #####
    if features_precomputed is not None and targets_precomputed is not None:
        features = features_precomputed
        targets  = targets_precomputed
    else:
        data = normalize_results(data)
        attc_cap = config.get("attc_cap", None)
        dim = int(data["params"]["dim"])
        na  = int(data["params"]["num_agents"])
        # Auto-detect attc_type from data if not overridden
        if attc_type is None:
            attc_type = data["params"].get("attc_type", "pursuit_ego")
        pair_indices = parse_pair_idx(pair_idx, na)
        if len(pair_indices) == 1:
            print(f"\nExtracting test features for pair {pair_indices[0]} ({dim}D) ...")
            features, targets = extract_pair_features(data, pair_indices[0], attc_cap=attc_cap)
        else:
            print(f"\nExtracting test features for {len(pair_indices)} pairs {pair_indices} ({dim}D) ...")
            features, targets = extract_multi_pair_features(data, pair_indices, attc_cap=attc_cap)

    # Default attc_type if still unset (precomputed path)
    if attc_type is None:
        attc_type = "pursuit_ego"

    if len(targets) == 0:
        print("WARNING: No finite aTTC samples in test data.")
        return {"attc_true": np.array([]), "attc_pred": np.array([]),
                "mae": np.nan, "rmse": np.nan, "mape": np.nan}

    ##### Predict #####
    log_transform = config.get("log_transform", False)
    X = (features - feat_mean) / feat_std
    X_neg = (-features - feat_mean) / feat_std  # negated features for f(-x)
    Y_true = targets

    print(f"  aTTC type: {attc_type}")

    if film:
        v_max_min = norm_stats["v_max_min"]
        v_max_max = norm_stats["v_max_max"]
        if v_max_array is None:
            raise ValueError("v_max_array required for FiLM model testing")
        if v_max_max - v_max_min < 1e-10:
            X_vmax = np.full((len(Y_true), 1), 0.5, dtype=np.float32)
        else:
            X_vmax = ((v_max_array - v_max_min) / (v_max_max - v_max_min)
                      ).reshape(-1, 1).astype(np.float32)
        Y_pred_ego = model.predict([X, X_vmax]).squeeze()
        if attc_type == "pursuit_min":
            Y_pred_neg = model.predict([X_neg, X_vmax]).squeeze()
            Y_pred = np.minimum(Y_pred_ego, Y_pred_neg)
        else:
            Y_pred = Y_pred_ego
    else:
        Y_pred_ego = model.predict(X).squeeze()
        if attc_type == "pursuit_min":
            Y_pred_neg = model.predict(X_neg).squeeze()
            Y_pred = np.minimum(Y_pred_ego, Y_pred_neg)
        else:
            Y_pred = Y_pred_ego

    # Inverse log-transform: model outputs log(1 + aTTC), convert back to seconds
    if log_transform:
        Y_pred = np.expm1(Y_pred)   # exp(pred) - 1
        print(f"  Log-transform: predictions mapped back via exp(pred) - 1")

    # Optionally cap predictions for fair comparison with capped targets
    if cap_pred is not None:
        n_capped = int(np.sum(Y_pred > cap_pred))
        Y_pred = np.minimum(Y_pred, cap_pred)
        print(f"  cap_pred={cap_pred:.1f}s — clamped {n_capped} predictions "
              f"({100*n_capped/len(Y_pred):.1f}%)")

    ##### Metrics #####
    mae  = float(np.mean(np.abs(Y_true - Y_pred)))
    rmse = float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))
    mape = float(100 * np.mean(np.abs((Y_true - Y_pred) / (Y_true + 1e-6))))

    print(f"\n{'='*40}")
    print(f"  Test Results  ({len(Y_true)} samples)")
    print(f"  aTTC type: {attc_type}")
    if attc_type == "pursuit_min":
        print(f"  Prediction: min(f(x), f(-x))")
    else:
        print(f"  Prediction: f(x)")
    print(f"{'='*40}")
    print(f"  MAE:  {mae:.4f} s")
    print(f"  RMSE: {rmse:.4f} s")
    print(f"  MAPE: {mape:.1f} %")

    # Binned MAE by aTTC range
    bins = [(0, 10), (10, 30), (30, 60), (60, np.inf)]
    binned_mae = {}
    print(f"\n  Binned MAE:")
    for lo, hi in bins:
        mask = (Y_true >= lo) & (Y_true < hi)
        n_bin = int(np.sum(mask))
        if n_bin > 0:
            mae_bin = float(np.mean(np.abs(Y_true[mask] - Y_pred[mask])))
            label = f"[{lo:.0f}, {hi:.0f})" if np.isfinite(hi) else f"[{lo:.0f}, inf)"
            binned_mae[f"mae_{lo:.0f}_{hi:.0f}" if np.isfinite(hi) else f"mae_{lo:.0f}_inf"] = mae_bin
            print(f"    {label:>12}: {mae_bin:.4f} s  ({n_bin} samples)")
        else:
            label = f"[{lo:.0f}, {hi:.0f})" if np.isfinite(hi) else f"[{lo:.0f}, inf)"
            print(f"    {label:>12}: N/A  (0 samples)")

    print(f"{'='*40}\n")

    ##### Plots #####
    fig1 = _plot_time_histories(Y_true, Y_pred, mae, rmse, mape, save_dir,
                                attc_type=attc_type)
    fig2 = _plot_pdfs(Y_true, Y_pred, save_dir, attc_type=attc_type)

    return {
        "attc_true": Y_true,
        "attc_pred": Y_pred,
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
        "binned_mae": binned_mae,
        "fig_time_histories": fig1,
        "fig_pdfs": fig2,
    }


# ============================================================================
#                         PLOTTING
# ============================================================================

def _plot_time_histories(Y_true, Y_pred, mae, rmse, mape, save_dir,
                         attc_type="pursuit_ego"):
    """Plot 1: scatter + time histories + error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        pass

    # Label for plots
    if attc_type == "pursuit_min":
        pred_label = "min(f(x), f(-x))"
        type_label = "pursuit\\_min"
    else:
        pred_label = "f(x)"
        type_label = "pursuit\\_ego"

    N = len(Y_true)
    sample_idx = np.arange(N)
    error = Y_pred - Y_true

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel (a): Scatter ---
    ax = axes[0]
    ax.scatter(Y_true, Y_pred, s=8, alpha=0.4, c="steelblue", edgecolors="none")
    lims = [0, max(Y_true.max(), Y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"True aTTC [{type_label}] (s)", fontsize=13)
    ax.set_ylabel(f"Predicted [{pred_label}] (s)", fontsize=13)
    ax.set_title("Predicted vs True", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    # Annotation
    txt = f"MAE={mae:.3f}s\nRMSE={rmse:.3f}s\nMAPE={mape:.1f}%"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # --- Panel (b): Time history ---
    ax = axes[1]
    ax.plot(sample_idx, Y_true, "b-",  linewidth=1.2,
            label=f"True [{type_label}]", alpha=0.8)
    ax.plot(sample_idx, Y_pred, "r--", linewidth=1.2,
            label=f"Pred [{pred_label}]", alpha=0.8)
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("aTTC (s)", fontsize=13)
    ax.set_title("Time History", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # --- Panel (c): Error ---
    ax = axes[2]
    ax.plot(sample_idx, error, "g-", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(np.mean(error), color="r", linestyle=":", linewidth=1.5,
               label=f"Mean = {np.mean(error):.3f}s")
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Error (pred - true) (s)", fontsize=13)
    ax.set_title("Prediction Error", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    suptitle = f"aTTC Surrogate — Test Results [{type_label}]"
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir is not None:
        path = Path(save_dir) / "test_time_histories.png"
        fig.savefig(str(path), dpi=150)
        print(f"Time history plot saved to {path}")
    return fig


def _plot_pdfs(Y_true, Y_pred, save_dir, attc_type="pursuit_ego"):
    """Plot 2: PDFs of true/predicted aTTC and error distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        pass

    if attc_type == "pursuit_min":
        pred_label = "Pred [min(f(x), f(-x))]"
        type_label = "pursuit\\_min"
    else:
        pred_label = "Pred [f(x)]"
        type_label = "pursuit\\_ego"

    error = Y_pred - Y_true
    n_bins = 80

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): aTTC distribution (full range) ---
    ax = axes[0]
    hi = max(Y_true.max(), Y_pred.max())
    edges = np.linspace(0, hi, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    counts_true, _ = np.histogram(Y_true, bins=edges, density=True)
    counts_pred, _ = np.histogram(Y_pred, bins=edges, density=True)

    ax.plot(centers, counts_true, "b-",  linewidth=2,
            label=f"True [{type_label}]")
    ax.plot(centers, counts_pred, "r--", linewidth=2,
            label=pred_label)
    ax.set_yscale("log")
    ax.set_xlabel("aTTC (s)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("aTTC Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # --- Panel (b): Error distribution (full range) ---
    ax = axes[1]
    lo_e = error.min()
    hi_e = error.max()
    edges_e = np.linspace(lo_e, hi_e, n_bins)
    centers_e = (edges_e[:-1] + edges_e[1:]) / 2
    counts_e, _ = np.histogram(error, bins=edges_e, density=True)

    ax.plot(centers_e, counts_e, "g-", linewidth=2)
    ax.set_yscale("log")
    ax.axvline(0, color="k", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(error), color="r", linestyle=":", linewidth=1.5,
               label=f"Mean = {np.mean(error):.3f}s")
    ax.axvline(np.mean(error) + np.std(error), color="orange", linestyle=":",
               linewidth=1, label=f"Std = {np.std(error):.3f}s")
    ax.axvline(np.mean(error) - np.std(error), color="orange", linestyle=":",
               linewidth=1)
    ax.set_xlabel("Error (pred - true) (s)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("Error Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    suptitle = f"aTTC Surrogate — Distribution Comparison [{type_label}]"
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir is not None:
        path = Path(save_dir) / "test_pdfs.png"
        fig.savefig(str(path), dpi=150)
        print(f"PDF plot saved to {path}")
    return fig


# ============================================================================
#                              CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    from auspice_ml_utils import load_mat
    from auspice_ml_split import split_structs

    parser = argparse.ArgumentParser(description="Test aTTC surrogate model.")
    parser.add_argument("--mat_file",   required=True, help="Path to .mat file")
    parser.add_argument("--model_dir",  required=True,
                        help="Path to results directory containing model.h5 + config.json")
    parser.add_argument("--ratio_skip", type=float, default=0.8,
                        help="Skip first ratio_skip fraction (use remainder as test)")
    parser.add_argument("--pair_idx",   type=str, default="0",
                        help="Pair index: int, comma-separated ints, or 'all' (default: 0)")
    args = parser.parse_args()

    data = load_mat(args.mat_file)
    _, data_test = split_structs(data, ratio=args.ratio_skip)

    model_path  = str(Path(args.model_dir) / "model.h5")
    config_path = str(Path(args.model_dir) / "config.json")

    results = auspice_ml_test(
        data_test,
        model=model_path,
        config=config_path,
        pair_idx=args.pair_idx,
        save_dir=args.model_dir,
    )
