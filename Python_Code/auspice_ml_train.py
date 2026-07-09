"""
auspice_ml_train.py — Train an aTTC surrogate model from simulation data.

Learns the map  (delta_p, delta_v) -> aTTC  for a given agent pair using a
small dense network.  Saves model + config, plots training history.

Usage (imported):
    from auspice_ml_train import auspice_ml_train
    model, history, config = auspice_ml_train(data_train, loss='mse', epochs=200)

Usage (CLI):
    python auspice_ml_train.py --mat_file ../Results/attc_train_2D.mat \
        --loss mse --epochs 200 --tag my_run
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from auspice_ml_utils import (
    extract_pair_features, extract_multi_pair_features, parse_pair_idx,
    normalize_results, build_model, get_loss,
    parse_loss_name, compute_ow_weights, compute_iw_weights, RESULTS_DIR,
)


def auspice_ml_train(data, loss='mse', epochs=200, val_split=0.1,
                     batch_size=512, hidden='64,64,32', lr=1e-3,
                     patience=20, shuffle=True, seed=42,
                     tag=None, pair_idx=0, ow_clip_max=50.0,
                     ow_mode='output_only', attc_cap=None, attc_filter=None,
                     film=False, v_max_array=None,
                     features_precomputed=None, targets_precomputed=None,
                     log_transform=False, huber_delta=1.0,
                     mat_files=None):
    """Train aTTC surrogate model.

    Parameters
    ----------
    data : dict or None
        Training data with keys 'x', 'attc', 'params'.  Can be None if
        features_precomputed and targets_precomputed are provided.
    loss : str
        Loss function name: 'mse', 'mae', 'huber', 'mape', 'log_mse'.
        Prefix with 'ow_' for output-weighted variant (e.g. 'ow_mse').
    epochs : int
        Maximum training epochs.
    val_split : float
        Fraction of training data held out for validation (default 0.1).
    batch_size : int
        Mini-batch size.
    hidden : str
        Comma-separated hidden layer sizes (e.g. '64,64,32').
    lr : float
        Initial learning rate for Adam.
    patience : int
        Early-stopping patience (epochs without improvement).
    shuffle : bool
        Shuffle training samples before splitting (default True).
    seed : int
        Random seed for reproducibility.
    tag : str or None
        Run name for save directory.  Defaults to auto-generated timestamp.
    pair_idx : int, str, or list
        Agent pair selection (0-based).  Accepts a single int, a
        comma-separated string (e.g. "0,2,5"), a list of ints, or
        "all" to train on every pair.
    ow_clip_max : float
        Maximum OW weight after normalisation (default 50).  Only used when
        loss starts with 'ow_'.
    ow_mode : str
        'output_only' (default) — w = 1/p_y, assumes uniform input density.
        'full' — w = p_x/p_y with marginal KDEs for p_x.
    attc_cap : float or None
        If set, cap inf aTTC values at this value (seconds) instead of
        discarding them.  E.g. attc_cap=120 replaces inf with 120s.
    attc_filter : float or None
        If set, discard all samples with aTTC > this threshold (seconds).
        Applied after attc_cap.
    film : bool
        If True, build a FiLM-conditioned model with v_max as conditioning.
    v_max_array : ndarray (N,) or None
        Per-sample v_max values.  Required when film=True.
    features_precomputed : ndarray or None
        Pre-extracted features (N, 2*dim).  Bypasses feature extraction from data.
    targets_precomputed : ndarray or None
        Pre-extracted targets (N,).  Bypasses feature extraction from data.
    log_transform : bool
        If True, train on log(1 + aTTC) instead of raw aTTC.
        Compresses the target range and improves learning.  The inverse
        transform exp(pred) - 1 is applied at test time. Default is False.

    Returns
    -------
    model : tf.keras.Model
    history : dict
    config : dict  (includes norm_stats, save_dir)
    """
    import tensorflow as tf
    from tensorflow import keras

    np.random.seed(seed)
    tf.random.set_seed(seed)

    if film and v_max_array is None:
        raise ValueError("v_max_array is required when film=True")

    ##### Parse loss name for weighting prefix (ow_ / iw_) and IW power #####
    base_loss, weighting, iw_power = parse_loss_name(loss)

    ##### Feature extraction #####
    if features_precomputed is not None and targets_precomputed is not None:
        # Use pre-extracted features (multi-v_max pipeline)
        features = features_precomputed
        targets  = targets_precomputed
        dim = features.shape[1] // 2
        pair_indices = [0]  # placeholder
    else:
        # Standard single-file extraction
        data = normalize_results(data)
        attc_type = data["params"].get("attc_type", "unknown")
        if attc_type != "pursuit_ego":
            raise ValueError(
                f"Training data has attc_type='{attc_type}'. "
                f"Only 'pursuit_ego' is valid for NN training. "
                f"pursuit_min = min(f(x), f(-x)) is non-smooth and should not "
                f"be learned directly — train on pursuit_ego instead."
            )
        dim = int(data["params"]["dim"])
        na  = int(data["params"]["num_agents"])
        pair_indices = parse_pair_idx(pair_idx, na)
        if len(pair_indices) == 1:
            print(f"\nExtracting features for pair {pair_indices[0]} ({dim}D) ...")
            features, targets = extract_pair_features(data, pair_indices[0],
                                                      attc_cap=attc_cap,
                                                      attc_filter=attc_filter)
        else:
            print(f"\nExtracting features for {len(pair_indices)} pairs {pair_indices} ({dim}D) ...")
            features, targets = extract_multi_pair_features(data, pair_indices,
                                                            attc_cap=attc_cap,
                                                            attc_filter=attc_filter)

    if len(targets) == 0:
        raise ValueError("No finite aTTC samples found.  Cannot train.")

    ##### Log-transform targets #####
    if log_transform:
        targets = np.log1p(targets)   # log(1 + aTTC)
        print(f"  Log-transform: targets mapped to log(1 + aTTC), "
              f"range [{targets.min():.3f}, {targets.max():.3f}]")

    ##### Normalisation #####
    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0) + 1e-8
    X = (features - feat_mean) / feat_std
    Y = targets   # log-space if log_transform, physical units otherwise

    ##### FiLM: normalise v_max (min-max to [0, 1]) #####
    v_max_min, v_max_max = None, None
    X_vmax = None
    if film:
        v_max_min = float(v_max_array.min())
        v_max_max = float(v_max_array.max())
        if v_max_max - v_max_min < 1e-10:
            # Single v_max — normalise to 0.5
            X_vmax = np.full((len(Y), 1), 0.5, dtype=np.float32)
        else:
            X_vmax = ((v_max_array - v_max_min) / (v_max_max - v_max_min)
                      ).reshape(-1, 1).astype(np.float32)

    ##### Compute sample weights (before shuffle, on full training set) #####
    weight_info = None
    if weighting == "ow":
        print("\nComputing output-weighted (OW) sample weights ...")
        print(f"  w(x,y) = p_x(x) / p_y(y_true)  [Blanchard & Sapsis 2021]")
        sample_weights_all, weight_info = compute_ow_weights(
            X, Y, clip_max=ow_clip_max, mode=ow_mode)
    elif weighting == "iw":
        print("\nComputing inverse-weighted (IW) sample weights ...")
        print(f"  w(y) = 1 / y_true^{iw_power}")
        sample_weights_all, weight_info = compute_iw_weights(
            Y, clip_max=ow_clip_max, power=iw_power)
    else:
        sample_weights_all = None

    ##### Shuffle #####
    N = len(Y)
    idx = np.random.permutation(N) if shuffle else np.arange(N)

    ##### Train / val split #####
    n_val = max(1, int(val_split * N))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]

    # FiLM: split conditioning inputs too
    if film:
        Xv_train = X_vmax[train_idx]
        Xv_val   = X_vmax[val_idx]
    print(f"Train: {len(Y_train)}  |  Val: {len(Y_val)}")

    ##### Split OW weights to train/val #####
    if sample_weights_all is not None:
        sw_train = sample_weights_all[train_idx]
        sw_val   = sample_weights_all[val_idx]
    else:
        sw_train = None
        sw_val   = None

    ##### Build model #####
    hidden_sizes = [int(h) for h in hidden.split(",")]
    model = build_model(input_dim=X.shape[1], hidden_sizes=hidden_sizes,
                        film=film, attc_cap=attc_cap)
    if attc_cap is not None:
        print(f"  Output soft-capped at {attc_cap:.1f} s  "
              "(pred = cap - softplus(cap - softplus(a4)))")
    model.summary()

    loss_fn = get_loss(loss, huber_delta=huber_delta)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=loss_fn,
        metrics=["mae"],
    )

    ##### Prepare save directory early (for ModelCheckpoint) #####
    if tag is None:
        tag = f"attc_{dim}D_{loss}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if log_transform and "log" not in tag:
        tag = "log_" + tag
    save_dir = RESULTS_DIR / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "model.h5"

    ##### Callbacks #####
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(5, patience // 3),
            min_lr=1e-6, min_delta=1e-4, verbose=1
        ),
    ]

    ##### Fit #####
    if log_transform:
        print("\n  NOTE: Training in log-space — loss and MAE shown per epoch")
        print("        are in log(1 + aTTC) units, NOT seconds.")
        print("        Final val MAE will be reported in seconds.\n")

    t0 = time.time()
    if film:
        X_train_input = [X_train, Xv_train]
        X_val_input   = [X_val, Xv_val]
    else:
        X_train_input = X_train
        X_val_input   = X_val

    fit_kwargs = dict(
        validation_data=(X_val_input, Y_val) if sw_val is None
                        else (X_val_input, Y_val, sw_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    if sw_train is not None:
        fit_kwargs['sample_weight'] = sw_train

    hist = model.fit(X_train_input, Y_train, **fit_kwargs)
    elapsed = time.time() - t0
    n_epochs_run = len(hist.history["loss"])
    print(f"\nTraining complete in {elapsed:.1f}s  ({n_epochs_run} epochs)")
    if weighting:
        print(f"  ({weighting.upper()}-weighted {base_loss.upper()} loss)")

    ##### Evaluation on val set (unweighted metrics for fair comparison) #####
    Y_pred_raw = model.predict(X_val_input).squeeze()
    if log_transform:
        # Report metrics in original (seconds) space
        Y_val_orig  = np.expm1(Y_val)       # exp(y) - 1
        Y_pred_orig = np.expm1(Y_pred_raw)  # exp(pred) - 1
        mae_val  = float(np.mean(np.abs(Y_val_orig - Y_pred_orig)))
        rmse_val = float(np.sqrt(np.mean((Y_val_orig - Y_pred_orig) ** 2)))
        mape_val = float(100 * np.mean(np.abs((Y_val_orig - Y_pred_orig)
                                               / (Y_val_orig + 1e-6))))
        print(f"Val MAE:  {mae_val:.4f} s  (in original space)")
    else:
        mae_val  = float(np.mean(np.abs(Y_val - Y_pred_raw)))
        rmse_val = float(np.sqrt(np.mean((Y_val - Y_pred_raw) ** 2)))
        mape_val = float(100 * np.mean(np.abs((Y_val - Y_pred_raw)
                                               / (Y_val + 1e-6))))
        print(f"Val MAE:  {mae_val:.4f} s")
    print(f"Val RMSE: {rmse_val:.4f} s")
    print(f"Val MAPE: {mape_val:.1f} %")

    ##### Save (final — ModelCheckpoint already saved best during training) #####
    model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")

    norm_stats = {
        "feat_mean": feat_mean.tolist(),
        "feat_std":  feat_std.tolist(),
    }
    if film:
        norm_stats["v_max_min"] = v_max_min
        norm_stats["v_max_max"] = v_max_max
    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    history["wall_time_s"] = elapsed
    history["val_mae_final"]  = mae_val
    history["val_rmse_final"] = rmse_val
    history["val_mape_final"] = mape_val

    config = {
        "dim":        dim,
        "loss":       loss,
        "weighting":  weighting,
        "hidden":     hidden,
        "lr":         lr,
        "batch_size": batch_size,
        "epochs_run": n_epochs_run,
        "pair_idx":   pair_indices,
        "val_split":  val_split,
        "shuffle":    shuffle,
        "seed":       seed,
        "attc_cap":   attc_cap,
        "attc_filter": attc_filter,
        "log_transform": log_transform,
        "huber_delta": huber_delta,
        "film":       film,
        "norm_stats": norm_stats,
        "history":    history,
        "save_dir":   str(save_dir),
        "mat_files":  list(mat_files) if mat_files is not None else None,
    }
    if film:
        config["cond_hidden"] = [16, 16]
        config["v_max_values"] = sorted(set(float(v)
                                            for v in v_max_array))
    if weight_info is not None:
        config["weight_info"] = {
            "type":      weighting,
            "clip_max":  ow_clip_max,
            "w_mean":    weight_info['w_mean'],
            "w_std":     weight_info['w_std'],
            "w_min":     weight_info['w_min'],
            "w_max":     weight_info['w_max'],
            "w_median":  weight_info['w_median'],
            "n_clipped": weight_info['n_clipped'],
        }
        if weighting == "ow":
            config["weight_info"]["ow_mode"] = ow_mode

    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    ##### Plot training history #####
    _plot_training_history(history, save_dir)

    ##### Plot weight distribution (if applicable) #####
    if sample_weights_all is not None:
        _plot_sample_weights(sample_weights_all, Y, save_dir, weighting)

    return model, history, config


def _plot_training_history(history, save_dir):
    """Plot training & validation loss and MAE curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["loss"],     "b-",  linewidth=2, label="Train loss")
    ax.plot(epochs, history["val_loss"], "r--", linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.tick_params(labelsize=11)

    # MAE
    ax = axes[1]
    ax.plot(epochs, history["mae"],     "b-",  linewidth=2, label="Train MAE")
    ax.plot(epochs, history["val_mae"], "r--", linewidth=2, label="Val MAE")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MAE (s)", fontsize=13)
    ax.set_title("Training & Validation MAE", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.tick_params(labelsize=11)

    fig.suptitle("aTTC Surrogate — Training History", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plot_path = Path(save_dir) / "training_loss.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Training plot saved to {plot_path}")


def _plot_sample_weights(weights, targets, save_dir, weighting):
    """Plot sample weight diagnostics (works for OW and IW)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = weighting.upper()  # "OW" or "IW"

    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel (a): Weight distribution ---
    ax = axes[0]
    ax.hist(weights, bins=80, density=True, color="steelblue", edgecolor="none",
            alpha=0.8)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1.5, label="w = 1")
    ax.axvline(np.median(weights), color="r", linestyle=":", linewidth=1.5,
               label=f"Median = {np.median(weights):.2f}")
    ax.set_xlabel("Sample Weight w", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"{label} Weight Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # --- Panel (b): Weight vs target ---
    ax = axes[1]
    ax.scatter(targets, weights, s=6, alpha=0.4, c="steelblue", edgecolors="none")
    ax.set_xlabel("aTTC target (s)", fontsize=13)
    ax.set_ylabel("Sample Weight w", fontsize=13)
    ax.set_title("Weight vs Target", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # --- Panel (c): Weighted vs unweighted target distribution ---
    ax = axes[2]
    hi = np.quantile(targets, 0.98)
    bins = np.linspace(0, hi, 60)
    ax.hist(targets, bins=bins, density=True, alpha=0.5, color="blue",
            label="Unweighted", edgecolor="none")
    ax.hist(targets, bins=bins, density=True, alpha=0.5, color="red",
            weights=weights, label=f"{label}-weighted", edgecolor="none")
    ax.set_xlabel("aTTC (s)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Effective Target Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.suptitle(f"{label} Sample Weights — Diagnostics",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plot_path = Path(save_dir) / f"{weighting}_weights.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"{label} weight plot saved to {plot_path}")


# ============================================================================
#                              CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    from auspice_ml_utils import load_mat

    parser = argparse.ArgumentParser(description="Train aTTC surrogate model.")
    parser.add_argument("--mat_file",   required=True, help="Path to .mat file")
    parser.add_argument("--loss",       default="mse", help="Loss function (default: mse)")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--val_split",  type=float, default=0.1)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--hidden",     default="64,64,32")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--pair_idx",   type=str,   default="0",
                        help="Pair index: int, comma-separated ints, or 'all' (default: 0)")
    parser.add_argument("--tag",        default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data = load_mat(args.mat_file)
    model, history, config = auspice_ml_train(
        data,
        loss=args.loss,
        epochs=args.epochs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        patience=args.patience,
        pair_idx=args.pair_idx,
        tag=args.tag,
        seed=args.seed,
    )
