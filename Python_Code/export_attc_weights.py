#!/usr/bin/env python
"""
export_attc_weights.py — Export aTTC NN weights to .mat for MATLAB.

Reads model.h5 + config.json from a results directory and saves all weights,
biases, and normalization stats to a single .mat file that can be loaded in
MATLAB with:
    nn = load('attc_nn_weights.mat');

Supports both standard (Sequential) and FiLM-conditioned (Functional) models.

Usage:
    python export_attc_weights.py
    python export_attc_weights.py --model_dir ../Results/attc_3D_noCBF_T150k_mse
"""

import argparse
import json
import numpy as np
import scipy.io as sio
from pathlib import Path


def export_weights(model_dir):
    """Extract weights from model.h5 + config.json and save to .mat."""
    model_dir = Path(model_dir)
    model_path = model_dir / 'model.h5'
    config_path = model_dir / 'config.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load model (use project utility to handle TF 2.0/h5py compatibility)
    from auspice_ml_utils import load_keras_model
    model = load_keras_model(str(model_path))

    film = config.get('film', False)
    weights = {}

    if film:
        weights = _export_film_weights(model, config)
    else:
        weights = _export_standard_weights(model, config)

    # Save
    out_path = model_dir / 'attc_nn_weights.mat'
    sio.savemat(str(out_path), weights)

    if film:
        print(f"\nSaved FiLM model weights to {out_path}")
        print(f"  Main pathway: W1-W{len([k for k in weights if k.startswith('W')])} "
              f"(input_dim={weights['W1'].shape[0]})")
        print(f"  Conditioning: cond_W1-cond_W3")
        print(f"  v_max range: [{weights['v_max_min']:.4f}, {weights['v_max_max']:.4f}]")
    else:
        n_layers = len([k for k in weights if k.startswith('W')])
        print(f"\nSaved {n_layers}-layer NN weights to {out_path}")
        print(f"  Features: {int(weights['dim'])}D, "
              f"input_dim={weights['W1'].shape[0]}, "
              f"architecture: {' -> '.join(str(weights[f'W{i}'].shape[1]) for i in range(1, n_layers+1))}")

    return str(out_path)


def _export_standard_weights(model, config):
    """Export weights from a standard Sequential model."""
    weights = {}
    layer_idx = 0
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            layer_idx += 1
            kernel = w[0]  # shape: (input_dim, output_dim)
            bias = w[1]    # shape: (output_dim,)
            weights[f'W{layer_idx}'] = kernel.astype(np.float64)
            weights[f'b{layer_idx}'] = bias.reshape(-1, 1).astype(np.float64)
            print(f"  Layer {layer_idx}: {layer.name}, "
                  f"kernel={kernel.shape}, bias={bias.shape}")

    # Normalization stats
    norm_stats = config['norm_stats']
    weights['feat_mean'] = np.array(norm_stats['feat_mean'],
                                     dtype=np.float64).reshape(-1, 1)
    weights['feat_std'] = np.array(norm_stats['feat_std'],
                                    dtype=np.float64).reshape(-1, 1)
    weights['dim'] = np.float64(config['dim'])
    weights['log_transform'] = np.float64(1.0 if config.get('log_transform', False) else 0.0)
    cap = config.get('attc_cap', None)
    weights['attc_cap'] = np.float64(cap) if cap is not None else np.float64(-1.0)
    return weights


def _export_film_weights(model, config):
    """Export weights from a FiLM-conditioned Functional model.

    Separates main pathway (W1-W4, b1-b4) from conditioning pathway
    (cond_W1-cond_W3, cond_b1-cond_b3).
    """
    weights = {}

    # --- Main pathway: layers named 'main_dense_1', 'main_dense_2', etc. ---
    main_idx = 0
    for layer in model.layers:
        if (layer.name.startswith('main_dense_')
                or layer.name in ('output', 'output_raw')):
            w = layer.get_weights()
            if w:
                main_idx += 1
                kernel = w[0]
                bias = w[1]
                weights[f'W{main_idx}'] = kernel.astype(np.float64)
                weights[f'b{main_idx}'] = bias.reshape(-1, 1).astype(np.float64)
                print(f"  Main {main_idx}: {layer.name}, "
                      f"kernel={kernel.shape}, bias={bias.shape}")

    # --- Conditioning pathway: layers named 'cond_dense_1', 'cond_dense_2', 'cond_film_output' ---
    cond_idx = 0
    for layer in model.layers:
        if layer.name.startswith('cond_'):
            w = layer.get_weights()
            if w:
                cond_idx += 1
                kernel = w[0]
                bias = w[1]
                weights[f'cond_W{cond_idx}'] = kernel.astype(np.float64)
                weights[f'cond_b{cond_idx}'] = bias.reshape(-1, 1).astype(np.float64)
                print(f"  Cond {cond_idx}: {layer.name}, "
                      f"kernel={kernel.shape}, bias={bias.shape}")

    # --- Normalization stats ---
    norm_stats = config['norm_stats']
    weights['feat_mean'] = np.array(norm_stats['feat_mean'],
                                     dtype=np.float64).reshape(-1, 1)
    weights['feat_std'] = np.array(norm_stats['feat_std'],
                                    dtype=np.float64).reshape(-1, 1)
    weights['dim'] = np.float64(config['dim'])

    # --- FiLM-specific ---
    weights['film'] = np.float64(1.0)
    weights['v_max_min'] = np.float64(norm_stats['v_max_min'])
    weights['v_max_max'] = np.float64(norm_stats['v_max_max'])

    # Hidden sizes for gamma/beta splitting
    hidden = [int(h) for h in config['hidden'].split(',')]
    weights['film_sizes'] = np.array(hidden, dtype=np.float64).reshape(1, -1)

    # Log-transform flag
    weights['log_transform'] = np.float64(1.0 if config.get('log_transform', False) else 0.0)

    # Soft-cap at output (if any): pred = cap - softplus(cap - softplus(a4))
    # Stored as cap if present, -1.0 if not (sentinel for "no cap").
    cap = config.get('attc_cap', None)
    weights['attc_cap'] = np.float64(cap) if cap is not None else np.float64(-1.0)

    return weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export aTTC NN weights to .mat for MATLAB.")
    parser.add_argument('--model_dir',
                        default='../Results/nn_bs128_h128-128-64_vraw_iw_huber3_data1',
                        help='Path to model directory')
    args = parser.parse_args()

    print("=" * 50)
    print("  Export aTTC NN Weights to MATLAB")
    print("=" * 50)
    print(f"\nModel directory: {args.model_dir}")
    export_weights(args.model_dir)
    print("\nDone.")
