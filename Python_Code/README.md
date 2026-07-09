# Python_Code — TensorFlow training pipeline for the aTTC surrogate

Trains the neural network that `attc_vraw` (the aTTC-CBF) uses inside the
MATLAB simulator, and exports its weights into the plain-`.mat` format the
MATLAB forward pass reads.

## Setup

```bash
pip install -r requirements.txt      # tensorflow, numpy, scipy, matplotlib, h5py
```

## Pipeline

```
Matlab_Code/generate_training_data.m           (MATLAB — run first)
        │   20 x  ../Results/training_data_data1_3D_2A_T50000_vdes*_vmax*.mat
        ▼
run_ml_training_vraw.py                        (train the FiLM network)
        │   ../Results/nn_bs128_h128-128-64_vraw_iw_huber3_data1/
        │       model.h5  config.json  training_loss.png  ...
        ▼
export_attc_weights.py                         (export for MATLAB)
        │   attc_nn_weights.mat  (plain weight/bias/norm arrays)
        ▼
Matlab_Code (auspice_sim 'cbf_type','attc_vraw')
```

```bash
python run_ml_training_vraw.py                 # defaults: loss iw_huber3, tag data1
python export_attc_weights.py                  # writes attc_nn_weights.mat
python plot_paper_ml_results_vraw.py           # the paper's ML evaluation figures
```

## The model

- **Inputs (9-dim):** relative position Δp = p_i − p_j and both raw
  velocity vectors v_i, v_j (world frame). Giving the network both
  velocities — not just their difference — lets it reason about how the
  pursuer's reachable set compares to the ego's.
- **Conditioning:** the assumed pursuer speed limit v_max is injected via
  FiLM (feature-wise linear modulation): a small parallel branch
  (16-16 hidden units) produces per-layer scale γ and shift β applied to
  each main-branch pre-activation.
- **Main branch:** fully connected 128-128-64 → scalar aTTC prediction.
- **Loss:** inverse-weighted Huber (`iw_huber3`), weighting samples by
  1/τ*ⁿ so the low-aTTC (high-risk) regime — the regime the CBF actually
  acts in — dominates training.
- **Data:** ~10M samples from the 20 training files (4 evader cruise
  speeds × 5 assumed pursuer speeds); samples with infinite aTTC
  (no collision within the oracle horizon) are dropped. 75/25 train/test
  split, seed 42.

## Files

| File | Purpose |
|---|---|
| `run_ml_training_vraw.py` | Training driver: loads the 20 data files, extracts features, trains, saves `model.h5` + `config.json` + diagnostics. |
| `auspice_ml_utils.py` | Data loading (`.mat` v7.3 via h5py), normalization, model (de)serialization. |
| `auspice_ml_utils_vraw.py` | The 9-dim `[Δp, v_i, v_j]` feature extraction. |
| `auspice_ml_split.py` | Train/test splitting. |
| `auspice_ml_train.py` | Model construction (FiLM MLP), loss zoo, training loop. |
| `auspice_ml_test.py` | Held-out evaluation. |
| `export_attc_weights.py` | Exports Keras weights + normalization stats to `attc_nn_weights.mat` for the MATLAB forward pass. |
| `plot_paper_ml_results_vraw.py` | The paper's ML figures: conditional MAE/MPE vs threshold, error PDFs by v_max, true-vs-predicted scatter. |

## Reproducing the shipped network

The repo ships the exact network used in the paper
(`Results/nn_bs128_h128-128-64_vraw_iw_huber3_data1/`). To retrain it from
scratch: generate the training data in MATLAB (hours per v_desired job),
then run the three Python commands above with their defaults. Training
takes ~1 hour on a modern CPU node; results are deterministic up to
TensorFlow's per-platform nondeterminism.
