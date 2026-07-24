A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles

Reference implementation for the paper. This repository contains everything
needed to reproduce the paper's results:

- a MATLAB multi-agent flight simulator with a quadratic-program (QP)
  control-barrier-function (CBF) safety filter,
- the **aTTC-CBF**: a CBF whose barrier function is a neural-network
  surrogate of the *adversarial time-to-collision* (aTTC),
- the TensorFlow training pipeline for that surrogate, and
- the trained network used in the paper.



## Repository layout

```
Github_Journal/
├── Matlab_Code/           simulator + CBF filter + plotting     (see its README)
│   └── examples/          quickstart demo + paper reproduction scripts
├── Python_Code/           TensorFlow training pipeline          (see its README)
└── Results/
    └── nn_bs128_h128-128-64_vraw_iw_huber3_data1/
                           the trained aTTC surrogate used in the paper
                           (Keras model.h5 + exported attc_nn_weights.mat)
```

The simulator supports five safety-filter settings
(`cbf_type` = `none` | `hocbf` | `shocbf` | `attc_vraw` | `sattc_vraw`);
the example scripts reproduce the paper's headline comparison, which uses
the deterministic pair (`hocbf` vs `attc_vraw`) plus the `none` reference.

## Quickstart (5 minutes)

Requires MATLAB (R2022a or later; Optimization Toolbox recommended but not
required — a fallback QP solver is included). No Python needed: the trained
network ships with the repo.

```matlab
cd Matlab_Code/examples
example_quickstart          % 3 short runs: no CBF vs HOCBF vs aTTC-CBF
```

This runs a 100-second pursuit-evasion scenario under each safety filter and
plots the trajectories side by side with collision counts.

## Reproducing the paper results

The paper's statistics come from long-horizon (T = 50,000 s) stochastic
simulations. Each cell takes hours; run them on a workstation or cluster.

```matlab
cd Matlab_Code/examples
reproduce_random_sphere_experiments      % 9 cells: 3 scenarios x {none, HOCBF, aTTC}
reproduce_formation_experiments          % 4 cells: 2 scenarios x {HOCBF, aTTC}
plot_paper_figures                       % figures + summary tables from the cells
```

Pass a shorter horizon (e.g. `reproduce_random_sphere_experiments(1000)`)
for a fast sanity check of the full pipeline.

## Retraining the network (optional)

The trained network is included, so this is only needed if you want to
regenerate it from scratch or train variants:

1. **Generate labeled training data** (MATLAB, slow — hours per job):
   ```matlab
   cd Matlab_Code
   generate_training_data(0.15)   % repeat for 0.25, 0.35, 0.50
   ```
   This simulates two-agent scenarios and labels every timestep with the
   aTTC oracle at five assumed pursuer speeds → 20 .mat files in Results/.

2. **Train + export** (Python / TensorFlow):
   ```bash
   cd Python_Code
   pip install -r requirements.txt
   python run_ml_training_vraw.py          # trains, saves model.h5 + config.json
   python export_attc_weights.py           # writes attc_nn_weights.mat for MATLAB
   ```

See `Python_Code/README.md` for details on the architecture, loss, and
evaluation figures.

## Citing

If you use this code, please cite the paper (citation forthcoming).
