# A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles

Reference implementation for all published work on aTTC-CBFs. This repository contains everything
needed to reproduce the results from the following papers:

https://arxiv.org/abs/2608.14239

CDC Conference Paper - link coming soon

- a MATLAB multi-agent flight simulator with a quadratic-program (QP)
  control-barrier-function (CBF) safety filter,
- the **aTTC-CBF**: a CBF whose barrier function is a neural-network
  surrogate of the *adversarial time-to-collision* (aTTC),
- the TensorFlow training pipeline for that surrogate, and
- the trained network used in the paper.

*This repository was made with the help of AI*

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
(`cbf_type` = `none` | `hocbf` | `shocbf` | `attc_vraw` | `sattc_vraw`).
The random-sphere and formation scripts use the deterministic pair
(`hocbf` vs `attc_vraw`) plus the `none` reference; the pure-pursuit script
uses the stochastic pair (`shocbf` vs `sattc_vraw`).

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
reproduce_pursuit_experiments            % 2 runs:  pure pursuit, {sHOCBF, aTTC-SCBF}
reproduce_random_sphere_experiments      % 9 cells: 3 scenarios x {none, HOCBF, aTTC}
reproduce_formation_experiments          % 4 cells: 2 scenarios x {HOCBF, aTTC}
plot_paper_figures                       % figures + summary tables from the cells
```

Pass a shorter horizon (e.g. `reproduce_random_sphere_experiments(1000)`)
for a fast sanity check of the full pipeline.

### Pure pursuit

`reproduce_pursuit_experiments` is the scenario behind the paper's headline
table: a single evader orbiting a fixed waypoint against **five pursuers that
are 50% faster** (`pursuit_advantage = 1.5`), for T = 50,000 s. The evader is
the only agent carrying a safety filter; the pursuers fly a proportional
intercept at their maximum speed and are never inhibited.

| metric | aTTC-SCBF | sHOCBF |
|---|---|---|
| collision rate [per 100 s] | 3.7 | 16.2 |
| collisions / close encounter | 0.50 | 0.69 |

A **collision** is a pair distance below `2*r_ttc` = 0.2 km and a **close
encounter** below `4*r_ttc` = 0.4 km, both counted as rising edges per
evader-pursuer pair, so one sustained approach counts once. The second row is
therefore the fraction of close encounters that the evader fails to escape.

Scenario parameters, read from the paper's stored runs:

| | value |
|---|---|
| agents | 6 = 1 evader + 5 pursuers |
| evader speed | `v_desired` 0.25, `v_max` 0.5 km/s |
| pursuer speed | 0.75 km/s (`pursuit_advantage` 1.5) |
| control limits | `omega_max` 0.4 rad/s, `nu_max` 0.1 rad/s, `a_max` 0.05 km/s² |
| collision radius | `r_ttc` 0.1 km |
| horizon / step | T = 50,000 s, `dt` = 0.1 s |
| SCBF bound | `scbf_p` 0.9, `scbf_alpha` 0.1, `scbf_beta` 0.01 |

Note the 4:1 ratio between the yaw-rate and pitch-rate limits — the evader has
four times as much horizontal as vertical authority. The distance-based
barrier, knowing nothing of the vehicle limits, is biased toward the
high-authority channel; the aTTC barrier has learned them.

Once both runs exist the script calls `auspice_plot_pursuit_compare` on the
pair and writes the comparison figure set into `Results/paper_pursuit/`:
summary bars, min/mean/orbit-distance PDFs, control-at-collision scatters,
the proximity CDF, the evasion-channel and min-distance paper figures, and one
trajectory plot per method.  Holding both T = 50,000 s result structs in
memory at once needs roughly 1 GB.

The initial conditions are drawn randomly and the stored runs did not record
their seed, so the script seeds `rng(42)` and reproduces the experiment
statistically rather than bit-for-bit. At T = 50,000 s each arm accumulates
thousands of collision events and the rates above are stable to the quoted
precision; at short horizons they are not — a 300 s run sees only a handful
of events.

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
