# Matlab_Code — simulator, CBF safety filter, and analysis

## Core simulation

| File | Purpose |
|---|---|
| `auspice_sim.m` | Multi-agent 3D (or 2D) Dubins-type flight simulator. Configured entirely through name-value pairs; returns a results struct with state histories `x`, applied controls `u`, pair distances `dist`, and the full parameter set `params`. |
| `auspice_cbf.m` | The QP safety filter. For each controlled agent, builds one CBF constraint per nearby agent and solves `min ‖u − u_nom‖²` subject to those constraints + control bounds. Dispatches on `cbf_type`: `'hocbf'` (distance-based higher-order CBF baseline), `'attc_vraw'` (the paper's aTTC-CBF), or their chance-constrained stochastic counterparts `'shocbf'` / `'sattc_vraw'`. |
| `attc_nn_forward.m` | Forward pass + input gradients of the FiLM-conditioned aTTC network, implemented in plain MATLAB from the exported weight matrices (no toolboxes needed). |
| `attc_nn_forward_hessian.m` | Second-derivative (Hessian) pass, used by the Lie-derivative computation. |
| `auspice_kalman_filter.m`, `auspice_kf_predict.m` | Optional state estimator (off in all paper experiments). |
| `compute_r_eff.m` | Effective collision radius under the optional chance constraint. |

### Choosing a CBF

```matlab
res = auspice_sim('cbf_type', 'none',  ...);       % no safety filter
res = auspice_sim('cbf_type', 'hocbf', ...);       % distance-based baseline
res = auspice_sim('cbf_type', 'attc_vraw', ...);   % aTTC-CBF (the paper's method)
res = auspice_sim('cbf_type', 'shocbf', ...);      % stochastic (chance-constrained) HOCBF
res = auspice_sim('cbf_type', 'sattc_vraw', ...);  % stochastic aTTC-CBF
```

The paper's headline results use the deterministic pair (`hocbf` vs
`attc_vraw`), which is what the example scripts run.  The stochastic
variants (`shocbf`, `sattc_vraw`) add a chance constraint on the barrier
under the process noise (parameters `scbf_p`, `scbf_alpha`, `scbf_beta`)
and are included for completeness.

`attc_vraw` loads its network from
`../Results/nn_bs128_h128-128-64_vraw_<attc_loss>_<attc_data_tag>/attc_nn_weights.mat`
(relative to this folder). The shipped model is selected by the defaults
`attc_loss = 'iw_huber3'`, `attc_data_tag = 'data1'`.

Key aTTC-CBF knobs (all name-value pairs to `auspice_sim`):

- `attc_tau_min` — safety threshold τ_c [s]: the filter enforces aTTC ≥ τ_min.
- `attc_activate_tau` — constraint only imposed when predicted aTTC < this [s].
- `attc_alpha` — class-K gain in the CBF condition.
- `*_adv` variants (`attc_tau_min_adv`, ...) — separate values for
  evader-pursuer pairs vs evader-evader pairs.

## Training-data generation (for retraining the NN)

| File | Purpose |
|---|---|
| `generate_training_data.m` | Runs two-agent scenario simulations and labels every timestep with the aTTC oracle at five assumed pursuer speeds. Produces the 20 `training_data_data1_*.mat` files consumed by the Python trainer. |
| `compute_attc_trajectory.m` | The aTTC *oracle*: for each timestep and each pair, forward-integrates the pursuit dynamics under maximal adversarial control until collision (or the horizon). Expensive; this is what the NN learns to approximate. |

## Analysis / figures

| File | Purpose |
|---|---|
| `auspice_plot_journal.m` | Statistical figures for the independent (random-sphere) experiments: collision-rate / waypoint-rate / spread bar charts, near-miss P(collision) vs look-ahead, and outcome-conditioned maneuver statistics. |
| `auspice_plot_journal_formation.m` | Same for the formation experiments + minimum-enclosing-ball (MEB) formation-coherence PDF/CDF figures. |
| `sweep_cbf_extract_metrics.m` | Summary metrics (collision counts, waypoint progress, ...) from one result struct. |
| `nearmiss_window_extract.m` | Near-miss event extraction used by the plotters. |
| `min_enclosing_ball_radius.m` | Approximate (Frank-Wolfe) minimum enclosing ball; validated to <0.5% error. |
| `compute_formation_coherence_helpers.m` | Formation-coherence helper metrics. |

## Examples (`examples/`)

| Script | What it does | Runtime |
|---|---|---|
| `example_quickstart.m` | 3 short pursuit runs (none / HOCBF / aTTC-CBF), side-by-side trajectory plot. | minutes |
| `reproduce_random_sphere_experiments.m` | The paper's 9 independent-pursuit cells at T = 50,000 s. | hours/cell |
| `reproduce_formation_experiments.m` | The paper's 4 formation cells at T = 50,000 s. | hours/cell |
| `plot_paper_figures.m` | All statistical figures + console summary tables from the saved cells. | minutes |

Both `reproduce_*` scripts accept a shorter horizon argument (e.g.
`reproduce_random_sphere_experiments(1000)`) for pipeline sanity checks, and
skip cells whose output file already exists — so they can be re-run to fill
in missing cells or resumed after interruption.
