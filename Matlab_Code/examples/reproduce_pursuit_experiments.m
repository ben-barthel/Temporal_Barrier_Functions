function reproduce_pursuit_experiments(T_sim)
%REPRODUCE_PURSUIT_EXPERIMENTS  Paper experiment: pure pursuit-evasion.
%
%   reproduce_pursuit_experiments()        % full T = 50,000 s runs (paper)
%   reproduce_pursuit_experiments(2000)    % shorter sanity runs
%
%   One evader orbiting a fixed waypoint against five pursuers that are 50%
%   faster, run twice: once with the stochastic aTTC-CBF and once with the
%   stochastic higher-order (distance) CBF.  This is the "pure pursuit"
%   scenario behind the paper's headline table:
%
%       metric                            aTTC-SCBF     sHOCBF
%       ------------------------------    ---------     ------
%       collision rate  [per 100 s]            3.7       16.2
%       collisions / close encounter          0.50       0.69
%
%   A collision is a pair-distance below 2*r_ttc = 0.2 km; a close encounter
%   is below 4*r_ttc = 0.4 km.  Both are counted as rising edges, per
%   evader-pursuer pair, so a single sustained approach counts once.
%
%   Each run is saved to ../Results/paper_pursuit/<suffix>.mat with the raw
%   result struct (res) and the summary metrics.  At the paper's
%   T = 50,000 s each run takes a few hours on a laptop; the two are
%   independent and can be run in either order or on separate machines.
%
%   Once both runs exist the script calls auspice_plot_pursuit_compare on the
%   pair, which writes the comparison figure set (summary bars, distance and
%   orbit distributions, control-at-collision scatters, proximity CDF,
%   evasion-channel and min-distance paper figures, and one trajectory plot
%   per method) into the same folder.  Holding both T = 50,000 s result
%   structs in memory at once needs roughly 1 GB.
%
%   Reproducibility.  The initial conditions are drawn randomly, so the run
%   is seeded with rng(42) below.  The paper's stored runs did not record
%   their seed, so this reproduces the experiment statistically rather than
%   bit-for-bit: at T = 50,000 s each arm accumulates thousands of collision
%   events, and the rates above are stable to the quoted precision.

if nargin < 1 || isempty(T_sim)
    T_sim = 50000;                 % paper value
end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
out_dir = fullfile(here, '..', '..', 'Results', 'paper_pursuit');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

seed = 42;

%% ---- Scenario (values read from the paper's stored runs) --------------
% 6 agents = 1 evader (agent 1, CBF-protected) + 5 pursuers (uninhibited).
% The evader cruises at v_desired = 0.25 km/s with v_max = 0.5 km/s; the
% pursuers get pursuit_advantage * v_max = 0.75 km/s and fly at it.
% Control limits (omega_max 0.4, nu_max 0.1, a_max 0.05 rad/s, rad/s, km/s^2)
% and the process-noise level are the auspice_sim defaults used in the paper.
scenario = { ...
    'dim',                  3, ...
    'mission',              'pursuit', ...
    'num_agents',           6, ...
    'num_pursuers',         5, ...
    'pursuit_advantage',    1.5, ...      % pursuers 50% faster
    'v_desired',            0.25, ...
    'v_max',                0.5, ...
    'r_ttc',                0.1, ...      % collision at 2*r_ttc = 0.2 km
    'T_sim',                T_sim, ...
    'dt',                   0.1, ...
    'noise_std_mult',       1, ...
    'use_kalman_filter',    false, ...
    'use_chance_constraint', false, ...
    'compute_ttc',          true, ...
    'compute_attc',         false};

%% ---- CBF hyperparameters (as in the paper) ---------------------------
% Stochastic aTTC-CBF.  Uses the network shipped in ../Results/.  The paper's
% stored runs name this network 'wPur4b' after its training-data tag; it was
% renamed 'data1' for release.  The exported attc_nn_weights.mat is
% byte-identical -- only the provenance strings in config.json differ.
% The evader-pursuer ("_adv") thresholds are looser than the canonical ones
% because an adversarial pair needs to be engaged earlier and released later.
sattc_args = { ...
    'cbf_type',              'sattc_vraw', ...
    'attc_loss',             'iw_huber3', ...
    'attc_data_tag',         'data1', ...
    'attc_alpha',            0.1, ...
    'attc_activate_tau',     10, ...      % engage below 10 s aTTC (ee pairs)
    'attc_activate_tau_adv', 20, ...      % ... and below 20 s for ep pairs
    'attc_tau_min',          1, ...       % h = aTTC - tau_min
    'attc_tau_min_adv',      5, ...
    'r_cbf',                 1};

% Stochastic higher-order (distance) CBF baseline.
shocbf_args = { ...
    'cbf_type',              'shocbf', ...
    'cbf_alpha1',            0.1, ...
    'cbf_alpha2',            1, ...
    'cbf_activate_dist',     4, ...       % engage below 4 km (ee pairs)
    'cbf_activate_dist_adv', 16, ...      % ... and below 16 km for ep pairs
    'r_cbf',                 0.5, ...
    'r_cbf_adv',             1};

% Stochastic-CBF parameters shared by both arms (auspice_sim defaults, listed
% here because they set the probabilistic safety bound of Theorem II.3).
scbf_args = {'scbf_p', 0.9, 'scbf_alpha', 0.1, 'scbf_beta', 0.01};

variants = {
    struct('label', 'aTTC-SCBF', 'suffix', 'sattc_vraw__iw_huber3__data1', ...
           'args', {sattc_args});
    struct('label', 'sHOCBF',    'suffix', 'shocbf', ...
           'args', {shocbf_args})};

%% ---- Run --------------------------------------------------------------
all_metrics = cell(numel(variants), 1);
all_res     = cell(numel(variants), 1);

for vi = 1:numel(variants)
    V = variants{vi};
    out_file = fullfile(out_dir, sprintf('%s.mat', V.suffix));
    if exist(out_file, 'file')
        fprintf('[skip] %s already exists\n', out_file);
        S = load(out_file, 'res');
        all_res{vi} = S.res;
        % Recompute rather than trusting the stored metrics, so an old file
        % written under a different metric definition still reports correctly.
        m           = i_pursuit_metrics(S.res);
        m.label     = V.label;
        all_metrics{vi} = m;
        continue;
    end

    fprintf('\n================================================\n');
    fprintf('  pursuit   cbf = %s   T_sim = %g s\n', V.label, T_sim);
    fprintf('================================================\n');

    rng(seed);
    t0  = tic;
    res = auspice_sim(scenario{:}, scbf_args{:}, V.args{:});
    run_seconds = toc(t0);

    metrics             = i_pursuit_metrics(res);
    metrics.label       = V.label;
    metrics.run_seconds = run_seconds;
    all_metrics{vi}     = metrics;
    all_res{vi}         = res;

    save(out_file, 'res', 'metrics', '-v7.3');
    fprintf('  saved %s   (%.1f s wall time)\n', out_file, run_seconds);
end

%% ---- Headline table ---------------------------------------------------
fprintf('\n\n================ pursuit headline results ================\n');
fprintf('%-12s %22s %28s\n', 'CBF', 'collisions per 100 s', 'collisions / close encounter');
fprintf('%s\n', repmat('-', 1, 64));
for vi = 1:numel(all_metrics)
    m = all_metrics{vi};
    if isempty(m), continue; end
    fprintf('%-12s %22.1f %28.2f\n', m.label, m.coll_rate_per100s, m.coll_per_encounter);
end
fprintf('\nPaper values: aTTC-SCBF 3.7 / 0.50,  sHOCBF 16.2 / 0.69\n\n');

%% ---- Comparison figures ------------------------------------------------
% auspice_plot_pursuit_compare takes the result structs directly and writes
% the full figure set.  Order matters only for plot colours and legend order:
% the aTTC arm is passed first so it takes the first palette entry.
if all(~cellfun(@isempty, all_res))
    fprintf('Generating comparison figures in %s ...\n', out_dir);
    auspice_plot_pursuit_compare(all_res{:}, ...
        'save_dir',   out_dir, ...
        'save_plots', true, ...
        'make_movie', false);
else
    fprintf(['Skipping figures: both runs must be present.  Re-run this ' ...
             'script to fill in the missing arm.\n']);
end
end


%% ========================================================================
%  Headline metrics for the pursuit scenario
%  Counts rising edges of the evader-pursuer pair distance crossing below
%  2*r_col (collision) and 4*r_col (close encounter).  Only pairs involving
%  the evader (agent 1) are counted, which for the 'pursuit' mission is the
%  first na-1 columns of res.dist.
% =========================================================================
function m = i_pursuit_metrics(res)
    p  = res.params;
    na = double(p.num_agents);
    if isfield(p, 'r_ttc') && double(p.r_ttc) > 0
        r_col = double(p.r_ttc);
    else
        r_col = double(p.r_cbf);
    end
    d_coll = 2 * r_col;
    d_enc  = 4 * r_col;

    dist = double(res.dist);
    ego_pairs = 1:(na - 1);

    % Per-pursuer counts.  The paper's "collisions / close encounter" is the
    % MEAN OVER PURSUERS of each pursuer's own ratio, not the pooled ratio --
    % this is what auspice_plot_pursuit_compare reports, and the two differ
    % whenever the pursuers are not equally engaged.
    np_       = numel(ego_pairs);
    col_per_p = zeros(np_, 1);
    enc_per_p = zeros(np_, 1);
    for q = 1:np_
        dq           = dist(:, ego_pairs(q));
        col_per_p(q) = sum(diff([false; dq < d_coll]) == 1);
        enc_per_p(q) = sum(diff([false; dq < d_enc ]) == 1);
    end
    n_coll = sum(col_per_p);
    n_enc  = sum(enc_per_p);

    T = double(p.T_sim);
    m.n_collisions        = n_coll;
    m.n_close_encounters  = n_enc;
    m.coll_per_pursuer    = col_per_p;
    m.enc_per_pursuer     = enc_per_p;
    m.coll_rate_per100s   = n_coll / (T / 100);
    m.coll_per_encounter  = mean(col_per_p ./ max(enc_per_p, 1));
    m.min_dist            = min(dist(:));
    m.collision_radius_km = d_coll;
    m.encounter_radius_km = d_enc;
    m.T_sim               = T;

    fprintf('  %d collisions (d < %.2f km), %d close encounters (d < %.2f km)\n', ...
            n_coll, d_coll, n_enc, d_enc);
    fprintf('  %.2f collisions per 100 s   |   %.3f collisions per close encounter\n', ...
            m.coll_rate_per100s, m.coll_per_encounter);
end
