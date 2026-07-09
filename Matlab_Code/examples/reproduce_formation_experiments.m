function reproduce_formation_experiments(T_sim)
%REPRODUCE_FORMATION_EXPERIMENTS  Paper experiments: formation-flight
%   pursuit evasion (Table "per-cell statistics", formation scenarios).
%
%   reproduce_formation_experiments()        % full T = 50,000 s runs
%   reproduce_formation_experiments(1000)    % shorter sanity runs
%
%   Runs 4 cells = 2 missions x 2 CBF variants:
%     missions:  formation_pursuit, 8 formation evaders + 3 pursuers,
%                pursuit advantage 0.9 (slow) and 1.5 (fast)
%     variants:  hocbf  /  attc_vraw (iw_huber3 / data1)
%
%   Formation spacing v_space = 2 km; the CBF activation scales are
%   adjusted from the random-sphere values so the safety filter does not
%   break formation under nominal (non-adversarial) conditions.
%
%   Each cell is saved to ../Results/paper_formation/<tag>__<cbf>.mat.
%   At the paper's T = 50,000 s each cell takes several hours.

if nargin < 1 || isempty(T_sim), T_sim = 50000; end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', ...
                   'Results', 'paper_formation');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

seed    = 42;
v_space = 2;          % formation slot spacing [km]
v_max   = 0.5;        % evader speed cap [km/s] (attc_activate_tau = v_space/v_max)

missions = {
    struct('pursuit_advantage',0.9, 'tag','formation_pursuit_adv0p9'), ...
    struct('pursuit_advantage',1.5, 'tag','formation_pursuit_adv1p5')};

% CBF hyperparameters — scaled with v_space (identical to the paper)
hocbf_args = { ...
    'cbf_type', 'hocbf', ...
    'cbf_alpha1',            0.1, ...
    'cbf_activate_dist',     v_space * 0.9, ...
    'cbf_activate_dist_adv', 16, ...
    'r_cbf',                 v_space * 0.5, ...
    'r_cbf_adv',             1};
attc_args = { ...
    'cbf_type', 'attc_vraw', ...
    'attc_loss',             'iw_huber3', ...
    'attc_data_tag',         'data1', ...
    'attc_alpha',            0.1, ...
    'attc_activate_tau',     v_space / v_max, ...
    'attc_activate_tau_adv', 20, ...
    'attc_tau_min',          1, ...
    'attc_tau_min_adv',      5, ...
    'r_cbf',                 1};

variants = {
    struct('suffix', 'hocbf',                         'args', {hocbf_args});
    struct('suffix', 'attc_vraw__iw_huber3__data1',   'args', {attc_args})};

for mi = 1:numel(missions)
    M = missions{mi};
    for vi = 1:numel(variants)
        V = variants{vi};
        out_file = fullfile(out_dir, sprintf('%s__%s.mat', M.tag, V.suffix));
        if exist(out_file, 'file')
            fprintf('[skip] %s already exists\n', out_file);
            continue;
        end

        fprintf('\n================================================\n');
        fprintf('  mission = %s   cbf = %s   T_sim = %d s\n', ...
                M.tag, V.suffix, T_sim);
        fprintf('================================================\n');

        rng(seed);
        t0 = tic;
        args = V.args;
        res = auspice_sim( ...
            'dim',               3, ...
            'mission',           'formation_pursuit', ...
            'num_agents',        11, ...
            'num_pursuers',      3, ...
            'pursuit_advantage', M.pursuit_advantage, ...
            'formation_spacing', v_space, ...
            'r_ttc',             0.1, ...
            'T_sim',             T_sim, ...
            'noise_std_mult',    1, ...
            'use_kalman_filter', false, ...
            'compute_attc',      false, ...
            args{:});
        run_seconds = toc(t0);

        metrics             = sweep_cbf_extract_metrics(res);
        metrics.cbf_type    = V.suffix;
        metrics.run_seconds = run_seconds;

        save(out_file, 'res', 'metrics', '-v7.3');
        fprintf('  saved %s   (%.1f s wall time)\n', out_file, run_seconds);
    end
end

fprintf('\nAll cells complete.  Run plot_paper_figures.m next.\n');
end
