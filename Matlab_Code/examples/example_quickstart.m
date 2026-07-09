%EXAMPLE_QUICKSTART  Minimal demo: 3 CBF variants on a short pursuit run.
%
%   Runs a 100-second 3D pursuit-evasion scenario (5 evaders + 2 faster
%   pursuers) three times — with no safety filter, with the distance-based
%   HOCBF, and with the aTTC-CBF — and plots the evader trajectories side
%   by side.  Total runtime is a few minutes on a laptop.
%
%   Run from anywhere:
%       >> cd <repo>/Matlab_Code/examples
%       >> example_quickstart

clear; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

T_sim = 100;                     % short demo horizon [s]
rng_seed = 42;

common = { ...
    'dim',               3, ...
    'mission',           'random_sphere_pursuit', ...
    'num_agents',        7, ...
    'num_pursuers',      2, ...
    'pursuit_advantage', 1.5, ...     % pursuers 50% faster than evaders
    'r_ttc',             0.1, ...
    'T_sim',             T_sim, ...
    'noise_std_mult',    1, ...
    'use_kalman_filter', false, ...
    'compute_attc',      false};

% CBF hyperparameters (same values as the paper experiments)
hocbf_args = { ...
    'cbf_type', 'hocbf', ...
    'cbf_alpha1',            0.1, ...
    'cbf_activate_dist',     4, ...
    'cbf_activate_dist_adv', 16, ...
    'r_cbf',                 0.5, ...
    'r_cbf_adv',             1};

attc_args = { ...
    'cbf_type', 'attc_vraw', ...           % loads the shipped data1 NN
    'attc_loss',             'iw_huber3', ...
    'attc_data_tag',         'data1', ...
    'attc_alpha',            0.1, ...
    'attc_activate_tau',     10, ...
    'attc_activate_tau_adv', 20, ...
    'attc_tau_min',          1, ...
    'attc_tau_min_adv',      5, ...
    'r_cbf',                 1};

none_args = {'cbf_type', 'none', 'r_cbf', 1};

variants = {
    struct('label', 'no CBF',   'args', {none_args});
    struct('label', 'HOCBF',    'args', {hocbf_args});
    struct('label', 'aTTC-CBF', 'args', {attc_args})};

results = cell(3, 1);
for k = 1:3
    fprintf('\n=== [%d/3] %s ===\n', k, variants{k}.label);
    rng(rng_seed);                        % identical initial conditions
    args = variants{k}.args;
    results{k} = auspice_sim(common{:}, args{:});
end

%% --- Plot evader trajectories side by side ---
fig = figure('Color', 'w', 'Position', [80 80 1500 500]);
for k = 1:3
    res = results{k};
    em  = logical(res.params.evader_mask(:));
    ax  = subplot(1, 3, k); hold(ax, 'on'); grid(ax, 'on');

    for a = 1:res.params.num_agents
        p = squeeze(res.x(:, a, 1:3));
        if em(a)
            plot3(ax, p(:,1), p(:,2), p(:,3), '-', ...
                'Color', [0.00 0.45 0.74], 'LineWidth', 1.0);
        else
            plot3(ax, p(:,1), p(:,2), p(:,3), '-', ...
                'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
        end
        scatter3(ax, p(end,1), p(end,2), p(end,3), 50, 'filled', ...
            'MarkerFaceColor', [0.2 0.2 0.2]);
    end

    % Count collisions (inter-agent distance < 2*r_ttc, rising edge)
    d = res.dist;
    n_col = 0;
    for pp = 1:size(d, 2)
        below = d(:, pp) < 2 * res.params.r_ttc;
        n_col = n_col + sum(diff([false; below]) == 1);
    end

    axis(ax, 'equal');
    xlabel(ax, 'x [km]'); ylabel(ax, 'y [km]'); zlabel(ax, 'z [km]');
    view(ax, 32, 18);
    title(ax, sprintf('%s   (%d collision events)', ...
        variants{k}.label, n_col), 'FontWeight', 'normal');
end
sgtitle(fig, sprintf(['Pursuit evasion, T = %d s.  ' ...
    'Blue = evaders, red = pursuers.'], T_sim));

fprintf('\nDone.  Blue evaders, red pursuers; titles show collision counts.\n');
