function auspice_plot_pursuit_compare(varargin)
%AUSPICE_PLOT_PURSUIT_COMPARE  Compare 1–N CBF methods on pursuit mission.
%
%   auspice_plot_pursuit_compare(results1)
%   auspice_plot_pursuit_compare(results1, results2, results3)
%   auspice_plot_pursuit_compare(r1, r2, r3, 'save_dir', '../Results', 'make_movie', true)
%
%   Inputs:
%     results1..N — structs from auspice_sim or JAX .mat export
%
%   Optional name-value pairs (after all result structs):
%     'save_dir'   — directory for output (default: pwd)
%     'make_movie' — generate trajectory movies (default: false)
%
%   Optional name-value pairs (cont.):
%     'prox_n_max' - x-axis limit of the proximity CDF, in units of r_col
%                    (default: 40)
%
%   A summary table (collision rate, collisions per close encounter, mean and
%   median distance to the nearest pursuer) is printed to the command window.
%
%   Figures:
%     1:      2×2 summary (collision rate, encounter rate, speed scatter, dist PDF)
%     2:      2×2 distribution comparison (min-, mean-, orbit-distance PDFs)
%     3:      control at collision (speed, turn rate, acceleration)
%     4:      time fraction with d_min < n*r_col, linear and log scale
%     5:      velocity components v*cos(gamma) / v*sin(gamma), 4-column axis
%     6:      min-distance PDF, standalone (paper figure)
%     7:      yaw rate vs pitch rate, one 4-column axis (paper figure)
%     8:      lateral vs vertical manoeuvre acceleration (paper figure)
%     9–N+8:  trajectory plot for each method
%     (optional) movies for each method, capped at 500 s

%% Parse inputs: separate result structs from name-value pairs
n_results = 0;
for k = 1:length(varargin)
    if isstruct(varargin{k})
        n_results = n_results + 1;
    else
        break;
    end
end
all_results = varargin(1:n_results);
remaining = varargin(n_results+1:end);

ip = inputParser;
addParameter(ip, 'save_dir', pwd, @ischar);
addParameter(ip, 'make_movie', false, @islogical);
addParameter(ip, 'save_plots', false, @islogical);   % opt-in: save figures
addParameter(ip, 'plot_scale', [1.1 1.5], @(x) isnumeric(x) && any(numel(x)==[1 2]) && all(x > 0));
addParameter(ip, 'prox_n_max', 40, @(x) isnumeric(x) && isscalar(x) && x > 0);
parse(ip, remaining{:});
save_dir   = ip.Results.save_dir;
make_movie = ip.Results.make_movie;
save_plots = ip.Results.save_plots;
prox_n_max = ip.Results.prox_n_max;   % x-axis limit of the proximity CDF, in units of r_col
plot_scale = ip.Results.plot_scale;   % saved-figure size: [width_scale height_scale] (scalar=uniform)

if ~exist(save_dir, 'dir'), mkdir(save_dir); end
assert(n_results >= 1, 'At least one results struct required.');

%% Plotting conventions
fn = 'Times New Roman';
f_scale = 0.75;
fs_label  = 20*f_scale;
fs_title  = 14*f_scale;
fs_tick   = 20*f_scale;
fs_legend = 20*f_scale;
lw = 2.5;

t_encounter_radius = 10;

% Paper typography -- used ONLY by Figures 5 and 6, which are meant to drop
% into a double-column layout.  Rendered ~7.3 in wide and included at
% \columnwidth (~3.4 in), 20 pt in-figure text lands at ~9 pt on the page.
% Those two are also saved at scale 1: i_save_fig's default 1.1x1.5 enlarges
% the canvas without enlarging text, which would shrink it on the page.
fs_label_p  = 20;
fs_tick_p   = 20;
fs_legend_p = 18;
fs_title_p  = 22;
lw_p        = 3;
ms_p        = 16;

% Color palette and line styles
color_palette = [
    0.00 0.45 0.74;   % blue
    0.85 0.33 0.10;   % orange
    0.47 0.67 0.19;   % green
    0.64 0.08 0.18;   % dark red
    0.49 0.18 0.56;   % purple
    0.30 0.75 0.93;   % light blue
];
line_styles = {'-', '--', '-.', ':', '-', '--'};

%% Process all results
infos  = cell(n_results, 1);
colors = zeros(n_results, 3);
styles = cell(n_results, 1);
labels = cell(n_results, 1);
label_map = containers.Map( ...
    {'hocbf', 'shocbf', 'rff', 'none', 'attc_vraw', 'sattc_vraw', 'hybrid'}, ...
    {'HOCBF', 'HOCBF', 'RFF', 'None', 'aTTC-CBF', 'aTTC-CBF', 'Hybrid'});
for k = 1:n_results
    infos{k}  = extract_pursuit_info(all_results{k}, t_encounter_radius);
    colors(k,:) = color_palette(mod(k-1, size(color_palette,1)) + 1, :);
    styles{k}   = line_styles{mod(k-1, length(line_styles)) + 1};
    ctype = lower(strtrim(infos{k}.cbf_type));
    if isKey(label_map, ctype)
        labels{k} = label_map(ctype);
    else
        labels{k} = upper(infos{k}.cbf_type);
    end
end
title_str = strjoin(labels, ' vs ');

% v_max for ego
vm = all_results{1}.params.v_max;
if numel(vm) > 1, v_max_ego = double(vm(1)); else, v_max_ego = double(vm); end
if infos{1}.dim == 2, v_idx = 4; else, v_idx = 6; end
ego_pairs = 1:(infos{1}.na - 1);

%% ========================================================================
%  Command-window summary
%    coll/100 s      — ego-vs-pursuer collision rate (rising edges of d < 2*r_col)
%    coll/encounter  — mean over pursuers of collisions / close encounters
%                      (close encounter = rising edge of d < 4*r_col)
%    mean min d      — time-mean of the distance to the nearest pursuer
%    median min d    - time-median of the same quantity
% =========================================================================
sum_col_rate = zeros(n_results, 1);
sum_enc_frac = zeros(n_results, 1);
sum_min_dist = zeros(n_results, 1);
sum_med_dist = zeros(n_results, 1);
for k = 1:n_results
    T_k = double(all_results{k}.params.T_sim);
    sum_col_rate(k) = infos{k}.total_collisions / (T_k / 100);

    [cl_k, co_k]    = count_per_pursuer(all_results{k}, infos{k});
    sum_enc_frac(k) = mean(co_k ./ max(cl_k, 1));

    dmin_k          = min(double(all_results{k}.dist(:, ego_pairs)), [], 2);
    sum_min_dist(k) = mean(dmin_k);
    sum_med_dist(k) = median(dmin_k);
end

lab_w = max([cellfun(@length, labels(:))', 6]);
fmt_h = sprintf('%%-%ds  %%12s  %%16s  %%18s  %%20s\n', lab_w);
fmt_r = sprintf('%%-%ds  %%12.3f  %%16.3f  %%18.4f  %%20.4f\n', lab_w);
fprintf('\n');
fprintf('Pursuit summary  (collision < %.3f km, close encounter < %.3f km)\n', ...
    infos{1}.threshold, infos{1}.encounter_threshold);
fprintf(fmt_h, 'Method', 'coll/100 s', 'coll/encounter', 'mean min d [km]', 'median min d [km]');
fprintf('%s\n', repmat('-', 1, lab_w + 74));
for k = 1:n_results
    fprintf(fmt_r, labels{k}, sum_col_rate(k), sum_enc_frac(k), sum_min_dist(k), sum_med_dist(k));
end
fprintf('\n');

%% ========================================================================
%  Figure 1: 2×2 summary
% =========================================================================
h_summary = figure('Position', [50 50 1000 750], 'Color', 'w');

% --- (a) Collision rate per 100 s ---
subplot(2, 2, 1); hold on; grid on;
col_rates = zeros(n_results, 1);
for k = 1:n_results
    T_k = double(all_results{k}.params.T_sim);
    col_rates(k) = infos{k}.total_collisions / (T_k / 100);
end
bh = bar(1:n_results, col_rates, 0.6, 'EdgeColor', 'k');
for k = 1:n_results
    bh.FaceColor = 'flat';
    bh.CData(k,:) = colors(k,:);
end
xticks(1:n_results); xticklabels(labels);
ylabel('Collisions per 100 s', 'FontName', fn, 'FontSize', fs_label);
title('(a) Collision Rate', 'FontName', fn, 'FontSize', fs_title);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

% --- (b) Collision / close encounter rate (per pursuer) ---
subplot(2, 2, 3); hold on; grid on;
n_pursuers = infos{1}.na - 1;
rate_matrix = zeros(n_pursuers, n_results);   % rows=pursuers, cols=methods
for k = 1:n_results
    [cl, co] = count_per_pursuer(all_results{k}, infos{k});
    rate_matrix(:, k) = co ./ max(cl, 1);
end
bh2 = bar(1:n_pursuers, rate_matrix, 'grouped', 'EdgeColor', 'k');
for k = 1:n_results
    bh2(k).FaceColor = colors(k,:);
end
xticks(1:n_pursuers);
pursuer_labels = arrayfun(@(i) sprintf('P%d', i+1), 1:n_pursuers, 'UniformOutput', false);
xticklabels(pursuer_labels);
xlabel('Pursuer', 'FontName', fn, 'FontSize', fs_label);
ylabel('Collision / Close Encounter', 'FontName', fn, 'FontSize', fs_label);
title('(b) Encounter-to-Collision Rate', 'FontName', fn, 'FontSize', fs_title);
legend(bh2, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
ylim([0 1.05]);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

% --- (c) Ego speed at collision (scatter + diamond mean) ---
subplot(2, 2, 2); hold on; grid on;
for k = 1:n_results
    [spds, ~] = get_collision_stats(all_results{k}, infos{k});
    if ~isempty(spds)
        x_jit = k + 0.05 * randn(size(spds));
        scatter(x_jit, spds, 30, colors(k,:), 'filled', ...
            'MarkerFaceAlpha', 0.15, 'HandleVisibility', 'off');
        plot(k, mean(spds), 'd', 'Color', colors(k,:), ...
            'MarkerFaceColor', colors(k,:), 'MarkerSize', 14, 'HandleVisibility', 'off');
    end
end
yline(v_max_ego, '--k', 'LineWidth', 1.5);
xlim([0.5 n_results+0.5]); xticks(1:n_results); xticklabels(labels);
ylabel('Ego speed at collision [km/s]', 'FontName', fn, 'FontSize', fs_label);
title('(a) Ego Speed at Collision', 'FontName', fn, 'FontSize', fs_title);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

% --- (d) Min-distance PDF ---
subplot(2, 2, 4); hold on; grid on;
n_bins_d = 400;
edges_d = linspace(0, 2, n_bins_d + 1);
bin_w_d = edges_d(2) - edges_d(1);
centers_d = edges_d(1:end-1) + bin_w_d / 2;
h_pdf = gobjects(n_results, 1);
for k = 1:n_results
    ego_dk = min(double(all_results{k}.dist(:, ego_pairs)), [], 2);
    [cdk, ~] = histcounts(ego_dk, edges_d);
    pdfdk = cdk / (sum(cdk) * bin_w_d);
    h_pdf(k) = plot(centers_d, pdfdk, styles{k}, 'Color', colors(k,:), 'LineWidth', lw);
end
xline(infos{1}.threshold, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xlabel('Min distance to pursuer [km]', 'FontName', fn, 'FontSize', fs_label);
ylabel('PDF', 'FontName', fn, 'FontSize', fs_label);
title('(b) Min-Distance Distribution', 'FontName', fn, 'FontSize', fs_title);
legend(h_pdf, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
xlim([0, 2]);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

sgtitle(title_str, 'FontName', fn, 'FontSize', fs_title + 2, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 2: Distribution comparison (2×2, fourth slot intentionally empty)
%    (a) Min-distance PDF     (b) Mean-distance PDF
%    (c) Orbit-distance PDF
% =========================================================================
h_dist = figure('Position', [80 50 1100 850], 'Color', 'w');

n_bins_pdf = 120;

% --- (a) Min-distance PDF ---
subplot(2, 2, 1); hold on; grid on;
edges_md = linspace(0, 2, n_bins_pdf + 1);
bw_md = edges_md(2) - edges_md(1);
ctr_md = edges_md(1:end-1) + bw_md / 2;
h_md = gobjects(n_results, 1);
for k = 1:n_results
    ego_dk = min(double(all_results{k}.dist(:, ego_pairs)), [], 2);
    [ck, ~] = histcounts(ego_dk, edges_md);
    pdfk = ck / (sum(ck) * bw_md);
    h_md(k) = plot(ctr_md, pdfk, styles{k}, 'Color', colors(k,:), 'LineWidth', lw);
end
xline(infos{1}.threshold, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xlabel('Min distance to pursuer [km]', 'FontName', fn, 'FontSize', fs_label);
ylabel('PDF', 'FontName', fn, 'FontSize', fs_label);
title('(a) Min-Distance Distribution', 'FontName', fn, 'FontSize', fs_title);
legend(h_md, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
xlim([0, 2]);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

% --- (b) Mean-distance PDF ---
subplot(2, 2, 2); hold on; grid on;
% Match panel (a)'s bin width so the two distance PDFs are directly
% comparable.  Binning still spans the full 0-30 km support, so the density is
% normalised over all the data -- the xlim below only zooms, it does not
% condition the PDF on d < 5 km.
n_bins_mean = round(30 / bw_md);
edges_mean = linspace(0, 30, n_bins_mean + 1);
bw_mean = edges_mean(2) - edges_mean(1);
ctr_mean = edges_mean(1:end-1) + bw_mean / 2;
h_mean = gobjects(n_results, 1);
for k = 1:n_results
    ego_dk = mean(double(all_results{k}.dist(:, ego_pairs)), 2);
    [ck, ~] = histcounts(ego_dk, edges_mean);
    pdfk = ck / (sum(ck) * bw_mean);
    h_mean(k) = plot(ctr_mean, pdfk, styles{k}, 'Color', colors(k,:), 'LineWidth', lw);
end
xlabel('Mean distance to pursuers [km]', 'FontName', fn, 'FontSize', fs_label);
ylabel('PDF', 'FontName', fn, 'FontSize', fs_label);
title('(b) Mean-Distance Distribution', 'FontName', fn, 'FontSize', fs_title);
xlim([0 5]);
legend(h_mean, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
set(gca, 'FontName', fn, 'FontSize', fs_tick);

% --- (c) Distance-from-orbited-waypoint PDF ---
% The pursuit mission gives the evader ONE fixed waypoint (auspice_sim.m, case
% 'pursuit': [10*v_max, 0]), which it cannot stop at and therefore orbits.  The
% spread of this PDF is the orbit radius distribution.  Plotted over the whole
% record, so the initial inbound leg shows up as a tail at large distance.
subplot(2, 2, 3); hold on; grid on;
wp_ok = true;
d_wp  = cell(n_results, 1);
pdim  = infos{1}.dim;
for k = 1:n_results
    Pk = all_results{k}.params;
    if ~isfield(Pk, 'waypoints') || isempty(Pk.waypoints) || isempty(Pk.waypoints{1})
        wp_ok = false; break;
    end
    wp_k    = double(Pk.waypoints{1}(1, 1:pdim));
    pos_k   = reshape(double(all_results{k}.x(:, 1, 1:pdim)), [], pdim);
    d_wp{k} = vecnorm(pos_k - wp_k, 2, 2);
end
if wp_ok
    wp_hi   = max(cellfun(@max, d_wp));
    edges_w = linspace(0, wp_hi, n_bins_pdf + 1);
    bw_w    = edges_w(2) - edges_w(1);
    ctr_w   = edges_w(1:end-1) + bw_w / 2;
    h_wp    = gobjects(n_results, 1);
    for k = 1:n_results
        [ck, ~] = histcounts(d_wp{k}, edges_w);
        pdfk    = ck / (sum(ck) * bw_w);
        h_wp(k) = plot(ctr_w, pdfk, styles{k}, 'Color', colors(k,:), 'LineWidth', lw);
    end
    legend(h_wp, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
else
    text(0.5, 0.5, 'no waypoint recorded', 'Units', 'normalized', ...
         'HorizontalAlignment', 'center', 'FontName', fn, 'FontSize', fs_label);
end
xlabel('Distance from orbited waypoint [km]', 'FontName', fn, 'FontSize', fs_label);
ylabel('PDF', 'FontName', fn, 'FontSize', fs_label);
title('(c) Orbit-Distance Distribution', 'FontName', fn, 'FontSize', fs_title);
set(gca, 'FontName', fn, 'FontSize', fs_tick);

sgtitle(sprintf('Distributions — %s', title_str), ...
    'FontName', fn, 'FontSize', fs_title + 2, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 3: Control-at-collision scatters, 2×2 (identical style to the
%  speed-at-collision panel in Figure 1).
%    (a) ego speed   (b) ego |turning rate|   (c) ego acceleration (signed)
%    (d) ego climb rate dz/dt (signed)
% =========================================================================
% Control limits for reference lines (fall back to defaults if absent)
if isfield(all_results{1}.params, 'omega_max')
    omega_max = double(all_results{1}.params.omega_max);
else
    omega_max = 0.4;
end
if isfield(all_results{1}.params, 'a_max')
    a_max = double(all_results{1}.params.a_max);
else
    a_max = 0.05;
end

% Gather collision stats once per method
cc_speed = cell(n_results, 1);
cc_turn  = cell(n_results, 1);
cc_acc   = cell(n_results, 1);
cc_climb = cell(n_results, 1);
for k = 1:n_results
    [cc_speed{k}, ~, cc_turn{k}, cc_acc{k}, cc_climb{k}] = ...
        get_collision_stats(all_results{k}, infos{k});
end

% Panel specs: {data, ylabel, title, ref_value, symmetric_ref, take_abs}
panels = { ...
    {cc_speed, 'Ego speed at collision [km/s]',          '(a) Speed at Collision',        v_max_ego, false, false}, ...
    {cc_turn,  'Ego |turn rate| at collision [rad/s]',   '(b) Turning Rate at Collision', omega_max, false, true }, ...
    {cc_acc,   'Ego acceleration at collision [km/s^2]', '(c) Acceleration at Collision', a_max,     true,  false}, ...
    {cc_climb, 'Ego climb rate at collision [km/s]',     '(d) Climb Rate at Collision',   v_max_ego, true,  false} };
% Climb rate is dz/dt = v*sin(gamma), sign kept (negative = descending).  The
% model has no gamma limit -- only nu_max on the gamma RATE -- so the reference
% lines are +/- v_max, the kinematic bound reached in pure vertical flight.

h_ctrl = figure('Position', [110 50 1100 850], 'Color', 'w');
for s = 1:numel(panels)
    data    = panels{s}{1};
    ylab    = panels{s}{2};
    ttl     = panels{s}{3};
    ref     = panels{s}{4};
    sym_ref = panels{s}{5};
    do_abs  = panels{s}{6};

    subplot(2, 2, s); hold on; grid on;
    for k = 1:n_results
        yv = data{k};
        if do_abs, yv = abs(yv); end
        if ~isempty(yv)
            x_jit = k + 0.05 * randn(size(yv));
            scatter(x_jit, yv, 30, colors(k,:), 'filled', ...
                'MarkerFaceAlpha', 0.15, 'HandleVisibility', 'off');
            plot(k, mean(yv), 'd', 'Color', colors(k,:), ...
                'MarkerFaceColor', colors(k,:), 'MarkerSize', 14, 'HandleVisibility', 'off');
        end
    end
    yline(ref, '--k', 'LineWidth', 1.5);
    if sym_ref, yline(-ref, '--k', 'LineWidth', 1.5); end
    xlim([0.5 n_results+0.5]); xticks(1:n_results); xticklabels(labels);
    ylabel(ylab, 'FontName', fn, 'FontSize', fs_label);
    title(ttl, 'FontName', fn, 'FontSize', fs_title);
    set(gca, 'FontName', fn, 'FontSize', fs_tick);
end
sgtitle(sprintf('Control at Collision — %s', title_str), ...
    'FontName', fn, 'FontSize', fs_title + 2, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 4: Time fraction with d_min < n * r_col
%    Empirical CDF of the distance to the nearest pursuer, with the abscissa
%    in units of r_col.  n = 2 is the collision threshold, n = 4 the close-
%    encounter threshold, so those two ordinates are the time fractions spent
%    in collision and in close encounter respectively.
% =========================================================================
n_grid = linspace(0, prox_n_max, 400);
prox_F = zeros(numel(n_grid), n_results);
for k = 1:n_results
    dmin_k = min(double(all_results{k}.dist(:, ego_pairs)), [], 2);
    edges  = [n_grid * infos{k}.r_col, inf];
    cnts   = histcounts(dmin_k, edges);
    % F(n_j) = fraction with d_min < n_j*r_col  ->  counts strictly below edge j
    prox_F(:, k) = [0, cumsum(cnts(1:end-1))]' / numel(dmin_k);
end

h_prox = figure('Position', [140 50 1200 480], 'Color', 'w');
for s = 1:2
    subplot(1, 2, s); hold on; grid on;
    h_pc = gobjects(n_results, 1);
    for k = 1:n_results
        h_pc(k) = plot(n_grid, prox_F(:, k), styles{k}, ...
            'Color', colors(k,:), 'LineWidth', lw);
    end
    xline(2, '--k', '2 r_{col}', 'LineWidth', 1.5, 'HandleVisibility', 'off', ...
        'FontName', fn, 'FontSize', fs_tick, 'LabelVerticalAlignment', 'bottom');
    xline(4, ':k',  '4 r_{col}', 'LineWidth', 1.5, 'HandleVisibility', 'off', ...
        'FontName', fn, 'FontSize', fs_tick, 'LabelVerticalAlignment', 'top');
    xlabel('n  [multiples of r_{col}]', 'FontName', fn, 'FontSize', fs_label);
    ylabel('Fraction of time with d_{min} < n r_{col}', 'FontName', fn, 'FontSize', fs_label);
    xlim([0 prox_n_max]);
    if s == 1
        ylim([0 1]);
        title('(a) Linear Scale', 'FontName', fn, 'FontSize', fs_title);
    else
        set(gca, 'YScale', 'log');
        y_lo = min(prox_F(prox_F > 0));
        if isempty(y_lo), y_lo = 1e-6; end
        ylim([10^floor(log10(y_lo)) 1]);
        title('(b) Log Scale', 'FontName', fn, 'FontSize', fs_title);
    end
    legend(h_pc, labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'southeast');
    set(gca, 'FontName', fn, 'FontSize', fs_tick);
end
sgtitle(sprintf('Time Fraction Below n r_{col} - %s', title_str), ...
    'FontName', fn, 'FontSize', fs_title + 2, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 5: Horizontal vs vertical evasion on one 4-column axis
%    Columns 1..n  = |turn rate| at collision  (LEFT axis,  rad/s)
%    Columns n+1.. = climb rate at collision   (RIGHT axis, km/s)
%  Same data as Figure 3 panels (b) and (d), co-located for comparison.  The
%  two halves carry different units and different bounds, so the left and
%  right scales are NOT comparable in magnitude -- read each half against its
%  own dashed limit, not against the other half.
% =========================================================================
opt_dual = struct('n_results', n_results, 'labels', {labels}, 'colors', colors, ...
                  'fn', fn, 'fs_label', fs_label_p, 'fs_tick', fs_tick_p, ...
                  'ms', ms_p, 'lw', lw_p);
% Velocity components at collision: v*cos(gamma) horizontally, v*sin(gamma)
% vertically.  These are the two legs of the velocity vector
% (v_h^2 + v_z^2 = v^2), so they share units AND a bound, and go on one axis.
% Note the horizontal leg carries no yaw dependence -- psi is an azimuth, it
% picks a direction inside the plane rather than projecting out of it, so
% there is no v*sin(psi) counterpart to v*sin(gamma).
cc_vh = cell(n_results, 1);
for k = 1:n_results
    cc_vh{k} = sqrt(max(cc_speed{k}.^2 - cc_climb{k}.^2, 0));   % = v*cos(gamma)
end
% Two axes rather than one shared: the horizontal leg is a magnitude and is
% clamped to [0, v_max], while the vertical leg keeps its sign over
% [-v_max, v_max].  NOTE the halves therefore have the same units but a 2x
% difference in scale -- equal heights across the divider are NOT equal
% speeds.  Read each half against its own axis.
opt_v = opt_dual;
opt_v.ylimL = [0, v_max_ego];
opt_v.ylimR = [-v_max_ego, v_max_ego];
h_evade = i_dual_axis_scatter( ...
    cc_vh, cc_climb, opt_v, ...
    'Ego horizontal speed at collision [km/s]', v_max_ego, false, ...
    'Ego climb rate at collision [km/s]',       v_max_ego, true);

%% ========================================================================
%  Figure 6: Min-distance PDF on its own axis (paper figure)
%  Same quantity as Figure 2 panel (a); duplicated standalone so it can be
%  dropped into a double-column layout without cropping a subplot.
% =========================================================================
h_mind = figure('Position', [170 50 700 580], 'Color', 'w');
hold on; grid on;
edges_p = linspace(0, 2, n_bins_pdf + 1);
bw_p    = edges_p(2) - edges_p(1);
ctr_p   = edges_p(1:end-1) + bw_p / 2;
h_mp    = gobjects(n_results, 1);
for k = 1:n_results
    ego_dk  = min(double(all_results{k}.dist(:, ego_pairs)), [], 2);
    [ck, ~] = histcounts(ego_dk, edges_p);
    pdfk    = ck / (sum(ck) * bw_p);
    h_mp(k) = plot(ctr_p, pdfk, styles{k}, 'Color', colors(k,:), 'LineWidth', lw_p);
end
xline(infos{1}.threshold, '--k', 'LineWidth', lw_p - 1, 'HandleVisibility', 'off');
xlabel('Min distance to pursuer [km]', 'FontName', fn, 'FontSize', fs_label_p);
ylabel('PDF', 'FontName', fn, 'FontSize', fs_label_p);
legend(h_mp, labels, 'FontName', fn, 'FontSize', fs_legend_p, 'Location', 'best');
xlim([0, 2]);
set(gca, 'FontName', fn, 'FontSize', fs_tick_p);

%% ========================================================================
%  Figure 7: Yaw rate vs pitch rate on one 4-column axis (paper figure)
%  Both are control channels in rad/s, so unlike Figure 5 the two halves are
%  the same physical quantity -- but their BOUNDS differ (omega_max = %g,
%  nu_max = %g), so they still get separate axes, each scaled to its own limit.
%  Yaw is plotted as |u_psi| (left/right is symmetric); pitch keeps its sign,
%  since nose-up and nose-down are not interchangeable.
% =========================================================================
cc_pitch = cell(n_results, 1);
for k = 1:n_results
    [~, ~, ~, ~, ~, cc_pitch{k}] = get_collision_stats(all_results{k}, infos{k});
end
if isfield(all_results{1}.params, 'nu_max')
    nu_max = double(all_results{1}.params.nu_max);
else
    nu_max = 0.1;
end
h_rates = i_dual_axis_scatter( ...
    cellfun(@abs, cc_turn, 'UniformOutput', false), cc_pitch, opt_dual, ...
    'Ego |yaw rate| at collision [rad/s]',   omega_max, false, ...
    'Ego pitch rate at collision [rad/s]',   nu_max,    true);

%% ========================================================================
%  Figure 8: Lateral vs vertical manoeuvre acceleration (paper figure)
%  The acceleration normal to the velocity vector, split by plane:
%      horizontal (centripetal)  a_lat  = v*cos(gamma) * u_psi
%      vertical   (normal)       a_vert = v * u_gamma
%  Both km/s^2, so this is the pairing in which "how hard is it manoeuvring
%  horizontally vs vertically" is a fair question.  Bounds differ because
%  omega_max and nu_max differ 4x, so the axes stay separate.
% =========================================================================
cc_alat  = cell(n_results, 1);
cc_avert = cell(n_results, 1);
for k = 1:n_results
    cc_alat{k}  = cc_vh{k}    .* cc_turn{k};      % v*cos(gamma)*psi_dot
    cc_avert{k} = cc_speed{k} .* cc_pitch{k};     % v*gamma_dot
end
h_accel = i_dual_axis_scatter( ...
    cellfun(@abs, cc_alat, 'UniformOutput', false), cc_avert, opt_dual, ...
    'Ego |lateral accel| at collision [km/s^2]', v_max_ego*omega_max, false, ...
    'Ego vertical accel at collision [km/s^2]',  v_max_ego*nu_max,    true);

%% ========================================================================
%  Figures 9–N+8: Trajectory plots
% =========================================================================
traj_duration = 500;  % seconds

color_evader  = [0.00 0.45 0.74];
h_traj_figs   = gobjects(n_results, 1);   % capture per-method trajectory figures

for k = 1:n_results
    R = all_results{k};
    info = infos{k};
    na  = info.na;
    dim = info.dim;
    x   = double(R.x);
    t   = double(R.t(:));

    % Truncate
    k_end = find(t <= traj_duration, 1, 'last');
    if isempty(k_end), k_end = length(t); end

    % Pursuer colors: shades of the method color
    p_colors = zeros(na, 3);
    p_colors(1,:) = color_evader;
    base = colors(k,:);
    for i = 2:na
        frac = (i - 2) / max(na - 2, 1);
        p_colors(i,:) = (1 - frac) * base + frac * (base * 0.4);
    end

    figure('Name', sprintf('Trajectories — %s', labels{k}), ...
        'Position', [60+40*k 80 850 700], 'Color', 'w');
    h_traj_figs(k) = gcf;

    if dim == 3
        hold on; grid on; axis equal;
        h_traj = gobjects(na, 1);
        for i = 1:na
            tx = squeeze(x(1:k_end, i, 1));
            ty = squeeze(x(1:k_end, i, 2));
            tz = squeeze(x(1:k_end, i, 3));
            h_traj(i) = plot3(tx, ty, tz, '-', 'Color', p_colors(i,:), 'LineWidth', 1.5);
            plot3(tx(1), ty(1), tz(1), 's', 'Color', p_colors(i,:), ...
                'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
            plot3(tx(end), ty(end), tz(end), 'o', 'Color', p_colors(i,:), ...
                'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end
        xlabel('x [km]', 'FontName', fn, 'FontSize', fs_label);
        ylabel('y [km]', 'FontName', fn, 'FontSize', fs_label);
        zlabel('z [km]', 'FontName', fn, 'FontSize', fs_label);
        view(30, 25);
    else
        hold on; grid on; axis equal;
        h_traj = gobjects(na, 1);
        for i = 1:na
            tx = squeeze(x(1:k_end, i, 1));
            ty = squeeze(x(1:k_end, i, 2));
            h_traj(i) = plot(tx, ty, '-', 'Color', p_colors(i,:), 'LineWidth', 1.5);
            plot(tx(1), ty(1), 's', 'Color', p_colors(i,:), ...
                'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
            plot(tx(end), ty(end), 'o', 'Color', p_colors(i,:), ...
                'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end
        xlabel('x [km]', 'FontName', fn, 'FontSize', fs_label);
        ylabel('y [km]', 'FontName', fn, 'FontSize', fs_label);
    end

    agent_labels = cell(1, na);
    agent_labels{1} = 'Evader';
    for i = 2:na, agent_labels{i} = sprintf('Pursuer %d', i); end
    legend(h_traj, agent_labels, 'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
    title(sprintf('%s — first %d s  (%d collisions)', ...
        labels{k}, traj_duration, infos{k}.total_collisions), ...
        'FontName', fn, 'FontSize', fs_title);
    set(gca, 'FontName', fn, 'FontSize', fs_tick);
end

%% ========================================================================
%  Save the static figures  (opt-in via 'save_plots', true)
% =========================================================================
if save_plots
    i_save_fig(h_summary, save_dir, 'pursuit_summary',              plot_scale);
    i_save_fig(h_dist,    save_dir, 'pursuit_distributions',        plot_scale);
    i_save_fig(h_ctrl,    save_dir, 'pursuit_control_at_collision', plot_scale);
    i_save_fig(h_prox,    save_dir, 'pursuit_proximity_cdf',       plot_scale);
    % scale 1: these two are sized for a paper column already
    i_save_fig(h_evade,   save_dir, 'pursuit_evasion_channels',    1);
    i_save_fig(h_mind,    save_dir, 'pursuit_min_distance_pdf',    1);
    i_save_fig(h_rates,   save_dir, 'pursuit_yaw_pitch_rates',     1);
    i_save_fig(h_accel,   save_dir, 'pursuit_manoeuvre_accel',     1);
    for kf = 1:n_results
        i_save_fig(h_traj_figs(kf), save_dir, sprintf('pursuit_traj_%s', labels{kf}), plot_scale);
    end
end

%% ========================================================================
%  Optional: Trajectory movies (capped at 500 s)
% =========================================================================
if make_movie
    movie_duration = 500;
    sim_speed      = 25;
    target_fps     = 30;
    trail_sec      = 20;

    for k = 1:n_results
        R = all_results{k};
        info = infos{k};
        na  = info.na;
        dim = info.dim;
        x   = double(R.x);
        t   = double(R.t(:));
        dt  = info.dt;
        dist_k = double(R.dist);

        k_end_movie = find(t <= movie_duration, 1, 'last');
        if isempty(k_end_movie), k_end_movie = length(t); end

        % Cumulative collisions
        threshold_k = infos{k}.threshold;
        col_ev = zeros(length(t), 1);
        for p = 1:(na-1)
            d_p = dist_k(:, p);
            col_ev = col_ev + double(diff([false; d_p < threshold_k]) == 1);
        end
        cum_col = cumsum(col_ev);

        % Frame setup
        fps_raw    = sim_speed / dt;
        frame_skip = max(1, round(fps_raw / target_fps));
        actual_fps = fps_raw / frame_skip;
        frame_idx  = 1:frame_skip:k_end_movie;
        if frame_idx(end) ~= k_end_movie
            frame_idx(end+1) = k_end_movie;
        end
        trail_steps = round(trail_sec / dt);

        % Axis limits
        if dim == 3
            ax_x = squeeze(x(1:k_end_movie,:,1));
            ax_y = squeeze(x(1:k_end_movie,:,2));
            ax_z = squeeze(x(1:k_end_movie,:,3));
            pad = 0.1;
            xl = [min(ax_x(:)) max(ax_x(:))] + [-pad pad].*max(range(ax_x(:)),1);
            yl = [min(ax_y(:)) max(ax_y(:))] + [-pad pad].*max(range(ax_y(:)),1);
            zl = [min(ax_z(:)) max(ax_z(:))] + [-pad pad].*max(range(ax_z(:)),1);
        else
            ax_x = squeeze(x(1:k_end_movie,:,1));
            ax_y = squeeze(x(1:k_end_movie,:,2));
            pad = 0.1;
            xl = [min(ax_x(:)) max(ax_x(:))] + [-pad pad].*max(range(ax_x(:)),1);
            yl = [min(ax_y(:)) max(ax_y(:))] + [-pad pad].*max(range(ax_y(:)),1);
        end

        vid_name = sprintf('pursuit_%s_%dD_%dagents_%ds', ...
            lower(labels{k}), dim, na, movie_duration);
        vid_path = fullfile(save_dir, [vid_name '.mp4']);
        vw = VideoWriter(vid_path, 'MPEG-4');
        vw.FrameRate = actual_fps;
        vw.Quality   = 95;
        open(vw);

        fprintf('  Writing video: %s  (%d frames)\n', vid_path, length(frame_idx));
        fig_vid = figure('Position', [100 100 900 700], 'Color', 'w', 'Visible', 'off');

        for fi = 1:length(frame_idx)
            kk = frame_idx(fi);
            clf(fig_vid);
            ax_v = axes(fig_vid); hold(ax_v, 'on'); grid(ax_v, 'on');
            k_trail = max(1, kk - trail_steps);

            if dim == 3
                for i = 2:na
                    plot3(ax_v, squeeze(x(k_trail:kk,i,1)), squeeze(x(k_trail:kk,i,2)), ...
                        squeeze(x(k_trail:kk,i,3)), '-', 'Color', [colors(k,:) 0.3], 'LineWidth', 1);
                end
                plot3(ax_v, squeeze(x(k_trail:kk,1,1)), squeeze(x(k_trail:kk,1,2)), ...
                    squeeze(x(k_trail:kk,1,3)), '-', 'Color', [color_evader 0.5], 'LineWidth', 1.5);
                for i = 2:na
                    plot3(ax_v, x(kk,i,1), x(kk,i,2), x(kk,i,3), 'o', ...
                        'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 6);
                end
                plot3(ax_v, x(kk,1,1), x(kk,1,2), x(kk,1,3), 'o', ...
                    'Color', color_evader, 'MarkerFaceColor', color_evader, 'MarkerSize', 10);
                xlim(ax_v, xl); ylim(ax_v, yl); zlim(ax_v, zl);
                zlabel(ax_v, 'z [km]', 'FontName', fn, 'FontSize', fs_label);
                view(ax_v, 30, 25);
            else
                for i = 2:na
                    plot(ax_v, squeeze(x(k_trail:kk,i,1)), squeeze(x(k_trail:kk,i,2)), ...
                        '-', 'Color', [colors(k,:) 0.3], 'LineWidth', 1);
                end
                plot(ax_v, squeeze(x(k_trail:kk,1,1)), squeeze(x(k_trail:kk,1,2)), ...
                    '-', 'Color', [color_evader 0.5], 'LineWidth', 1.5);
                for i = 2:na
                    plot(ax_v, x(kk,i,1), x(kk,i,2), 'o', ...
                        'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 6);
                end
                plot(ax_v, x(kk,1,1), x(kk,1,2), 'o', ...
                    'Color', color_evader, 'MarkerFaceColor', color_evader, 'MarkerSize', 10);
                xlim(ax_v, xl); ylim(ax_v, yl);
            end

            xlabel(ax_v, 'x [km]', 'FontName', fn, 'FontSize', fs_label);
            ylabel(ax_v, 'y [km]', 'FontName', fn, 'FontSize', fs_label);
            set(ax_v, 'FontName', fn, 'FontSize', fs_tick);
            title(ax_v, sprintf('%s  |  t = %.1f s  |  Collisions: %d', ...
                labels{k}, t(kk), cum_col(kk)), ...
                'FontName', fn, 'FontSize', fs_title, 'FontWeight', 'bold');

            drawnow;
            writeVideo(vw, getframe(fig_vid));

            if mod(fi, 200) == 0 || fi == length(frame_idx)
                fprintf('    Frame %d / %d\n', fi, length(frame_idx));
            end
        end

        close(vw); close(fig_vid);
        fprintf('  Video saved: %s\n', vid_path);
    end
end

end


%% ========================================================================
%  Helper: i_dual_axis_scatter
%  A 4-column (2*n_results) single-axis scatter: the first n_results columns
%  on the LEFT y-axis, the rest on the RIGHT.  Used for the paper figures that
%  put a horizontal and a vertical evasion channel side by side.  The two
%  halves carry different scales, so each reference line is drawn only over the
%  columns it governs and the axis colours stay neutral (columns are coloured
%  by method, so a coloured axis would imply a mapping that is not there).
% =========================================================================
function h = i_dual_axis_scatter(dataL, dataR, o, ylabL, refL, symL, ylabR, refR, symR, shared)
%   shared (optional): when the two halves carry the SAME units, pass a single
%   ylabel string and everything goes on one axis.  Two autoscaled axes with
%   identical units invite a misreading -- the eye compares heights, not scales.
    if nargin < 10, shared = ''; end
    use_shared = ~isempty(shared);
    h = figure('Position', [140 50 700 580], 'Color', 'w');
    hold on; grid on;
    n  = o.n_results;
    xL = 1:n;
    xR = n + (1:n);
    ax_col = [0.15 0.15 0.15];

    for side = 1:2
        if side == 1
            D = dataL; xv = xL; ylab = ylabL; rf = refL; sym = symL;
            if ~use_shared, yyaxis left; end
        else
            D = dataR; xv = xR; ylab = ylabR; rf = refR; sym = symR;
            if ~use_shared, yyaxis right; end
        end
        for k = 1:n
            yv = D{k};
            yv = yv(isfinite(yv));
            if isempty(yv), continue; end
            scatter(xv(k) + 0.05*randn(size(yv)), yv, 30, o.colors(k,:), 'filled', ...
                    'MarkerFaceAlpha', 0.15, 'HandleVisibility', 'off');
            plot(xv(k), mean(yv), 'd', 'Color', o.colors(k,:), ...
                 'MarkerFaceColor', o.colors(k,:), 'MarkerSize', o.ms, ...
                 'HandleVisibility', 'off');
        end
        % reference line(s) drawn only across this half's columns
        seg = [min(xv) - 0.5, max(xv) + 0.5];
        plot(seg, [rf rf], '--k', 'LineWidth', o.lw - 1, 'HandleVisibility', 'off');
        if sym
            plot(seg, [-rf -rf], '--k', 'LineWidth', o.lw - 1, 'HandleVisibility', 'off');
        end
        if ~use_shared
            ylabel(ylab, 'FontName', o.fn, 'FontSize', o.fs_label);
            set(gca, 'YColor', ax_col);
            % optional explicit range for this half
            if side == 1 && isfield(o, 'ylimL') && ~isempty(o.ylimL)
                ylim(o.ylimL);
            elseif side == 2 && isfield(o, 'ylimR') && ~isempty(o.ylimR)
                ylim(o.ylimR);
            end
        end
    end
    if use_shared
        ylabel(shared, 'FontName', o.fn, 'FontSize', o.fs_label);
        set(gca, 'YColor', ax_col);
    end

    xline(n + 0.5, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', o.lw - 1, ...
          'HandleVisibility', 'off');
    xlim([0.5, 2*n + 0.5]);
    xticks([xL, xR]);
    xticklabels([o.labels(:)', o.labels(:)']);
    xn = @(xv) (mean(xv) - 0.5) / (2*n);
    text(xn(xL), 0.97, 'Horizontal', 'Units', 'normalized', 'FontName', o.fn, ...
         'FontSize', o.fs_label, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'top', 'FontWeight', 'bold');
    text(xn(xR), 0.97, 'Vertical', 'Units', 'normalized', 'FontName', o.fn, ...
         'FontSize', o.fs_label, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'top', 'FontWeight', 'bold');
    set(gca, 'FontName', o.fn, 'FontSize', o.fs_tick);
end


%% ========================================================================
%  Helper: i_save_fig  (png + pdf + fig)
% =========================================================================
function i_save_fig(h, save_dir, name, scale)
    if nargin < 4 || isempty(scale), scale = [1.1 1.5]; end
    if isscalar(scale), scale = [scale scale]; end     % [width_scale height_scale]
    name = regexprep(name, '[^\w\-]', '_');            % filesystem-safe
    stem = fullfile(save_dir, name);
    % Size the OUTPUT via PaperPosition + print, NOT by resizing the on-screen
    % figure: an interactive figure window is clamped to the display height, so
    % growing its Position widens but cannot heighten it (squat output). print
    % renders to the paper size regardless of the window, so height scales too.
    oldU   = get(h, 'Units');  set(h, 'Units', 'pixels');
    pos_px = get(h, 'Position');  set(h, 'Units', oldU);
    w_in = pos_px(3) / 96 * scale(1);
    h_in = pos_px(4) / 96 * scale(2);
    oldPU = get(h,'PaperUnits');     oldPM = get(h,'PaperPositionMode');
    oldPP = get(h,'PaperPosition');  oldPS = get(h,'PaperSize');
    set(h, 'PaperUnits','inches', 'PaperPositionMode','manual', ...
           'PaperPosition',[0 0 w_in h_in], 'PaperSize',[w_in h_in]);
    try
        print(h, [stem '.png'], '-dpng', '-r200');
        print(h, [stem '.pdf'], '-dpdf', '-vector');
        savefig(h, [stem '.fig']);
        fprintf('  saved figure -> %s.{png,pdf,fig}  (%.1f x %.1f in)\n', stem, w_in, h_in);
    catch ME
        fprintf('  [figure save failed (%s): %s]\n', name, ME.message);
    end
    set(h, 'PaperUnits',oldPU, 'PaperPositionMode',oldPM, ...
           'PaperPosition',oldPP, 'PaperSize',oldPS);   % restore paper settings
end

%% ========================================================================
%  Helper: extract_pursuit_info
% =========================================================================
function info = extract_pursuit_info(R, t_encounter_radius)
    params = R.params;
    na   = double(params.num_agents);
    dt   = double(params.dt);
    t    = double(R.t(:));
    dist = double(R.dist);
    N    = length(t) - 1;

    if isfield(params, 'r_ttc') && double(params.r_ttc) > 0
        r_col = double(params.r_ttc);
    else
        r_col = double(params.r_cbf);
    end
    threshold = 2 * r_col;
    ego_pairs = 1:(na-1);

    col_events = zeros(N+1, 1);
    for p = ego_pairs
        d_p = dist(:, p);
        in_collision = d_p < threshold;
        edges = diff([false; in_collision]) == 1;
        col_events = col_events + double(edges);
    end
    total_collisions = sum(col_events);

    ego_dists = dist(:, ego_pairs);
    min_ego_dist = min(ego_dists, [], 2);

    encounter_threshold = 4 * r_col;
    in_encounter = min_ego_dist < encounter_threshold & min_ego_dist >= threshold;
    encounter_edges = diff([false; in_encounter]) == 1;
    encounter_timesteps_raw = find(encounter_edges);

    collision_exclusion_steps = round(t_encounter_radius / dt);
    encounter_timesteps = [];
    encounter_pursuer   = [];
    for ei = 1:length(encounter_timesteps_raw)
        k = encounter_timesteps_raw(ei);
        k_lo = max(1, k - collision_exclusion_steps);
        k_hi = min(N+1, k + collision_exclusion_steps);
        if any(col_events(k_lo:k_hi) > 0), continue; end
        [~, closest_pair] = min(ego_dists(k, :));
        encounter_timesteps(end+1) = k;              %#ok<AGROW>
        encounter_pursuer(end+1)   = closest_pair + 1; %#ok<AGROW>
    end

    cbt = params.cbf_type;
    if iscell(cbt), cbt = cbt{1}; end
    if isnumeric(cbt), cbt = char(cbt); end

    info.cbf_type    = strtrim(cbt);
    info.na          = na;
    info.dim         = double(params.dim);
    info.dt          = dt;
    info.t           = t;
    info.dist        = dist;
    info.r_col       = r_col;
    info.threshold   = threshold;
    info.encounter_threshold = encounter_threshold;
    info.total_collisions    = total_collisions;
    info.encounter_timesteps = encounter_timesteps(:);
    info.encounter_pursuer   = encounter_pursuer(:);
end


%% ========================================================================
%  Helper: get_collision_stats
% =========================================================================
function [speeds, depths, turn_rates, accels, climbs, pitch_rates] = get_collision_stats(R, info)
    [kt, kp] = get_collision_events(R, info);
    n = length(kt);
    speeds     = zeros(n, 1);
    depths     = zeros(n, 1);
    turn_rates = zeros(n, 1);   % raw (signed) yaw rate u_psi / omega
    accels     = zeros(n, 1);   % raw (signed) acceleration command
    climbs     = zeros(n, 1);   % dz/dt = v*sin(gamma); NaN in 2D (no vertical channel)
    pitch_rates = zeros(n, 1);  % u_gamma, the flight-path-angle rate; NaN in 2D
    if info.dim == 2, v_idx = 4; a_idx = 2; else, v_idx = 6; a_idx = 3; end
    g_idx = 5;                  % flight-path angle (3D only)
    has_u = isfield(R, 'u') && ~isempty(R.u);
    if has_u, nu = size(R.u, 1); end   % control history is one step shorter than state/dist
    for i = 1:n
        speeds(i) = double(R.x(kt(i), 1, v_idx));
        depths(i) = double(R.dist(kt(i), kp(i) - 1));
        if info.dim == 3
            climbs(i) = double(R.x(kt(i), 1, v_idx)) * sin(double(R.x(kt(i), 1, g_idx)));
        else
            climbs(i) = NaN;
        end
        if has_u
            ku = min(kt(i), nu);                       % clamp final-step collisions
            turn_rates(i) = double(R.u(ku, 1, 1));     % u_psi (3D) / omega (2D)
            accels(i)     = double(R.u(ku, 1, a_idx)); % acceleration command
            if info.dim == 3
                pitch_rates(i) = double(R.u(ku, 1, 2));   % u_gamma
            else
                pitch_rates(i) = NaN;                     % 2D u = [omega; a]
            end
        else
            turn_rates(i) = NaN;
            accels(i)     = NaN;
            pitch_rates(i) = NaN;
        end
    end
end


%% ========================================================================
%  Helper: get_collision_events
% =========================================================================
function [kt, kp] = get_collision_events(R, info)
    na = info.na;
    dist = double(R.dist);
    N = size(dist, 1);
    threshold = info.threshold;
    col_events = zeros(N, 1);
    col_pair   = zeros(N, 1);
    for p = 1:(na-1)
        dp = dist(:, p);
        ic = dp < threshold;
        ed = diff([false; ic]) == 1;
        col_events = col_events + double(ed);
        col_pair(ed) = p + 1;
    end
    kt = find(col_events > 0);
    kp = col_pair(kt);
end


%% ========================================================================
%  Helper: count_per_pursuer
% =========================================================================
function [close_counts, col_counts] = count_per_pursuer(R, info)
    n_pursuers = info.na - 1;
    close_counts = zeros(n_pursuers, 1);
    col_counts   = zeros(n_pursuers, 1);
    for p = 1:n_pursuers
        dp = double(R.dist(:, p));
        close_counts(p) = sum(diff([false; dp < info.encounter_threshold]) == 1);
        col_counts(p)   = sum(diff([false; dp < info.threshold]) == 1);
    end
end
