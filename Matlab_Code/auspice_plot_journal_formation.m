function auspice_plot_journal_formation(varargin)
%AUSPICE_PLOT_JOURNAL_FORMATION  Journal-paper-style figures for the
%   formation_pursuit sweep.  At creation time this is an exact copy of
%   auspice_plot_journal; it exists as a sibling so formation-specific
%   figures (e.g. MEB-radius dynamics, formation-spread metrics) can be
%   added here without touching the random_sphere version.
%
%   Convention matches the Journal_Paper plotting scripts:
%       - color  = CBF type (grey/dark-red/blue for none/shocbf/aTTC)
%       - marker = scenario (auto-cycled if not specified)
%       - Times New Roman, doubled font sizes
%       - PNG (300 dpi) + PDF (vector) + FIG (interactive) per figure
%
%   Inputs (positional, in order):
%     r1, r2, ..., rN
%       Each `ri` may be either
%         (a) a raw auspice_sim result struct (has .x, .params, .dist), OR
%         (b) a loaded .mat struct (has .res and optional .metrics).
%       Mixed types are allowed.
%
%   Optional name-value pairs (after the result structs):
%     'save_dir'   directory for output (default: pwd)
%     'labels'     {1xN cell}  override the auto-derived xtick / legend
%                              labels (e.g. {'HOCBF, slow', 'aTTC, slow'}).
%                              Two-line labels: separate with '\newline'.
%     'colors'     Nx3 matrix  override the auto color-by-cbf_type
%     'markers'    {1xN cell}  override marker symbols (default cycles)
%     'meb_xcap'   km          fixed x-axis cap for the MEB-radius PDF
%                              (default: max(5*sphere_R, 1.05*q99))
%     'meb_nbins'  scalar or   bins in MEB PDF (default 100).  If a vector
%                  vector      is supplied, one PDF figure is generated
%                              per value using the SAME (expensive) MEB
%                              radius series — useful for tuning bin count
%                              without recomputing the MEB series.  Files
%                              are suffixed `_nb<N>` when >1 value given.
%     'figsize'    [W H]       figure pixel size (default [1200 750])
%     'name_tag'   string      filename prefix for outputs (default '')
%     'figure_list' string     which figure group(s) to render (default 'meb').
%                              Valid: 'full', 'meb', 'scatter', 'lookahead'.
%                              Groups:
%                                'summary'  = collision_rate + waypoint_rate +
%                                             collisions_per_waypoint + dist_origin_median
%                                'meb'      = meb_radius_pdf + meb_radius_cdf
%                                'lookahead' = nm_collision_fraction_vs_lookahead +
%                                              nm_avg_{speed,angrate,accel}_vs_lookahead
%                                'scatter'   = nm_scatter_{speed,angrate,accel}_{slow,fast}_pur
%                              'full' renders all four groups; the other options
%                              render only the named group.
%     'save_figures' bool      save each rendered figure to .{png,pdf,fig}
%                              (default false).  When false figures are only
%                              shown on screen.
%
%   Figures produced:
%     fig_collision_rate.{png,pdf,fig}           stacked ee + ep, 1 bar/result
%     fig_waypoint_rate.{png,pdf,fig}            1 bar/result, colored by cbf
%     fig_collisions_per_waypoint.{png,pdf,fig}  stacked ee/wp + ep/wp,
%                                                 cost-per-progress metric
%     fig_dist_origin_median.{png,pdf,fig}       1 bar/result, colored by cbf
%     fig_meb_radius_pdf.{png,pdf,fig}           PDF zoomed to [0, meb_xcap]
%     fig_meb_radius_cdf.{png,pdf,fig}           CDF on log x, full range
%     fig_nm_collision_fraction_vs_lookahead.{png,pdf,fig}
%                              P(collision|near-miss) vs T_lookahead per result
%     fig_nm_avg_speed_vs_lookahead.{png,pdf,fig}
%     fig_nm_avg_angrate_vs_lookahead.{png,pdf,fig}
%     fig_nm_avg_accel_vs_lookahead.{png,pdf,fig}
%                              avoided- vs collided-group window-mean of
%                              the evader's speed / sqrt(yaw^2+pitch^2) /
%                              |accel| over [t_nm, t_nm + T_lookahead].
%                              Pursuit cells only (pair_kind = 'ep').
%     fig_nm_scatter_<var>_<slow|fast>_pur.{png,pdf,fig}
%                              Jittered scatter at fixed T_lookahead =
%                              nm_scatter_T_lookahead (default 1 s).
%                              var in {speed, angrate, accel}, scenario in
%                              {slow, fast} = 6 figures total.  Two columns
%                              per figure (sHOCBF, aTTC).  Points coloured
%                              red (collided) / blue (avoided); mean bars
%                              per outcome overlaid.
%
%   Example
%     S1 = load('random_sphere_pursuit_adv0p9__hocbf.mat');
%     S2 = load('random_sphere_pursuit_adv0p9__attc_vraw__iw_huber3__data1.mat');
%     S3 = load('random_sphere_pursuit_adv1p5__hocbf.mat');
%     S4 = load('random_sphere_pursuit_adv1p5__attc_vraw__iw_huber3__data1.mat');
%     auspice_plot_journal(S1, S2, S3, S4, ...
%         'labels', {'HOCBF\newlineslow','aTTC\newlineslow', ...
%                    'HOCBF\newlinefast','aTTC\newlinefast'}, ...
%         'save_dir', '../Results/Journal_Paper/custom');

    % --- Parse: structs first, then name/value pairs ---
    n_results = 0;
    for k = 1:numel(varargin)
        if isstruct(varargin{k}), n_results = n_results + 1; else, break; end
    end
    all_results = varargin(1:n_results);
    remaining   = varargin(n_results+1:end);

    ip = inputParser;
    addParameter(ip, 'save_dir',  pwd,        @(s) ischar(s) || isstring(s));
    addParameter(ip, 'labels',    {},         @iscell);
    addParameter(ip, 'colors',    [],         @isnumeric);
    addParameter(ip, 'markers',   {},         @iscell);
    addParameter(ip, 'meb_xcap',  100,        @isnumeric);   % km; hard cap, no auto-99th-pct
    addParameter(ip, 'meb_nbins', 1000,       @isnumeric);   % bins on [0, meb_xcap]
    addParameter(ip, 'cdf_xmin',  0.1,        @isnumeric);   % km; lower bound for the log-x CDF figure
    addParameter(ip, 'cdf_xmax',  NaN,        @isnumeric);   % km; if NaN, auto from data max
    addParameter(ip, 'nm_d_near',       0.4,       @isnumeric);   % km
    addParameter(ip, 'nm_d_collision',  0.2,       @isnumeric);   % km
    addParameter(ip, 'nm_T_lookaheads', 0.1:0.2:3, @isnumeric);   % s
    addParameter(ip, 'nm_scatter_T_lookahead', 1.0, @isnumeric);  % s, fixed T for scatter
    addParameter(ip, 'nm_scatter_max_pts',     2000, @isnumeric); % per (cell, outcome) cap
    addParameter(ip, 'figsize',   [1200 750], @isnumeric);
    addParameter(ip, 'name_tag',  '',         @(s) ischar(s) || isstring(s));
    addParameter(ip, 'figure_list', 'meb',    @(s) ischar(s) || isstring(s));
    addParameter(ip, 'save_figures', false,   @(x) islogical(x) || isnumeric(x));
    parse(ip, remaining{:});

    save_dir       = char(ip.Results.save_dir);
    custom_labels  = ip.Results.labels;
    custom_colors  = ip.Results.colors;
    custom_markers = ip.Results.markers;
    meb_xcap       = ip.Results.meb_xcap;
    meb_nbins      = double(ip.Results.meb_nbins(:))';   % row vector: 1+ nbins values
    cdf_xmin       = ip.Results.cdf_xmin;
    cdf_xmax_opt   = ip.Results.cdf_xmax;
    nm_d_near       = double(ip.Results.nm_d_near);
    nm_d_collision  = double(ip.Results.nm_d_collision);
    nm_T_lookaheads = double(ip.Results.nm_T_lookaheads(:)');
    nm_scatter_T    = double(ip.Results.nm_scatter_T_lookahead);
    nm_scatter_max  = double(ip.Results.nm_scatter_max_pts);
    figsize        = ip.Results.figsize;
    name_tag       = char(ip.Results.name_tag);
    figure_list    = char(lower(string(ip.Results.figure_list)));
    do_save        = logical(ip.Results.save_figures);
    valid_lists = {'full', 'meb', 'scatter', 'lookahead'};
    if ~any(strcmp(figure_list, valid_lists))
        error('auspice_plot_journal_formation:badFigureList', ...
            'figure_list must be one of %s (got %s)', ...
            strjoin(valid_lists, ', '), figure_list);
    end
    fprintf('  figure_list=%s  save_figures=%d\n', figure_list, do_save);

    assert(n_results >= 1, 'auspice_plot_journal_formation:noResults', ...
        'At least one result struct is required.');
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    if ~isempty(name_tag) && ~endsWith(name_tag, '_')
        name_tag = [name_tag '_'];
    end

    % --- Plot conventions (match Journal_Paper) ---
    fn        = 'Times New Roman';
    fs_label  = 26;
    fs_tick   = 22;
    fs_legend = 22;
    lw        = 2.5;
    light_mix = 0.55;          % how much white is mixed into the ee stack

    % --- Build cell info: normalize + derive label/color/marker per result ---
    cells = build_cell_info(all_results, custom_labels, custom_colors, ...
                            custom_markers);
    n_cells = numel(cells);

    fprintf('\n=== auspice_plot_journal_formation ===\n');
    fprintf('  n_results : %d\n', n_cells);
    fprintf('  save_dir  : %s\n', save_dir);
    for k = 1:n_cells
        fprintf('   [%d] cbf=%-12s  label="%s"\n', k, ...
            cells{k}.cbf_type, strrep(cells{k}.label, '\newline', ' / '));
    end
    fprintf('\n');

    % --- Compute per-cell summary values ---
    rate_ee = nan(n_cells, 1);
    rate_ep = nan(n_cells, 1);
    wp      = nan(n_cells, 1);
    r_med   = nan(n_cells, 1);
    meb_series   = cell(n_cells, 1);
    nm_delta_t   = cell(n_cells, 1);
    nm_speed_win = cell(n_cells, 1);
    nm_yaw_win   = cell(n_cells, 1);
    nm_pitch_win = cell(n_cells, 1);
    nm_accel_win = cell(n_cells, 1);
    nm_dt        = nan(n_cells, 1);
    nm_pair_kind = repmat({''}, n_cells, 1);
    nm_n_events  = zeros(n_cells, 1);
    sphere_R_first = NaN;
    % Formation params (assumed identical across cells; captured from first ok cell)
    form_spacing_first = NaN;
    form_alpha_first   = NaN;
    n_ev_first         = NaN;

    for k = 1:n_cells
        res     = cells{k}.res;
        metrics = cells{k}.metrics;
        p       = res.params;
        T_sim   = double(p.T_sim);
        dim     = double(getfield_or(p, 'dim', 3));
        if isfield(p, 'evader_mask') && ~isempty(p.evader_mask)
            em = logical(p.evader_mask(:));
        else
            em = true(double(p.num_agents), 1);
        end
        if isnan(sphere_R_first)
            sphere_R_first = double(getfield_or(p, 'sphere_radius', 6.25));
        end
        if isnan(form_spacing_first)
            form_spacing_first = double(getfield_or(p, 'formation_spacing', NaN));
            form_alpha_first   = double(getfield_or(p, 'formation_stagger_angle', NaN));
            n_ev_first         = sum(em);
        end

        % Collisions per 100s — use metrics if available
        if isstruct(metrics) && isfield(metrics, 'col_ee_count')
            rate_ee(k) = double(metrics.col_ee_count) * 100 / T_sim;
        else
            rate_ee(k) = compute_collision_rate(res, em, 'ee') * 100 / T_sim;
        end
        if isstruct(metrics) && isfield(metrics, 'col_ep_count')
            rate_ep(k) = double(metrics.col_ep_count) * 100 / T_sim;
        else
            rate_ep(k) = compute_collision_rate(res, em, 'ep') * 100 / T_sim;
        end

        % Waypoint progress rate (per evader per 100 s)
        if isstruct(metrics) && isfield(metrics, 'wp_progress_rate_per100s')
            wp(k) = double(metrics.wp_progress_rate_per100s);
        else
            % Fallback via the canonical extractor if available
            try
                m_fb  = sweep_cbf_extract_metrics(res);
                wp(k) = double(getfield_or(m_fb, 'wp_progress_rate_per100s', NaN));
            catch
                wp(k) = NaN;
            end
        end

        % Median evader distance from origin
        x = double(res.x);
        ev_pos = x(:, em, 1:3);
        r_orig = sqrt(sum(ev_pos.^2, 3));
        r_med(k) = median(r_orig(:));

        % MEB radius time series (evaders only)
        if sum(em) >= 2
            meb_series{k} = compute_meb_series_local(x, em, dim);
        else
            meb_series{k} = [];
        end

        % Near-miss events: rising edges of d < d_near, with delta_t to
        % first d < d_collision afterwards.  Also extract per-event evader
        % speed / |yaw| / |pitch| / |accel| windows for the outcome-
        % conditioned lookahead averages.
        ed = nearmiss_window_extract(res, struct( ...
            'd_near',      nm_d_near, ...
            'd_collision', nm_d_collision, ...
            'T_max',       max([nm_T_lookaheads, nm_scatter_T]), ...
            'return_metric_windows', true));
        nm_delta_t{k}   = ed.delta_t_collision;
        nm_n_events(k)  = ed.n_events;
        nm_dt(k)        = ed.dt;
        nm_pair_kind{k} = ed.pair_kind;
        nm_speed_win{k} = ed.speed_window;
        nm_yaw_win{k}   = ed.yaw_window;
        nm_pitch_win{k} = ed.pitch_window;
        nm_accel_win{k} = ed.accel_window;
    end

    % =========================================================
    %  Figures 1 / 2 / 2b / 3: summary bar charts (group='summary')
    % =========================================================
    if want_figure(figure_list, 'summary')
    % =========================================================
    %  Figure 1: Collision rate (stacked ee + ep)
    % =========================================================
    xpos = 1:n_cells;
    xtl  = cellfun(@(c) c.label, cells, 'UniformOutput', false);
    cdata_ep = zeros(n_cells, 3);
    cdata_ee = zeros(n_cells, 3);
    cdata    = zeros(n_cells, 3);
    for k = 1:n_cells
        base = cells{k}.color;
        cdata(k, :)    = base;
        cdata_ep(k, :) = base;
        cdata_ee(k, :) = base + light_mix * (1 - base);
    end

    fig1 = figure('Name','collision rate','Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig1, fn);
    hold on; grid on; box on;
    bs = bar(xpos, [rate_ee, rate_ep], 'stacked', 'EdgeColor','none');
    bs(1).FaceColor = 'flat';  bs(1).CData = cdata_ee;
    bs(2).FaceColor = 'flat';  bs(2).CData = cdata_ep;
    set(gca, 'XTick', xpos, 'XTickLabel', xtl, ...
        'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    ylabel('Collisions per 100 s', 'FontName', fn, 'FontSize', fs_label);
    xlim([0.4, n_cells + 0.6]);
    % Legend with neutral-grey swatches
    ref_dark  = [0.40 0.40 0.40];
    ref_light = ref_dark + light_mix * (1 - ref_dark);
    h_ep_p = bar(nan, nan, 'FaceColor', ref_dark,  'EdgeColor','none');
    h_ee_p = bar(nan, nan, 'FaceColor', ref_light, 'EdgeColor','none');
    legend([h_ep_p h_ee_p], {'evader-pursuer','evader-evader'}, ...
        'FontName', fn, 'FontSize', fs_legend, 'Location','best');
    for k = 1:n_cells
        tot = rate_ee(k) + rate_ep(k);
        if isnan(tot), continue; end
        text(xpos(k), tot, sprintf(' %.3g', tot), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName', fn, 'FontSize', fs_tick);
    end
    save_three_maybe(fig1, fullfile(save_dir, [name_tag 'fig_collision_rate']), do_save);

    % =========================================================
    %  Figure 2: Waypoint progress rate
    % =========================================================
    fig2 = figure('Name','waypoint rate','Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig2, fn);
    hold on; grid on; box on;
    b2 = bar(xpos, wp, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b2.CData = cdata;
    set(gca, 'XTick', xpos, 'XTickLabel', xtl, ...
        'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    ylabel('Waypoints per evader per 100 s', 'FontName', fn, 'FontSize', fs_label);
    xlim([0.4, n_cells + 0.6]);
    for k = 1:n_cells
        if isnan(wp(k)), continue; end
        text(xpos(k), wp(k), sprintf(' %.3g', wp(k)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName', fn, 'FontSize', fs_tick);
    end
    save_three_maybe(fig2, fullfile(save_dir, [name_tag 'fig_waypoint_rate']), do_save);

    % =========================================================
    %  Figure 2b: Collisions per waypoint
    %  Combines the collision rate and waypoint progress rate into a
    %  single "cost-per-progress" metric:
    %      ratio = (ee + ep collisions per 100 s) / (waypoints per evader per 100 s)
    %  Stacked: ee/wp on bottom (lighter shade) + ep/wp on top (method
    %  colour), matching the convention from fig_collision_rate.
    %  Higher bar = more collisions per unit of formation progress.
    % =========================================================
    cols_ee_per_wp = nan(n_cells, 1);
    cols_ep_per_wp = nan(n_cells, 1);
    for k = 1:n_cells
        if isnan(wp(k)) || wp(k) <= 0, continue; end
        cols_ee_per_wp(k) = rate_ee(k) / wp(k);
        cols_ep_per_wp(k) = rate_ep(k) / wp(k);
    end

    fig2b = figure('Name','collisions per waypoint','Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig2b, fn);
    hold on; grid on; box on;
    bs2b = bar(xpos, [cols_ee_per_wp, cols_ep_per_wp], 'stacked', 'EdgeColor','none');
    bs2b(1).FaceColor = 'flat';  bs2b(1).CData = cdata_ee;     % ee = lighter
    bs2b(2).FaceColor = 'flat';  bs2b(2).CData = cdata_ep;     % ep = full
    set(gca, 'XTick', xpos, 'XTickLabel', xtl, ...
        'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    ylabel('Collisions per waypoint', 'FontName', fn, 'FontSize', fs_label);
    xlim([0.4, n_cells + 0.6]);
    ref_dark  = [0.40 0.40 0.40];
    ref_light = ref_dark + light_mix * (1 - ref_dark);
    h_ep_p2b = bar(nan, nan, 'FaceColor', ref_dark,  'EdgeColor','none');
    h_ee_p2b = bar(nan, nan, 'FaceColor', ref_light, 'EdgeColor','none');
    legend([h_ep_p2b h_ee_p2b], {'evader-pursuer','evader-evader'}, ...
        'FontName', fn, 'FontSize', fs_legend, 'Location','best');
    for k = 1:n_cells
        tot = cols_ee_per_wp(k) + cols_ep_per_wp(k);
        if isnan(tot), continue; end
        text(xpos(k), tot, sprintf(' %.3g', tot), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName', fn, 'FontSize', fs_tick);
    end
    save_three_maybe(fig2b, fullfile(save_dir, [name_tag 'fig_collisions_per_waypoint']), do_save);

    % =========================================================
    %  Figure 3: Median evader distance from origin
    % =========================================================
    fig3 = figure('Name','median dist origin','Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig3, fn);
    hold on; grid on; box on;
    b3 = bar(xpos, r_med, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b3.CData = cdata;
    set(gca, 'XTick', xpos, 'XTickLabel', xtl, ...
        'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    ylabel('Median evader distance from origin [km]', ...
        'FontName', fn, 'FontSize', fs_label);
    xlim([0.4, n_cells + 0.6]);
    for k = 1:n_cells
        if isnan(r_med(k)), continue; end
        text(xpos(k), r_med(k), sprintf(' %.3g', r_med(k)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName', fn, 'FontSize', fs_tick);
    end
    save_three_maybe(fig3, fullfile(save_dir, [name_tag 'fig_dist_origin_median']), do_save);
    end  % want_figure summary

    % =========================================================
    %  Figures 4 / 5: MEB radius PDF + CDF (group='meb')
    % =========================================================
    if want_figure(figure_list, 'meb')
    % =========================================================
    %  Figure 4: PDF of evader MEB (minimum-enclosing-ball) radius
    % =========================================================
    % Hard cap (default 100 km) — extreme outliers from runaway pursuit
    % events otherwise dominate the bin layout.  Override with
    % opts.meb_xcap.  Off-chart fraction is annotated in the legend.
    %
    % `meb_nbins` may be a vector; we generate one figure per value using
    % the MEB series computed once above.  Filename suffix `_nb<N>` is
    % appended when >1 value.
    meb_xcap_resolved = double(meb_xcap);
    % Best-case (nominal-formation) MEB reference — computed once, drawn per fig
    [best_meb, best_label] = compute_best_meb(form_spacing_first, ...
        form_alpha_first, n_ev_first);
    n_nb = numel(meb_nbins);
    fig4 = gobjects(n_nb, 1);
    for nbi = 1:n_nb
        nb = meb_nbins(nbi);
        meb_edges   = linspace(0, meb_xcap_resolved, nb + 1);
        meb_centers = 0.5 * (meb_edges(1:end-1) + meb_edges(2:end));
        marker_stride = max(1, floor(nb / 50));

        fig4(nbi) = figure('Name', sprintf('meb radius pdf (nb=%d)', nb), ...
            'Position',[60 60 figsize], 'Color','w');
        fig_defaults(fig4(nbi), fn);
        hold on; grid on; box on;
        h_lines  = gobjects(n_cells, 1);
        leg_meb  = cell(n_cells, 1);
        for k = 1:n_cells
            c = cells{k};
            leg_meb{k} = c.label;
            if isempty(meb_series{k})
                h_lines(k) = plot(nan, nan, '-', 'Color', c.color, 'LineWidth', lw, ...
                    'Marker', c.marker, 'MarkerSize', 12);
                leg_meb{k} = [leg_meb{k} ' (no data)'];
                continue;
            end
            mr = meb_series{k};
            off_frac = mean(mr > meb_xcap_resolved);
            cnt = histcounts(mr, meb_edges, 'Normalization', 'pdf');
            h_lines(k) = plot(meb_centers, cnt, '-', ...
                'Color', c.color, 'LineWidth', lw, ...
                'Marker', c.marker, 'MarkerSize', 12, ...
                'MarkerFaceColor', c.color, 'MarkerEdgeColor', 'k', ...
                'MarkerIndices', 1:marker_stride:numel(meb_centers));
            % Only annotate right-tail truncation when it's actually happening.
            if off_frac > 0.005
                leg_meb{k} = sprintf('%s  (%.1f%% > %g km)', ...
                    leg_meb{k}, 100*off_frac, meb_xcap_resolved);
            end
        end

        xlim([0 30]);
        if isfinite(best_meb)
            xline(best_meb, '--', best_label, ...
                'Color', [0.15 0.55 0.15], 'LineWidth', 2.0, ...
                'FontName', fn, 'FontSize', fs_tick, ...
                'LabelHorizontalAlignment', 'left', ...
                'LabelVerticalAlignment',   'middle');
        end
        xlabel('Evader MEB radius [km]', 'FontName', fn, 'FontSize', fs_label);
        ylabel('PDF', 'FontName', fn, 'FontSize', fs_label);
        set(gca, 'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
        title(sprintf('n_{bins} = %d', nb), 'FontName', fn, ...
            'FontSize', fs_label, 'FontWeight', 'normal');
        legend(h_lines, leg_meb, 'FontName', fn, 'FontSize', fs_legend, ...
            'Location', 'best', 'Box', 'on');

        if n_nb == 1
            fname_pdf = 'fig_meb_radius_pdf';
        else
            fname_pdf = sprintf('fig_meb_radius_pdf_nb%d', nb);
        end
        save_three_maybe(fig4(nbi), fullfile(save_dir, [name_tag fname_pdf]), do_save);
    end

    % =========================================================
    %  Figure 5: CDF of MEB radius on log x-axis (full data range)
    %  Shows the heavy tail: e.g. a cell where 91% of MEB samples lie
    %  past 100 km appears as a curve that only reaches y≈0.09 at the
    %  PDF's x-cap and slowly climbs to 1 across the rest of the log-x
    %  axis.  Companion to fig_meb_radius_pdf — the PDF shows the bulk,
    %  the CDF shows the tail.
    % =========================================================
    if isnan(cdf_xmax_opt) || cdf_xmax_opt <= 0
        cdf_xmax = 0;
        for k = 1:n_cells
            if ~isempty(meb_series{k})
                cdf_xmax = max(cdf_xmax, max(meb_series{k}));
            end
        end
        cdf_xmax = max(cdf_xmax * 1.05, meb_xcap_resolved);
    else
        cdf_xmax = cdf_xmax_opt;
    end
    cdf_xmax = 30;
    fig5 = figure('Name','meb radius cdf','Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig5, fn);
    hold on; grid on; box on;
    h_lines_cdf = gobjects(n_cells, 1);
    leg_cdf     = cell(n_cells, 1);
    n_plot_pts  = 5000;          % subsample for rendering speed
    for k = 1:n_cells
        c = cells{k};
        leg_cdf{k} = c.label;
        if isempty(meb_series{k})
            h_lines_cdf(k) = plot(nan, nan, '-', 'Color', c.color, 'LineWidth', lw, ...
                'Marker', c.marker, 'MarkerSize', 12);
            leg_cdf{k} = [leg_cdf{k} ' (no data)'];
            continue;
        end
        sorted = sort(meb_series{k}(~isnan(meb_series{k})));
        N = numel(sorted);
        F = (1:N)' / N;
        if N > n_plot_pts
            idx = round(linspace(1, N, n_plot_pts));
            s_plot = sorted(idx);
            F_plot = F(idx);
        else
            s_plot = sorted;
            F_plot = F;
        end
        % Clip plotted x to >= cdf_xmin so the line doesn't go off-axis
        s_plot = max(s_plot, cdf_xmin);
        h_lines_cdf(k) = plot(s_plot, F_plot, '-', ...
            'Color', c.color, 'LineWidth', lw, ...
            'Marker', c.marker, 'MarkerSize', 12, ...
            'MarkerFaceColor', c.color, 'MarkerEdgeColor', 'k', ...
            'MarkerIndices', round(linspace(1, numel(s_plot), 14)));
        % Add a quick "fraction > meb_xcap" annotation so this figure
        % reads as the natural companion to the PDF.
        off_frac = mean(meb_series{k} > meb_xcap_resolved);
        leg_cdf{k} = sprintf('%s  (%.2f%% > %g km)', ...
            leg_cdf{k}, 100*off_frac, meb_xcap_resolved);
    end
    set(gca, 'XScale', 'linear');
    xlim([cdf_xmin, cdf_xmax]);
    ylim([0, 1.02]);
    % Reference: vertical line at the PDF's cap so the eye can see
    % where the PDF zoom ends.
    xline(meb_xcap_resolved, ':', '', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);
    % Best-case nominal-formation MEB reference line (same as PDF figure)
    if isfinite(best_meb)
        xline(best_meb, '--', best_label, ...
            'Color', [0.15 0.55 0.15], 'LineWidth', 2.0, ...
            'FontName', fn, 'FontSize', fs_tick, ...
            'LabelHorizontalAlignment', 'right', ...
            'LabelVerticalAlignment',   'middle');
    end
    xlabel('Evader MEB radius [km]', 'FontName', fn, 'FontSize', fs_label);
    ylabel('CDF', 'FontName', fn, 'FontSize', fs_label);
    set(gca, 'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    legend(h_lines_cdf, leg_cdf, 'FontName', fn, 'FontSize', fs_legend, ...
        'Location', 'best', 'Box', 'on');
    save_three_maybe(fig5, fullfile(save_dir, [name_tag 'fig_meb_radius_cdf']), do_save);
    end  % want_figure meb

    % =========================================================
    %  Figures 6 / 7 / 8 / 9 / 10: outcome-conditioned lookahead (group='lookahead')
    % =========================================================
    if want_figure(figure_list, 'lookahead')
    % =========================================================
    %  Figure 6: P(collision | near-miss) vs T_lookahead
    %  Near-miss = rising edge of pair distance < nm_d_near (default 0.4 km).
    %  Outcome = collision iff pair distance < nm_d_collision (default 0.2 km)
    %  within the next T seconds.  X-axis sweeps T over nm_T_lookaheads.
    % =========================================================
    p_collision = nan(numel(nm_T_lookaheads), n_cells);
    for k = 1:n_cells
        if nm_n_events(k) == 0, continue; end
        for ti = 1:numel(nm_T_lookaheads)
            p_collision(ti, k) = mean(nm_delta_t{k} <= nm_T_lookaheads(ti));
        end
    end

    fig6 = figure('Name','P(collision|near-miss) vs lookahead', ...
                  'Position',[60 60 figsize], 'Color','w');
    fig_defaults(fig6, fn);
    hold on; grid on; box on;
    lw_nm   = 2 * lw;
    ms_nm   = 24;        % 2x the previous 12
    h_lines_nm = gobjects(0, 1);
    leg_nm     = {};
    for k = 1:n_cells
        c = cells{k};
        % Pursuit cells only — skip "0 pursuer" baseline rows
        if ~strcmp(nm_pair_kind{k}, 'ep'), continue; end
        if nm_n_events(k) == 0 || all(isnan(p_collision(:, k)))
            continue;
        end
        h = plot(nm_T_lookaheads, p_collision(:, k), '-', ...
            'Color', c.color, 'LineWidth', lw_nm, ...
            'Marker', c.marker, 'MarkerSize', ms_nm, ...
            'MarkerFaceColor', c.color, 'MarkerEdgeColor', 'k');
        h_lines_nm(end+1) = h;          %#ok<AGROW>
        leg_nm{end+1}     = c.label;    %#ok<AGROW>
    end
    xlabel('Lookahead horizon T [s]', 'FontName', fn, 'FontSize', fs_label);
    ylabel('P(collision \mid near-miss)', 'FontName', fn, 'FontSize', fs_label);
    set(gca, 'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    xlim([min(nm_T_lookaheads) - 0.05, max(nm_T_lookaheads) + 0.05]);
    ylim([0, max(1.05 * max([p_collision(:); 0.01]), 0.05)]);
    if ~isempty(h_lines_nm)
        legend(h_lines_nm, leg_nm, 'FontName', fn, 'FontSize', fs_legend, ...
            'Location', 'best', 'Box', 'on');
    end
    save_three_maybe(fig6, fullfile(save_dir, [name_tag 'fig_nm_collision_fraction_vs_lookahead']), do_save);

    % =========================================================
    %  Figures 7-10: outcome-conditioned averages vs T_lookahead
    %  Same logic as the random_sphere plotter: avoided vs collided
    %  near-misses, window mean over [t_nm, t_nm + T_lookahead]
    %  (truncated at t_collision for collided events).  Pursuit cells only.
    % =========================================================
    nm_metric_specs = { ...
        struct('fname','fig_nm_avg_speed_vs_lookahead',   'data','speed',   'ylab','Avg evader speed [km/s]'); ...
        struct('fname','fig_nm_avg_angrate_vs_lookahead', 'data','angrate', 'ylab','Avg angular rate \surd(\omega^2 + \nu^2) [rad/s]'); ...
        struct('fname','fig_nm_avg_accel_vs_lookahead',   'data','accel',   'ylab','Avg \|accel\| [km/s^2]') };
    for sp_i = 1:numel(nm_metric_specs)
        spec = nm_metric_specs{sp_i};
        plot_nm_lookahead_avg(cells, nm_pair_kind, nm_n_events, nm_dt, ...
            nm_delta_t, nm_speed_win, nm_yaw_win, nm_pitch_win, nm_accel_win, ...
            nm_T_lookaheads, spec, save_dir, name_tag, ...
            fn, fs_label, fs_tick, fs_legend, lw, figsize, do_save);
    end
    end  % want_figure lookahead

    % =========================================================
    %  Figures 11+: outcome-colored jittered scatter (group='scatter')
    %  Same logic as the random_sphere plotter — see plot_nm_outcome_scatter.
    % =========================================================
    if want_figure(figure_list, 'scatter')
    nm_scatter_specs = { ...
        struct('var','speed',   'label','Avg evader speed [km/s]',                              'fname','fig_nm_scatter_speed'); ...
        struct('var','angrate', 'label','Avg angular rate \surd(\omega^2 + \nu^2) [rad/s]',     'fname','fig_nm_scatter_angrate'); ...
        struct('var','accel',   'label','Avg |accel| [km/s^2]',                                 'fname','fig_nm_scatter_accel') };

    sc_slow = []; sc_fast = [];
    for k = 1:n_cells
        if ~strcmp(nm_pair_kind{k}, 'ep'), continue; end
        pa = double(getfield_or(cells{k}.res.params, 'pursuit_advantage', NaN));
        if isnan(pa), continue; end
        if pa < 1.0,  sc_slow(end+1) = k; end %#ok<AGROW>
        if pa >= 1.0, sc_fast(end+1) = k; end %#ok<AGROW>
    end
    scenarios = {struct('idx', sc_slow, 'tag', 'slow_pur', 'pretty', 'slow pursuers'); ...
                 struct('idx', sc_fast, 'tag', 'fast_pur', 'pretty', 'fast pursuers')};

    for sp_i = 1:numel(nm_scatter_specs)
        spec = nm_scatter_specs{sp_i};
        for sc_i = 1:numel(scenarios)
            sc = scenarios{sc_i};
            if isempty(sc.idx), continue; end
            plot_nm_outcome_scatter(cells, sc.idx, nm_dt, nm_delta_t, ...
                nm_speed_win, nm_yaw_win, nm_pitch_win, nm_accel_win, ...
                nm_scatter_T, nm_scatter_max, spec, sc, ...
                save_dir, name_tag, fn, fs_label, fs_tick, fs_legend, ...
                lw, figsize, do_save);
        end
    end
    end  % want_figure scatter

    % =========================================================
    %  Console summary table (for paper writeup)
    % =========================================================
    meb_med = nan(n_cells, 1);
    for k = 1:n_cells
        if ~isempty(meb_series{k})
            meb_med(k) = median(meb_series{k}, 'omitnan');
        end
    end
    fprintf('\n');
    fprintf('=== summary stats ===\n');
    fprintf('  %-30s  %10s  %10s  %10s  %10s  %12s  %10s\n', ...
        'cell', 'col_ee/100s', 'col_ep/100s', 'col_tot', 'wp/ev/100s', ...
        'med MEB [km]', 'col/wp');
    fprintf('  %s\n', repmat('-', 1, 30 + 2 + 6*12));
    for k = 1:n_cells
        c = cells{k};
        nice_label = strrep(c.label, '\newline', ' / ');
        tot = rate_ee(k) + rate_ep(k);
        if isnan(wp(k)) || wp(k) <= 0
            cols_per_wp = NaN;
        else
            cols_per_wp = tot / wp(k);
        end
        fprintf('  %-30s  %10.4g  %10.4g  %10.4g  %10.4g  %12.3f  %10.4g\n', ...
            nice_label, rate_ee(k), rate_ep(k), tot, wp(k), ...
            meb_med(k), cols_per_wp);
    end
    fprintf('\n');

    fprintf('  done — figures saved under %s\n\n', save_dir);
end


% =====================================================================
%  Cell-info builder
% =====================================================================

function cells = build_cell_info(all_results, custom_labels, custom_colors, custom_markers)
    n = numel(all_results);
    cells = cell(n, 1);
    auto_markers = {'s', '^', 'o', 'd', 'v', 'p', '>', '<'};
    for k = 1:n
        R = all_results{k};
        % Normalize: figure out which fields are present
        if isfield(R, 'res') && isstruct(R.res)
            res     = R.res;
            metrics = struct();
            if isfield(R, 'metrics') && isstruct(R.metrics), metrics = R.metrics; end
        elseif isfield(R, 'x') && isfield(R, 'params')
            res     = R;
            metrics = struct();
        else
            error('auspice_plot_journal_formation:badResult', ...
                'Result #%d has neither .res nor .x/.params fields.', k);
        end

        cbf_type = lower(strtrim(char(getfield_or(res.params, 'cbf_type', 'unknown'))));

        % Color
        if ~isempty(custom_colors) && size(custom_colors, 1) >= k
            col = custom_colors(k, :);
        else
            col = default_color_for_cbf(cbf_type);
        end

        % Marker
        if ~isempty(custom_markers) && k <= numel(custom_markers)
            marker = custom_markers{k};
        else
            marker = auto_markers{mod(k-1, numel(auto_markers)) + 1};
        end

        % Label (two-line by default unless user provided one)
        if ~isempty(custom_labels) && k <= numel(custom_labels)
            label = custom_labels{k};
        else
            label = default_label_for_cell(res.params, cbf_type);
        end

        cells{k} = struct( ...
            'res',     res, ...
            'metrics', metrics, ...
            'color',   col, ...
            'marker',  marker, ...
            'label',   label, ...
            'cbf_type', cbf_type);
    end
end


function col = default_color_for_cbf(cbf_type)
    switch cbf_type
        case 'none',    col = [0.50 0.50 0.50];                            % grey
        case 'shocbf',  col = [0.64 0.08 0.18];                            % dark red
        case {'attc_vraw','sattc_vraw','attc_jax','sattc_jax', ...
              'attc','sattc'}, col = [0.00 0.45 0.74];                     % blue
        case 'hocbf',   col = [0.85 0.33 0.10];                            % orange
        case 'rff',     col = [0.47 0.67 0.19];                            % green
        otherwise,      col = [0.30 0.30 0.30];
    end
end


function lbl = default_label_for_cell(p, cbf_type)
    switch cbf_type
        case 'none',    pretty_m = 'No CBF';
        case 'shocbf',  pretty_m = 'HOCBF';
        case {'attc_vraw','sattc_vraw'}, pretty_m = 'aTTC-CBF';
        otherwise,      pretty_m = upper(cbf_type);
    end
    np = double(getfield_or(p, 'num_pursuers', 0));
    if np > 0
        adv = double(getfield_or(p, 'pursuit_advantage', 1.0));
        if adv < 0.999
            scen = sprintf('slow (%.1f)', adv);
        elseif adv > 1.001
            scen = sprintf('fast (%.1f)', adv);
        else
            scen = 'eq  (1.0)';
        end
        lbl = sprintf('%s\\newline%s', pretty_m, scen);
    else
        lbl = sprintf('%s\\newline0 pur', pretty_m);
    end
end


% =====================================================================
%  Inline metric helpers (fallback when metrics struct is missing)
% =====================================================================

function n = compute_collision_rate(res, em, kind)
%COMPUTE_COLLISION_RATE  Count rising-edge collision events on ee or ep
%   pairs.  Returns a *count* (caller does the per-100s normalization).
    p = res.params;
    na = double(p.num_agents);
    r_ttc = double(getfield_or(p, 'r_ttc', 0.1));
    if ~isfield(res, 'dist') || isempty(res.dist)
        n = 0; return;
    end

    % Build pair (i, j) list and classify
    pair_ij = zeros(na*(na-1)/2, 2);
    pp = 0;
    for ii = 1:na
        for jj = (ii+1):na
            pp = pp + 1;
            pair_ij(pp, :) = [ii, jj];
        end
    end
    is_target = false(size(pair_ij, 1), 1);
    for pp = 1:size(pair_ij, 1)
        ii = pair_ij(pp, 1); jj = pair_ij(pp, 2);
        switch kind
            case 'ee', is_target(pp) = em(ii) && em(jj);
            case 'ep', is_target(pp) = xor(em(ii), em(jj));
        end
    end
    pair_sel = find(is_target);
    if isempty(pair_sel), n = 0; return; end
    in = double(res.dist(:, pair_sel)) < 2 * r_ttc;
    n  = sum(diff([false(1, size(in, 2)); in], 1, 1) == 1, 'all');
end


function r_series = compute_meb_series_local(x, em, pdim)
%COMPUTE_MEB_SERIES_LOCAL  Per-step minimum-enclosing-ball radius of all
%   evaders.  Frank-Wolfe iteration (80 iters), accurate to <0.5% for
%   the small point sets we use.  Same algorithm as
%   compute_meb_series in auspice_plot_team_pursuit_compare.m.
    n_steps    = size(x, 1);
    evader_ids = find(em);
    r_series   = zeros(n_steps, 1);
    if numel(evader_ids) < 2, return; end
    for k = 1:n_steps
        P = squeeze(x(k, evader_ids, 1:pdim));
        if size(P, 2) == 1, P = P'; end
        r_series(k) = min_enclosing_ball_radius(P);
    end
end


function [R, lbl] = compute_best_meb(v_space, alpha, n_e)
%COMPUTE_BEST_MEB  Nominal-formation MEB radius for a V chevron, general
%   half-angle alpha ∈ (0, pi/2].  The three boundary candidates are the
%   leader and the two wing tips; intermediate slots always sit inside.
%   The triangle they form has:
%       sides   a = L_R (leader → right tip)
%               b = L_L (leader → left  tip)
%               c = √(a² + b² − 2 a b cos 2α)     (Law of Cosines)
%       angles  apex (at leader) = 2α
%               A_R, A_L         (at the wing tips)
%   The MEB depends on which (if any) angle is obtuse:
%       (1) 2α ≤ acos(L_min/L_max):  obtuse at the SHORT-wing tip
%               → MEB diameter = L_max (the long arm), R = L_max / 2
%       (2) acos(L_min/L_max) < 2α < π/2:  acute triangle
%               → MEB = circumradius = c / (2 sin 2α)
%       (3) 2α ≥ π/2:  obtuse (or right) at the apex
%               → MEB diameter = c (tip-to-tip), R = c / 2
%   Sanity at boundaries:
%       2α = π/2:  c = √(a²+b²), case (2) and (3) agree → R = √(a²+b²)/2
%       L_R = L_L = L (symmetric V):  case (1) impossible; (2) gives
%               R = L / (2 cos(π/2 − α)) = L sin(...) … = L cos((π−2α)/2)
%               which simplifies to L / (2 cos α) — the well-known
%               isosceles-triangle circumradius.
%
%   Args
%     v_space  formation_spacing [km]
%     alpha    formation_stagger_angle [rad]   (V half-angle)
%     n_e      number of evaders (incl. leader)
    R = NaN; lbl = '';
    if ~(isfinite(v_space) && isfinite(alpha) && isfinite(n_e) && n_e >= 2)
        return;
    end
    L_R = ceil ((n_e - 1) / 2) * v_space;
    L_L = floor((n_e - 1) / 2) * v_space;
    L_max = max(L_R, L_L);
    L_min = min(L_R, L_L);

    two_alpha = 2 * alpha;
    % Threshold below which the short-wing-tip angle goes obtuse
    if L_max > 0
        two_alpha_thresh = acos(L_min / L_max);     % ∈ [0, π/2]
    else
        two_alpha_thresh = 0;
    end

    if two_alpha <= two_alpha_thresh + 1e-12
        % Case 1: obtuse at the SHORT-wing tip (or degenerate at α=0)
        R = L_max / 2;
    elseif two_alpha < pi/2 - 1e-12
        % Case 2: acute triangle — circumradius
        c = sqrt(L_R^2 + L_L^2 - 2 * L_R * L_L * cos(two_alpha));
        R = c / (2 * sin(two_alpha));
    else
        % Case 3: obtuse (or right) at apex — tip-to-tip as diameter
        c = sqrt(L_R^2 + L_L^2 - 2 * L_R * L_L * cos(two_alpha));
        R = c / 2;
    end
    lbl = sprintf('best MEB %.2f km', R);
end


% NOTE: min_enclosing_ball_radius is now a standalone function
% (Matlab_Code/min_enclosing_ball_radius.m).  Keep any algorithm changes
% there so all callers stay in sync.


% =====================================================================
%  Misc helpers
% =====================================================================

function plot_nm_lookahead_avg(cells, pair_kinds, n_events, dts, ...
        delta_ts, speed_wins, yaw_wins, pitch_wins, accel_wins, ...
        T_lookaheads, spec, save_dir, name_tag, ...
        fn, fs_label, fs_tick, fs_legend, lw, figsize, do_save)
%PLOT_NM_LOOKAHEAD_AVG  See auspice_plot_journal.m — identical logic.
    n_cells = numel(cells);
    n_T     = numel(T_lookaheads);

    fig = figure('Name', spec.fname, 'Position', [60 60 figsize], 'Color', 'w');
    set(fig, 'DefaultAxesFontName', fn, ...
             'DefaultTextFontName', fn, ...
             'DefaultLegendFontName', fn);
    hold on; grid on; box on;

    h_lines_av = gobjects(n_cells, 1);
    h_lines_co = gobjects(n_cells, 1);
    leg_av     = cell(n_cells, 1);
    leg_co     = cell(n_cells, 1);
    drew_av = false(n_cells, 1);
    drew_co = false(n_cells, 1);

    for k = 1:n_cells
        c = cells{k};
        if ~strcmp(pair_kinds{k}, 'ep'), continue; end
        if n_events(k) == 0, continue; end

        dt_k = dts(k);
        switch spec.data
            case 'speed', win = speed_wins{k};
            case 'yaw',   win = yaw_wins{k};
            case 'pitch', win = pitch_wins{k};
            case 'accel', win = accel_wins{k};
            case 'angrate'
                yw = yaw_wins{k}; pw = pitch_wins{k};
                if isempty(yw) || isempty(pw), win = [];
                else, win = sqrt(yw.^2 + pw.^2); end
            otherwise, error('unknown metric %s', spec.data);
        end
        if isempty(win), continue; end
        dtc = delta_ts{k};
        nE  = size(win, 2);
        nMax = size(win, 1);

        avg_av = nan(n_T, 1);
        avg_co = nan(n_T, 1);
        n_av_T = zeros(n_T, 1);
        n_co_T = zeros(n_T, 1);

        for ti = 1:n_T
            T = T_lookaheads(ti);
            n_T_steps = round(T / dt_k) + 1;
            is_col = dtc <= T;

            n_use = ones(nE, 1) * n_T_steps;
            idx_col = find(is_col);
            n_use(idx_col) = round(dtc(idx_col) / dt_k) + 1;
            n_use = min(max(n_use, 1), nMax);

            per_ev = nan(nE, 1);
            for ei = 1:nE
                per_ev(ei) = mean(win(1:n_use(ei), ei), 'omitnan');
            end
            if any(is_col)
                avg_co(ti) = mean(per_ev(is_col), 'omitnan');
                n_co_T(ti) = sum(is_col);
            end
            if any(~is_col)
                avg_av(ti) = mean(per_ev(~is_col), 'omitnan');
                n_av_T(ti) = sum(~is_col);
            end
        end

        if any(~isnan(avg_av))
            h_lines_av(k) = plot(T_lookaheads, avg_av, '-', ...
                'Color', c.color, 'LineWidth', lw, ...
                'Marker', c.marker, 'MarkerSize', 12, ...
                'MarkerFaceColor', c.color, 'MarkerEdgeColor', 'k');
            leg_av{k} = sprintf('%s avoided (n=%d..%d)', c.label, ...
                min(n_av_T(n_av_T > 0)), max(n_av_T));
            drew_av(k) = true;
        end
        if any(~isnan(avg_co))
            h_lines_co(k) = plot(T_lookaheads, avg_co, '--', ...
                'Color', c.color, 'LineWidth', lw, ...
                'Marker', c.marker, 'MarkerSize', 12, ...
                'MarkerFaceColor', 'w', 'MarkerEdgeColor', c.color);
            leg_co{k} = sprintf('%s collided (n=%d..%d)', c.label, ...
                min(n_co_T(n_co_T > 0)), max(n_co_T));
            drew_co(k) = true;
        end
    end

    xlabel('Lookahead horizon T [s]', 'FontName', fn, 'FontSize', fs_label);
    ylabel(spec.ylab, 'FontName', fn, 'FontSize', fs_label);
    set(gca, 'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    xlim([min(T_lookaheads) - 0.05, max(T_lookaheads) + 0.05]);

    h_leg = gobjects(0, 1); l_leg = {};
    for k = 1:n_cells
        if drew_av(k), h_leg(end+1) = h_lines_av(k); l_leg{end+1} = leg_av{k}; end %#ok<AGROW>
        if drew_co(k), h_leg(end+1) = h_lines_co(k); l_leg{end+1} = leg_co{k}; end %#ok<AGROW>
    end
    if ~isempty(h_leg)
        legend(h_leg, l_leg, 'FontName', fn, 'FontSize', fs_legend, ...
            'Location', 'best', 'Box', 'on');
    end
    save_three_maybe(fig, fullfile(save_dir, [name_tag spec.fname]), do_save);
end


function plot_nm_outcome_scatter(cells, cell_idx, dts, delta_ts, ...
        speed_wins, yaw_wins, pitch_wins, accel_wins, ...
        T_scatter, max_pts, spec, scenario, ...
        save_dir, name_tag, fn, fs_label, fs_tick, fs_legend, lw, figsize, do_save)
%PLOT_NM_OUTCOME_SCATTER  See auspice_plot_journal.m — identical logic.
    col_collision = [0.85 0.10 0.10];
    col_avoided   = [0.00 0.45 0.74];

    fname = sprintf('%s_%s', spec.fname, scenario.tag);
    fig = figure('Name', fname, 'Position', [60 60 figsize], 'Color', 'w');
    set(fig, 'DefaultAxesFontName', fn, ...
             'DefaultTextFontName', fn, ...
             'DefaultLegendFontName', fn);
    hold on; grid on; box on;

    rng(0);
    n_cols = numel(cell_idx);
    xtl    = cell(n_cols, 1);

    h_red  = scatter(nan, nan, 90, col_collision, 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.6);
    h_blue = scatter(nan, nan, 90, col_avoided,   'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.6);

    for col_i = 1:n_cols
        k    = cell_idx(col_i);
        dt_k = dts(k);
        dtc  = delta_ts{k};

        switch spec.var
            case 'speed', win = speed_wins{k};
            case 'yaw',   win = yaw_wins{k};
            case 'pitch', win = pitch_wins{k};
            case 'accel', win = accel_wins{k};
            case 'angrate'
                yw = yaw_wins{k}; pw = pitch_wins{k};
                if isempty(yw) || isempty(pw), win = [];
                else, win = sqrt(yw.^2 + pw.^2); end
            otherwise, error('unknown var %s', spec.var);
        end
        if isempty(win), continue; end
        nE   = size(win, 2);
        nMax = size(win, 1);

        n_T_steps = round(T_scatter / dt_k) + 1;
        is_col    = dtc <= T_scatter;
        n_use     = ones(nE, 1) * n_T_steps;
        idx_col   = find(is_col);
        n_use(idx_col) = round(dtc(idx_col) / dt_k) + 1;
        n_use = min(max(n_use, 1), nMax);

        vals = nan(nE, 1);
        for ei = 1:nE
            vals(ei) = mean(win(1:n_use(ei), ei), 'omitnan');
        end

        vals_col = vals(is_col);    vals_col = vals_col(~isnan(vals_col));
        vals_av  = vals(~is_col);   vals_av  = vals_av(~isnan(vals_av));
        n_col = numel(vals_col);
        n_av  = numel(vals_av);

        if n_col > max_pts
            vals_col_plot = vals_col(randperm(n_col, max_pts));
        else
            vals_col_plot = vals_col;
        end
        if n_av > max_pts
            vals_av_plot = vals_av(randperm(n_av, max_pts));
        else
            vals_av_plot = vals_av;
        end

        jit_av  = col_i + 0.18 * (rand(numel(vals_av_plot), 1) * 2 - 1);
        jit_col = col_i + 0.18 * (rand(numel(vals_col_plot), 1) * 2 - 1);
        scatter(jit_av,  vals_av_plot,  60, col_avoided,   'filled', ...
            'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.6, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.4);
        scatter(jit_col, vals_col_plot, 60, col_collision, 'filled', ...
            'MarkerFaceAlpha', 0.45, 'MarkerEdgeAlpha', 0.8, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.4);

        if ~isempty(vals_av)
            mu_av = mean(vals_av);
            plot([col_i - 0.30, col_i + 0.30], [mu_av mu_av], '-', ...
                'Color', col_avoided, 'LineWidth', lw + 1.0);
        end
        if ~isempty(vals_col)
            mu_col = mean(vals_col);
            plot([col_i - 0.30, col_i + 0.30], [mu_col mu_col], '-', ...
                'Color', col_collision, 'LineWidth', lw + 1.0);
        end

        xtl{col_i} = cells{k}.label;
    end

    set(gca, 'XTick', 1:n_cols, 'XTickLabel', xtl, ...
        'FontName', fn, 'FontSize', fs_tick, 'LineWidth', 1.2);
    xlim([0.4, n_cols + 0.6]);
    ylabel(spec.label, 'FontName', fn, 'FontSize', fs_label);
    legend([h_blue, h_red], ...
        {sprintf('avoided  (T = %.1f s)', T_scatter), ...
         sprintf('collided (\\Delta t \\leq %.1f s)', T_scatter)}, ...
        'FontName', fn, 'FontSize', fs_legend, 'Location', 'best');
    save_three_maybe(fig, fullfile(save_dir, [name_tag fname]), do_save);
end


function fig_defaults(fig, fn)
    set(fig, 'DefaultAxesFontName',  fn, ...
             'DefaultTextFontName',  fn, ...
             'DefaultLegendFontName', fn);
end


function v = getfield_or(s, fld, default)
    if isstruct(s) && isfield(s, fld), v = s.(fld); else, v = default; end
end


function save_three(fig, base)
    try
        exportgraphics(fig, [base '.png'], 'Resolution', 300);
        exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
        savefig(fig, [base '.fig']);
        fprintf('  saved %s.{png,pdf,fig}\n', base);
    catch ME
        fprintf('  [warn] could not save %s : %s\n', base, ME.message);
    end
end


function save_three_maybe(fig, base, do_save)
%SAVE_THREE_MAYBE  Only save when do_save is true; else no-op.
    if do_save
        save_three(fig, base);
    end
end


function tf = want_figure(figure_list, group)
%WANT_FIGURE  True iff `group` should be rendered under `figure_list`.
%   'full' selects everything; any other value renders only the matching group.
    tf = strcmp(figure_list, 'full') || strcmp(figure_list, group);
end
