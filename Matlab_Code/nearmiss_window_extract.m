function ed = nearmiss_window_extract(res, opts)
%NEARMISS_WINDOW_EXTRACT  Detect near-miss events (rising edges of
%   d < d_near) in an auspice_sim result and report, per event, the
%   time elapsed until the first d < d_collision moment (Inf if no
%   collision happens within +T_max seconds).
%
%   Lean version (no per-event window time-series): only the data needed
%   to compute P(collision | near-miss) at any T_lookahead value.
%
%   Inputs
%   ------
%   res          auspice_sim result struct (needs x, dist, params; and u
%                if return_metric_windows is true)
%   opts         optional struct:
%       opts.d_near       default 4 * params.r_ttc
%       opts.d_collision  default 2 * params.r_ttc
%       opts.T_max        seconds; lookahead horizon  (default 10)
%       opts.pair_kind    'ep' (default for missions with pursuers) or
%                          'ee' (fallback / no-pursuer missions)
%       opts.return_metric_windows  default false.  If true, also return
%                          per-event evader metric traces over [t_nm,
%                          t_nm + T_max] for speed and absolute-value
%                          control inputs.  Used by the cross-outcome
%                          analysis figures in auspice_plot_journal*.
%
%   Output
%   ------
%   ed.n_events            scalar — number of near-miss rising edges
%   ed.pair_kind           'ep' or 'ee'
%   ed.d_near              threshold actually used [km]
%   ed.d_collision         threshold actually used [km]
%   ed.T_max               lookahead horizon [s]
%   ed.dt                  sim timestep [s]
%   ed.delta_t_collision   (n_events, 1) — seconds from t_nm to first
%                          d < d_collision; Inf if no collision in +T_max
%   ed.t_nm_idx            (n_events, 1) — step index of each near-miss
%   ed.ev_id               (n_events, 1) — evader agent index per event
%   When opts.return_metric_windows is true:
%   ed.speed_window        (n_steps_max, n_events) — evader speed v(t)
%   ed.yaw_window          (n_steps_max, n_events) — |omega(t)| applied
%   ed.pitch_window        (n_steps_max, n_events) — |nu(t)| applied
%   ed.accel_window        (n_steps_max, n_events) — |a(t)| applied
%   Windows are aligned so row 1 corresponds to t = t_nm (offset 0) and
%   later rows cover t = t_nm + (row-1)*dt.  Samples past the end of
%   the run, or past size(res.u, 1) for control fields, are NaN.

    if nargin < 2, opts = struct(); end
    p   = res.params;
    dt  = double(p.dt);
    na  = double(p.num_agents);
    r_ttc = double(getfield_or(p, 'r_ttc', 0.1));

    d_near      = getfield_or(opts, 'd_near',      4 * r_ttc);
    d_collision = getfield_or(opts, 'd_collision', 2 * r_ttc);
    T_max       = getfield_or(opts, 'T_max',       10);
    return_metric_windows = getfield_or(opts, 'return_metric_windows', false);

    if isfield(p, 'evader_mask') && ~isempty(p.evader_mask)
        em = logical(p.evader_mask(:));
    else
        em = true(na, 1);
    end
    n_pu = sum(~em);

    pair_kind = getfield_or(opts, 'pair_kind', '');
    if isempty(pair_kind)
        if n_pu > 0, pair_kind = 'ep'; else, pair_kind = 'ee'; end
    end

    % Build pair list and class masks
    pair_ij = zeros(na*(na-1)/2, 2);
    pp = 0;
    for ii = 1:na
        for jj = (ii+1):na
            pp = pp + 1;
            pair_ij(pp, :) = [ii, jj];
        end
    end
    is_ee = false(size(pair_ij, 1), 1);
    is_ep = false(size(pair_ij, 1), 1);
    for pp = 1:size(pair_ij, 1)
        ii = pair_ij(pp,1); jj = pair_ij(pp,2);
        if em(ii) && em(jj),         is_ee(pp) = true;
        elseif xor(em(ii), em(jj)),  is_ep(pp) = true;
        end
    end
    switch pair_kind
        case 'ep', pair_sel = find(is_ep);
        case 'ee', pair_sel = find(is_ee);
        otherwise
            error('nearmiss_window_extract:pair_kind', ...
                  'unknown pair_kind %s', pair_kind);
    end

    dist = double(res.dist);                            % (N+1, n_pairs)
    n_steps = size(dist, 1);

    % Rising edges of d < d_near, per pair
    in_near = dist(:, pair_sel) < d_near;
    edges_in = diff([false(1, size(in_near, 2)); in_near], 1, 1) == 1;
    [t_nm_idx, p_nm_idx] = find(edges_in);

    n_events = numel(t_nm_idx);
    max_step_look = round(T_max / dt);
    n_steps_window = max_step_look + 1;        % inclusive of t_nm

    delta_t_collision = inf(n_events, 1);
    ev_id_vec = zeros(n_events, 1);

    % Optional per-event evader metric windows (speed + |omega|, |nu|, |a|)
    want_metrics = return_metric_windows;
    if want_metrics
        spd_idx = 6;                            % 3D Dubins state layout
        speed_window = nan(n_steps_window, n_events);
        yaw_window   = nan(n_steps_window, n_events);
        pitch_window = nan(n_steps_window, n_events);
        accel_window = nan(n_steps_window, n_events);
        if isfield(res, 'u') && ~isempty(res.u)
            u_all = double(res.u);
            n_u_steps = size(u_all, 1);
        else
            u_all = [];
            n_u_steps = 0;
        end
        x_all = double(res.x);
    end

    for ei = 1:n_events
        ti      = t_nm_idx(ei);
        p_glob  = pair_sel(p_nm_idx(ei));
        t_end   = min(ti + max_step_look, n_steps);
        d_slice = dist(ti : t_end, p_glob);
        i_col   = find(d_slice < d_collision, 1, 'first');
        if ~isempty(i_col)
            delta_t_collision(ei) = (i_col - 1) * dt;
        end

        % Identify the evader agent for this event
        ii = pair_ij(p_glob, 1);
        jj = pair_ij(p_glob, 2);
        if strcmp(pair_kind, 'ep')
            if em(ii) && ~em(jj),       ev_id = ii;
            elseif em(jj) && ~em(ii),   ev_id = jj;
            else,                       ev_id = ii;
            end
        else
            ev_id = ii;
        end
        ev_id_vec(ei) = ev_id;

        % --- Per-event metric windows (only if requested) ---
        if want_metrics
            n_avail = t_end - ti + 1;
            speed_window(1:n_avail, ei) = ...
                squeeze(x_all(ti:t_end, ev_id, spd_idx));
            if ~isempty(u_all)
                t_end_u = min(t_end, n_u_steps);
                n_u_avail = t_end_u - ti + 1;
                if n_u_avail >= 1
                    u_slice = squeeze(u_all(ti:t_end_u, ev_id, :));  % (T, 3)
                    if size(u_slice, 2) ~= 3 && size(u_slice, 1) == 3
                        u_slice = u_slice';            % handle 1-row edge case
                    end
                    rng_u = 1 : n_u_avail;
                    yaw_window  (rng_u, ei) = abs(u_slice(:, 1));
                    pitch_window(rng_u, ei) = abs(u_slice(:, 2));
                    accel_window(rng_u, ei) = abs(u_slice(:, 3));
                end
            end
        end
    end

    ed.n_events            = n_events;
    ed.pair_kind           = pair_kind;
    ed.d_near              = d_near;
    ed.d_collision         = d_collision;
    ed.T_max               = T_max;
    ed.dt                  = dt;
    ed.delta_t_collision   = delta_t_collision;
    ed.t_nm_idx            = t_nm_idx;
    ed.ev_id               = ev_id_vec;
    if want_metrics
        ed.speed_window = speed_window;
        ed.yaw_window   = yaw_window;
        ed.pitch_window = pitch_window;
        ed.accel_window = accel_window;
    end
end


% ---------------------------------------------------------------------
function v = getfield_or(s, fld, default)
    if isstruct(s) && isfield(s, fld), v = s.(fld); else, v = default; end
end
