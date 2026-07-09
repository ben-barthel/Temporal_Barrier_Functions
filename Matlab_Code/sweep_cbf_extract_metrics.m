function m = sweep_cbf_extract_metrics(R)
%SWEEP_CBF_EXTRACT_METRICS  Pull the per-run summary metrics needed for the
%   CBF parameter sweep.  Mirrors the logic in
%   auspice_plot_team_pursuit_compare so values are directly comparable.
%
%   Returns a struct with fields:
%     col_ee_count, col_ep_count    rising-edge collision counts
%     time_below_ee, time_below_ep  total pair-seconds inside threshold
%     mean_evader_v                 mean speed of evaders over the run
%     wp_progress_rate_per100s      per-evader (or leader-only for
%                                   formation*) waypoint clears per 100 s
%     min_dist                      smallest pair-distance over the run
%     n_evaders, n_pursuers         agent counts (for sanity)

    params = R.params;
    na = double(params.num_agents);
    dt = double(params.dt);

    if isfield(params, 'r_ttc') && double(params.r_ttc) > 0
        r_col = double(params.r_ttc);
    else
        r_col = double(params.r_cbf);
    end
    threshold = 2 * r_col;

    % Evader mask resolution (mirrors auspice_plot_team_pursuit_compare's
    % extract_team_info).  For non-pursuit missions, evader_mask is not
    % populated and we must NOT default to "agent 1 only" — that
    % mis-classifies every pair as evader-pursuer and zeroes out col_ee.
    if isfield(params, 'evader_mask') && ~isempty(params.evader_mask)
        em = logical(params.evader_mask(:));
    elseif isfield(params, 'pursuit_mode') && params.pursuit_mode
        em = false(na, 1); em(1) = true;
    elseif isfield(params, 'mission') && ...
           any(strcmp(strtrim(char(params.mission)), ...
               {'pursuit','team_pursuit','random_sphere_pursuit', ...
                'formation_pursuit'}))
        em = false(na, 1); em(1) = true;
    else
        em = true(na, 1);          % no pursuit -> everyone's an evader
    end

    % Pair classification (i<j ordering matches auspice_sim's compute_all_pairs)
    pair_ee = []; pair_ep = [];
    pair = 0;
    for i = 1:na
        for j = (i+1):na
            pair = pair + 1;
            ei = em(i); ej = em(j);
            if ei && ej
                pair_ee(end+1) = pair; %#ok<AGROW>
            elseif xor(ei, ej)
                pair_ep(end+1) = pair; %#ok<AGROW>
            end
        end
    end

    dist  = double(R.dist);
    if isempty(pair_ee)
        ee_in = false(size(dist, 1), 0);
    else
        ee_in = dist(:, pair_ee) < threshold;
    end
    if isempty(pair_ep)
        ep_in = false(size(dist, 1), 0);
    else
        ep_in = dist(:, pair_ep) < threshold;
    end

    % Rising-edge counts
    edges_ee = diff([false(1, size(ee_in, 2)); ee_in], 1, 1) == 1;
    edges_ep = diff([false(1, size(ep_in, 2)); ep_in], 1, 1) == 1;

    % Mean evader speed
    if double(params.dim) == 3, v_idx = 6; else, v_idx = 4; end
    v_evader = double(R.x(:, em, v_idx));

    % WP progress: leader for formation*, average evaders otherwise.
    miss = strtrim(char(params.mission));
    if any(strcmp(miss, {'formation', 'formation_pursuit'}))
        n_used = 1;
        agent_ids = 1;
    else
        agent_ids = find(em);
        n_used    = numel(agent_ids);
    end
    if isfield(R, 'wp_advance_count') && ~isempty(R.wp_advance_count)
        n_adv_total = sum(double(R.wp_advance_count(agent_ids)));
    else
        % Legacy fallback: same code as the plotter's count_wp_advances
        n_adv_total = 0;
        capture_r   = double(params.wp_capture_radius);
        pdim        = double(params.dim);
        for ii = agent_ids(:)'
            wps  = params.waypoints{ii};
            n_wp = size(wps, 1);
            if n_wp < 2, continue; end
            wp_idx = 1; n_loc = 0;
            for k = 1:size(R.x, 1)
                p = squeeze(double(R.x(k, ii, 1:pdim)))';
                if norm(p - wps(wp_idx, 1:pdim)) < capture_r
                    n_loc = n_loc + 1;
                    wp_idx = mod(wp_idx, n_wp) + 1;
                end
            end
            n_adv_total = n_adv_total + n_loc;
        end
    end
    wp_rate = (n_adv_total / max(n_used, 1)) / (double(params.T_sim) / 100);

    % Formation coherence (NaN for non-formation missions).
    %   coh_dim:  scalar mean follower-slot deviation [km], lower better
    %   coh_norm: exp(-coh_dim/formation_spacing), in (0,1], higher better
    %   coh_pf:   per-follower vector of mean slot deviations [km],
    %             length n_eva-1; needed by collators that compute the
    %             relative-to-baseline ratio d0_j / d_j.
    info.evader_mask = em;
    info.dim         = double(params.dim);
    h_coh    = compute_formation_coherence_helpers();
    coh_dim  = h_coh.coherence(R, info);
    coh_norm = h_coh.coherence_norm(R, info);
    coh_pf   = h_coh.coherence_per_follower(R, info);

    m = struct();
    m.col_ee_count                       = sum(edges_ee(:));
    m.col_ep_count                       = sum(edges_ep(:));
    m.time_below_ee                      = sum(ee_in(:)) * dt;
    m.time_below_ep                      = sum(ep_in(:)) * dt;
    m.mean_evader_v                      = mean(v_evader(:));
    m.wp_progress_rate_per100s           = wp_rate;
    m.min_dist                           = min(dist(:));
    m.formation_coherence                = coh_dim;
    m.formation_coherence_norm           = coh_norm;
    m.formation_coherence_per_follower   = coh_pf;
    m.n_evaders                          = sum(em);
    m.n_pursuers                         = na - sum(em);
end
