function results = compute_attc_trajectory(results, r_ttc, opts)
%COMPUTE_ATTC_TRAJECTORY  Compute aTTC for an entire trajectory.
%
%   results = compute_attc_trajectory(results, r_ttc)
%   results = compute_attc_trajectory(results, r_ttc, opts)
%
%   Takes a results struct (from auspice_sim or loaded from .mat) and an
%   r_ttc value.  If the results already contain a finite 'attc' field it
%   is returned unchanged; otherwise the aTTC is computed for every
%   timestep and added to the results struct.
%
%   Inputs:
%     results   - struct with at least:
%                   .x      [N x na x sd]  state history
%                   .params struct          simulation parameters
%     r_ttc     - collision radius for aTTC (km).  Default: 0.1
%     opts      - (optional) struct with fields:
%                   .attc_dt      aTTC timestep [s] (default: 0.2)
%                   .attc_horizon aTTC horizon  [s] (default: params.ttc_horizon)
%                   .attc_type    'bang', 'prop', 'pursuit_ego', or 'pursuit_min'
%                                 (default: 'pursuit_min')
%                   .verbose      print progress   (default: true)
%
%   Outputs:
%     results   - same struct with 'attc' field added/updated.
%                 Size: [N x num_pairs]

    arguments
        results   struct
        r_ttc     (1,1) double = 0.1
        opts.attc_dt      (1,1) double = 0.2
        opts.attc_horizon (1,1) double = NaN   % NaN = use params.ttc_horizon
        opts.attc_type    char = 'pursuit_ego'     % 'bang','prop','pursuit_ego','pursuit_min'
        opts.verbose      (1,1) logical = true
    end

    % -----------------------------------------------------------------
    % 1. Check if aTTC already exists and has finite values
    % -----------------------------------------------------------------
    if isfield(results, 'attc')
        n_finite = sum(isfinite(results.attc(:)));
        if n_finite > 0
            if opts.verbose
                pct = 100 * n_finite / numel(results.attc);
                fprintf('[compute_attc_trajectory] ''attc'' already present with %d finite values (%.1f%%). Returning unchanged.\n', ...
                    n_finite, pct);
            end
            return;
        end
    end

    % -----------------------------------------------------------------
    % 2. Build minimal parameter struct for aTTC computation
    % -----------------------------------------------------------------
    params = extract_params(results, r_ttc, opts);

    % -----------------------------------------------------------------
    % 3. Normalise the state array layout
    % -----------------------------------------------------------------
    x = results.x;
    sd = params.state_dim;
    na = params.num_agents;

    % results.x may be (sd, na, N) from auspice_sim or (N, na, sd) —
    % detect by checking which dimension equals sd.
    if ndims(x) == 3
        if size(x,1) == sd && size(x,3) ~= sd
            % Layout (sd, na, N) — permute to (N, na, sd)
            x = permute(x, [3 2 1]);
        end
    elseif ismatrix(x) && size(x,2) == sd
        % (N, sd) single-agent — reshape
        x = reshape(x, [size(x,1), 1, sd]);
    end

    N = size(x, 1);
    num_pairs = na * (na - 1) / 2;

    if opts.verbose
        fprintf('[compute_attc_trajectory] Computing aTTC for %d timesteps, %d agents (%d pairs), dim=%d\n', ...
            N, na, num_pairs, params.dim);
        fprintf('  r_ttc=%.4f km, attc_dt=%.4f s, attc_horizon=%.1f s\n', ...
            params.r_ttc, params.attc_dt, params.attc_horizon);
    end

    % -----------------------------------------------------------------
    % 4. Compute aTTC for every timestep
    % -----------------------------------------------------------------
    attc_hist = Inf * ones(N, num_pairs);
    d_gate = 2 * params.v_max * params.attc_horizon;
    report_interval = max(floor(N / 20), 1);

    for k = 1:N
        if opts.verbose && (mod(k-1, report_interval) == 0 || k == N)
            fprintf('\r  timestep %d/%d (%.0f%%)', k, N, 100*k/N);
        end

        x_curr = squeeze(x(k, :, :));   % (na, sd)
        if na == 1
            x_curr = x_curr(:)';        % ensure row
        end

        pair = 0;
        for i = 1:na
            for j = i+1:na
                pair = pair + 1;
                if params.dim == 2
                    dp = x_curr(i, 1:2) - x_curr(j, 1:2);
                else
                    dp = x_curr(i, 1:3) - x_curr(j, 1:3);
                end
                if norm(dp) <= d_gate
                    attc_hist(k, pair) = compute_attc( ...
                        x_curr(i,:), x_curr(j,:), params);
                end
            end
        end
    end

    if opts.verbose
        n_fin = sum(isfinite(attc_hist(:)));
        pct = 100 * n_fin / numel(attc_hist);
        fprintf('\n  Done. %d finite aTTC values (%.1f%%)\n', n_fin, pct);
    end

    % -----------------------------------------------------------------
    % 5. Attach to results and return
    % -----------------------------------------------------------------
    results.attc = attc_hist;
end


% =====================================================================
%                     PARAMETER EXTRACTION
% =====================================================================

function params = extract_params(results, r_ttc, opts)
%EXTRACT_PARAMS  Build minimal parameter struct for aTTC computation.
    p = results.params;

    params.dim         = get_field(p, 'dim', 2);
    params.num_agents  = get_field(p, 'num_agents', size(results.x, 2));
    params.state_dim   = 4 * (params.dim == 2) + 6 * (params.dim == 3);
    params.r_ttc       = r_ttc;
    params.v_max       = get_field(p, 'v_max', 0.5);
    params.omega_max   = get_field(p, 'omega_max', 0.1);
    params.a_max       = get_field(p, 'a_max', 0.05);
    params.nu_max      = get_field(p, 'nu_max', 0.1);
    params.ttc_horizon = get_field(p, 'ttc_horizon', 60);

    % aTTC-specific
    params.attc_dt   = opts.attc_dt;
    params.attc_type = opts.attc_type;
    if isnan(opts.attc_horizon)
        params.attc_horizon = params.ttc_horizon;
    else
        params.attc_horizon = opts.attc_horizon;
    end

    % Safety clamp on attc_dt
    if params.r_ttc > 0
        max_safe_dt = params.r_ttc / params.v_max;
        if params.attc_dt > max_safe_dt
            warning('attc_dt (%.3g) > r_ttc/v_max (%.3g); clamping.', ...
                params.attc_dt, max_safe_dt);
            params.attc_dt = max_safe_dt;
        end
    end
end


function val = get_field(s, name, default)
%GET_FIELD  Safely read a field from a struct with default.
    if isfield(s, name)
        val = double(s.(name));
        if numel(val) == 1
            val = val(1);   % unwrap scalar arrays
        end
    else
        val = default;
    end
end


% =====================================================================
%                    aTTC FORWARD SIMULATION
% =====================================================================

function attc = compute_attc(xi, xj, params)
%COMPUTE_ATTC  Adversarial TTC for a single pair — dispatches by attc_type.
    mode = params.attc_type;
    if strcmp(mode, 'pursuit_ego')
        % i = ego (nominal), j = pursuer — one direction only
        attc = compute_attc_pursuit(xi, xj, params);
    elseif strcmp(mode, 'pursuit_min')
        % Both directions, return min (worst case)
        a1 = compute_attc_pursuit(xi, xj, params);
        a2 = compute_attc_pursuit(xj, xi, params);
        attc = min(a1, a2);
    else
        % Symmetric (bang / prop): both agents crash-seek
        attc = compute_attc_symmetric(xi, xj, params);
    end
end


function attc = compute_attc_symmetric(xi, xj, params)
%COMPUTE_ATTC_SYMMETRIC  Both agents crash-seek toward each other.
    dt_w    = params.attc_dt;
    horizon = params.attc_horizon;
    r_col   = 2 * params.r_ttc;
    r_col2  = r_col^2;
    N_w     = ceil(horizon / dt_w);

    si = xi(:)';   % row vectors
    sj = xj(:)';

    for step = 1:N_w
        ui = crash_seeking_control(si, sj, params);
        uj = crash_seeking_control(sj, si, params);
        dxi = dynamics_row(si, ui, params);
        dxj = dynamics_row(sj, uj, params);
        si = si + dt_w * dxi;
        sj = sj + dt_w * dxj;

        % Clamp speed
        if params.dim == 2
            si(4) = max(0, min(params.v_max, si(4)));
            sj(4) = max(0, min(params.v_max, sj(4)));
        else
            si(6) = max(0, min(params.v_max, si(6)));
            sj(6) = max(0, min(params.v_max, sj(6)));
        end

        % Check collision
        if params.dim == 2
            dp = si(1:2) - sj(1:2);
        else
            dp = si(1:3) - sj(1:3);
        end

        if dp*dp' <= r_col2
            attc = step * dt_w;
            return;
        end
    end

    attc = Inf;
end


function attc = compute_attc_pursuit(x_nom, x_pur, params)
%COMPUTE_ATTC_PURSUIT  Asymmetric pursuit: nominal maintains course, pursuer chases.
%   x_nom = nominal agent state (maintains current heading/speed, zero control)
%   x_pur = pursuer agent state (proportional heading + max acceleration)
%   Uses continuous closest-approach detection between Euler steps.
    dt_w    = params.attc_dt;
    horizon = params.attc_horizon;
    r_col   = 2 * params.r_ttc;
    r_col2  = r_col^2;
    N_w     = ceil(horizon / dt_w);
    K_hdg   = 2.5;          % proportional heading gain
    pd      = 1:min(params.dim, 3);   % position indices

    s_nom = x_nom(:)';
    s_pur = x_pur(:)';

    for step = 1:N_w
        s_nom_old = s_nom;
        s_pur_old = s_pur;

        % Nominal agent: zero control (maintains heading & cruises)
        if params.dim == 2
            u_nom = [0, 0];
        else
            u_nom = [0, 0, 0];
        end

        % Pursuer: proportional heading toward nominal + max acceleration
        u_pur = pursuit_control(s_pur, s_nom, K_hdg, params);

        % Euler integration
        dx_nom = dynamics_row(s_nom, u_nom, params);
        dx_pur = dynamics_row(s_pur, u_pur, params);
        s_nom  = s_nom + dt_w * dx_nom;
        s_pur  = s_pur + dt_w * dx_pur;

        % Clamp speed
        if params.dim == 2
            s_nom(4) = max(0, min(params.v_max, s_nom(4)));
            s_pur(4) = max(0, min(params.v_max, s_pur(4)));
        else
            s_nom(6) = max(0, min(params.v_max, s_nom(6)));
            s_pur(6) = max(0, min(params.v_max, s_pur(6)));
        end

        % Continuous closest-approach check
        dp0 = s_nom_old(pd) - s_pur_old(pd);
        ddp = (s_nom(pd) - s_nom_old(pd)) - (s_pur(pd) - s_pur_old(pd));
        a_coeff = ddp * ddp';
        b_coeff = dp0 * ddp';

        if a_coeff > 0
            t_star = max(0, min(1, -b_coeff / a_coeff));
        else
            t_star = 0;
        end

        dp_min = dp0 + t_star * ddp;
        if dp_min * dp_min' <= r_col2
            attc = (step - 1 + t_star) * dt_w;
            return;
        end

        % Also check endpoint
        dp_end = s_nom(pd) - s_pur(pd);
        if dp_end * dp_end' <= r_col2
            attc = step * dt_w;
            return;
        end
    end

    attc = Inf;
end


function u = pursuit_control(s_pursuer, s_target, K_hdg, params)
%PURSUIT_CONTROL  Proportional heading toward target + max acceleration.
%   2D: u = [omega, a_max]
%   3D: u = [u_psi, u_gamma, a_max]
    if params.dim == 2
        dx = s_target(1) - s_pursuer(1);
        dy = s_target(2) - s_pursuer(2);
        psi_des = atan2(dy, dx);
        err = mod(psi_des - s_pursuer(3) + pi, 2*pi) - pi;
        omega = max(-params.omega_max, min(params.omega_max, K_hdg * err));
        u = [omega, params.a_max];
    else
        dx = s_target(1) - s_pursuer(1);
        dy = s_target(2) - s_pursuer(2);
        dz = s_target(3) - s_pursuer(3);
        d_horiz = sqrt(dx^2 + dy^2);

        psi_des   = atan2(dy, dx);
        psi_err   = mod(psi_des - s_pursuer(4) + pi, 2*pi) - pi;
        gamma_des = atan2(dz, max(d_horiz, 1e-6));
        gamma_err = gamma_des - s_pursuer(5);

        u_psi   = max(-params.omega_max, min(params.omega_max, K_hdg * psi_err));
        u_gamma = max(-params.nu_max,    min(params.nu_max,    K_hdg * gamma_err));
        u = [u_psi, u_gamma, params.a_max];
    end
end


function u = crash_seeking_control(s_self, s_target, params)
%CRASH_SEEKING_CONTROL  Crash-seeking control to steer toward target.
%   Mode 'bang': bang-bang (max turn rate)
%   Mode 'prop': proportional (K_hdg * error, clamped)
    K_hdg = 2.5;  % proportional gain (matches waypoint_policy)
    use_prop = isfield(params, 'attc_type') && strcmp(params.attc_type, 'prop');

    if params.dim == 2
        dx = s_target(1) - s_self(1);
        dy = s_target(2) - s_self(2);
        psi_des = atan2(dy, dx);
        err = mod(psi_des - s_self(3) + pi, 2*pi) - pi;

        if use_prop
            omega = max(-params.omega_max, min(params.omega_max, K_hdg * err));
        else  % bang
            if abs(err) < 1e-10
                omega = 0;
            else
                omega = sign(err) * params.omega_max;
            end
        end
        u = [omega, params.a_max];
    else
        dx = s_target(1) - s_self(1);
        dy = s_target(2) - s_self(2);
        dz = s_target(3) - s_self(3);
        d_horiz = sqrt(dx^2 + dy^2);

        psi_des = atan2(dy, dx);
        psi_err = mod(psi_des - s_self(4) + pi, 2*pi) - pi;

        gamma_des = atan2(dz, max(d_horiz, 1e-6));
        gamma_err = gamma_des - s_self(5);

        if use_prop
            u_psi   = max(-params.omega_max, min(params.omega_max, K_hdg * psi_err));
            u_gamma = max(-params.nu_max,    min(params.nu_max,    K_hdg * gamma_err));
        else  % bang
            if abs(psi_err) < 1e-10
                u_psi = 0;
            else
                u_psi = sign(psi_err) * params.omega_max;
            end
            if abs(gamma_err) < 1e-10
                u_gamma = 0;
            else
                u_gamma = sign(gamma_err) * params.nu_max;
            end
        end
        u = [u_psi, u_gamma, params.a_max];
    end
end


function dx = dynamics_row(x, u, params)
%DYNAMICS_ROW  Row-vector dynamics for aTTC forward sim.
    if params.dim == 2
        v = x(4);  theta = x(3);
        dx = [v*cos(theta), v*sin(theta), u(1), u(2)];
    else
        v = x(6);  psi = x(4);  gamma = x(5);
        dx = [v*cos(gamma)*cos(psi), ...
              v*cos(gamma)*sin(psi), ...
              v*sin(gamma), ...
              u(1), u(2), u(3)];
    end
end
