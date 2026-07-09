function out = compute_formation_coherence_helpers()
%COMPUTE_FORMATION_COHERENCE_HELPERS  Returns a struct of function handles
%   so other scripts can call compute_formation_coherence,
%   compute_formation_coherence_norm, and compute_formation_coherence_per_follower
%   without duplicating the math.
%
%   Usage:
%     h = compute_formation_coherence_helpers();
%     dim_val      = h.coherence(R, info);          % scalar [km], mean across
%                                                   %   followers and time
%     norm_val     = h.coherence_norm(R, info);     % scalar in (0,1], higher
%                                                   %   better; exp(-coh/d_form)
%     per_foll_vec = h.coherence_per_follower(R,info); % vector [km] of length
%                                                   %   (n_eva - 1), one per
%                                                   %   follower; NaN if mission
%                                                   %   isn't formation*
%
%   `info` must have at minimum: info.evader_mask, info.dim.
%   `R.params` must have formation_spacing and formation_stagger_angle for
%   non-NaN results.
    out.coherence              = @compute_formation_coherence;
    out.coherence_norm         = @compute_formation_coherence_norm;
    out.coherence_per_follower = @compute_formation_coherence_per_follower;
    out.coherence_per_follower_procrustes = ...
        @compute_formation_coherence_per_follower_procrustes;
end


function val = compute_formation_coherence(R, info)
    miss = strtrim(char(R.params.mission));
    if ~any(strcmp(miss, {'formation', 'formation_pursuit'}))
        val = NaN;
        return;
    end
    if ~isfield(R.params, 'formation_spacing') || ...
       ~isfield(R.params, 'formation_stagger_angle') || ...
       ~isfield(R.params, 'evader_mask')
        val = NaN;
        return;
    end

    em = logical(info.evader_mask(:));
    evader_ids = find(em);
    if numel(evader_ids) < 2
        val = NaN;
        return;
    end
    leader_id    = evader_ids(1);
    follower_ids = evader_ids(2:end);
    n_foll = numel(follower_ids);

    pdim   = double(info.dim);
    d_form = double(R.params.formation_spacing);
    alpha  = double(R.params.formation_stagger_angle);
    n_steps = size(R.x, 1);

    ox_all = zeros(n_foll, 1);
    oy_all = zeros(n_foll, 1);
    for fi = 1:n_foll
        k_rk    = double(follower_ids(fi)) - 1;
        rnk     = ceil(k_rk / 2);
        sd_side = (-1)^k_rk;
        ox_all(fi) = -rnk * d_form * cos(alpha);
        oy_all(fi) =  sd_side * rnk * d_form * sin(alpha);
    end

    devs = zeros(n_steps, n_foll);
    for k = 1:n_steps
        pos_lead = squeeze(double(R.x(k, leader_id, 1:pdim)))';
        if pdim == 2
            psi_l = double(R.x(k, leader_id, 3));
            cps = cos(psi_l); sps = sin(psi_l);
            bx = [cps, sps];
            by = [-sps, cps];
        else
            psi_l = double(R.x(k, leader_id, 4));
            gam_l = double(R.x(k, leader_id, 5));
            cps = cos(psi_l); sps = sin(psi_l);
            cga = cos(gam_l); sga = sin(gam_l);
            bx = [cga*cps, cga*sps, sga];
            by = [-sps,    cps,     0];
        end
        for fi = 1:n_foll
            ag = follower_ids(fi);
            slot = pos_lead + ox_all(fi) * bx + oy_all(fi) * by;
            agent_pos = squeeze(double(R.x(k, ag, 1:pdim)))';
            devs(k, fi) = norm(agent_pos - slot);
        end
    end

    val = mean(mean(devs, 2));
end


function val = compute_formation_coherence_norm(R, info)
    val = compute_formation_coherence(R, info);
    if isnan(val), return; end
    if ~isfield(R.params, 'formation_spacing'), val = NaN; return; end
    d_ref = double(R.params.formation_spacing);
    if d_ref <= 0, val = NaN; return; end
    val = exp(-val / d_ref);
end


function vec = compute_formation_coherence_per_follower(R, info)
%   Per-follower mean slot deviation [km], time-averaged.  Returns NaN
%   (single value) if the mission isn't formation/formation_pursuit.
%   Otherwise returns a column vector of length (n_eva - 1) — one entry
%   per follower in evader_mask order, agents 2 .. n_eva.

    miss = strtrim(char(R.params.mission));
    if ~any(strcmp(miss, {'formation', 'formation_pursuit'}))
        vec = NaN;
        return;
    end
    if ~isfield(R.params, 'formation_spacing') || ...
       ~isfield(R.params, 'formation_stagger_angle') || ...
       ~isfield(R.params, 'evader_mask')
        vec = NaN;
        return;
    end

    em = logical(info.evader_mask(:));
    evader_ids = find(em);
    if numel(evader_ids) < 2
        vec = NaN;
        return;
    end
    leader_id    = evader_ids(1);
    follower_ids = evader_ids(2:end);
    n_foll = numel(follower_ids);

    pdim   = double(info.dim);
    d_form = double(R.params.formation_spacing);
    alpha  = double(R.params.formation_stagger_angle);
    n_steps = size(R.x, 1);

    ox_all = zeros(n_foll, 1);
    oy_all = zeros(n_foll, 1);
    for fi = 1:n_foll
        k_rk    = double(follower_ids(fi)) - 1;
        rnk     = ceil(k_rk / 2);
        sd_side = (-1)^k_rk;
        ox_all(fi) = -rnk * d_form * cos(alpha);
        oy_all(fi) =  sd_side * rnk * d_form * sin(alpha);
    end

    devs = zeros(n_steps, n_foll);
    for k = 1:n_steps
        pos_lead = squeeze(double(R.x(k, leader_id, 1:pdim)))';
        if pdim == 2
            psi_l = double(R.x(k, leader_id, 3));
            cps = cos(psi_l); sps = sin(psi_l);
            bx = [cps, sps];
            by = [-sps, cps];
        else
            psi_l = double(R.x(k, leader_id, 4));
            gam_l = double(R.x(k, leader_id, 5));
            cps = cos(psi_l); sps = sin(psi_l);
            cga = cos(gam_l); sga = sin(gam_l);
            bx = [cga*cps, cga*sps, sga];
            by = [-sps,    cps,     0];
        end
        for fi = 1:n_foll
            ag = follower_ids(fi);
            slot = pos_lead + ox_all(fi) * bx + oy_all(fi) * by;
            agent_pos = squeeze(double(R.x(k, ag, 1:pdim)))';
            devs(k, fi) = norm(agent_pos - slot);
        end
    end

    % Mean across time for each follower (no follower-averaging).
    vec = mean(devs, 1)';   % column vector, n_foll x 1
end


function vec = compute_formation_coherence_per_follower_procrustes(R, info)
%COMPUTE_FORMATION_COHERENCE_PER_FOLLOWER_PROCRUSTES
%   Per-follower formation residual [km], time-averaged, using leader-anchored
%   Procrustes alignment.  Mirrors compute_formation_coherence_per_follower
%   but replaces the leader's instantaneous body-frame rotation with the
%   rotation that BEST FITS the observed followers (Kabsch / orthogonal
%   Procrustes).  Rationale: if the formation as a whole lags or rotates
%   relative to the leader's instantaneous pose, the slot-based version
%   punishes the lag; the Procrustes version reports the residual that
%   remains after the best rigid rotation around the leader.
%
%   For each timestep:
%     1. Q_obs = followers' positions minus leader's position  (m x d)
%     2. Q_tpl = body-frame slot offsets, same for all timesteps  (m x d)
%     3. H = Q_tpl' * Q_obs;  [U,~,V] = svd(H);
%        R* = V * diag([1,...,1, sgn(det(V*U'))]) * U';       % SO(d)
%     4. residual_j = q_j - R* * s_j_body
%     5. d_j(t) = || residual_j ||
%   Return time-mean per follower.
%
%   Returns NaN (single value) for non-formation missions.

    miss = strtrim(char(R.params.mission));
    if ~any(strcmp(miss, {'formation', 'formation_pursuit'}))
        vec = NaN; return;
    end
    if ~isfield(R.params, 'formation_spacing') || ...
       ~isfield(R.params, 'formation_stagger_angle') || ...
       ~isfield(R.params, 'evader_mask')
        vec = NaN; return;
    end

    em = logical(info.evader_mask(:));
    evader_ids = find(em);
    if numel(evader_ids) < 2, vec = NaN; return; end
    leader_id    = evader_ids(1);
    follower_ids = evader_ids(2:end);
    n_foll = numel(follower_ids);

    pdim   = double(info.dim);
    d_form = double(R.params.formation_spacing);
    alpha  = double(R.params.formation_stagger_angle);
    n_steps = size(R.x, 1);

    % Body-frame template (m x pdim).  Convention A: agent 2 right wing
    % rank 1, agent 3 left wing rank 1, agent 4 right rank 2, ...  Slots
    % lie in the body x-y plane (z = 0 for 3D).
    Q_tpl = zeros(n_foll, pdim);
    for fi = 1:n_foll
        k_rk    = double(follower_ids(fi)) - 1;
        rnk     = ceil(k_rk / 2);
        sd_side = (-1)^k_rk;
        Q_tpl(fi, 1) = -rnk * d_form * cos(alpha);
        Q_tpl(fi, 2) =  sd_side * rnk * d_form * sin(alpha);
        % Q_tpl(fi, 3) already 0 in 3D
    end

    devs = zeros(n_steps, n_foll);
    for k = 1:n_steps
        pos_lead = squeeze(double(R.x(k, leader_id, 1:pdim)))';

        % Leader-anchored observed positions (m x pdim)
        Q_obs = zeros(n_foll, pdim);
        for fi = 1:n_foll
            ag = follower_ids(fi);
            agent_pos = squeeze(double(R.x(k, ag, 1:pdim)))';
            Q_obs(fi, :) = agent_pos - pos_lead;
        end

        % Kabsch: optimal rotation taking Q_tpl into Q_obs (no centroid
        % subtraction — leader is the anchor of both configurations).
        H = Q_tpl' * Q_obs;         % pdim x pdim
        [U, ~, V] = svd(H);
        D = eye(pdim);
        D(end, end) = sign(det(V * U'));
        R_star = V * D * U';        % R_star in SO(pdim)

        % Aligned per-follower residuals
        rot_tpl = (R_star * Q_tpl')';   % m x pdim
        diffs   = Q_obs - rot_tpl;
        devs(k, :) = sqrt(sum(diffs.^2, 2))';
    end

    vec = mean(devs, 1)';   % column vector, n_foll x 1
end
