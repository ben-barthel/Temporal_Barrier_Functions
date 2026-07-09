function u_safe = auspice_cbf(x_curr, u_nom, Sigma, params)
%AUSPICE_CBF  CBF safety filter via QP.
%
%   u_safe = auspice_cbf(x_curr, u_nom, Sigma, params)
%
%   Supports four formulations (params.cbf_type):
%
%   'hocbf' — Higher-Order CBF (relative degree 2)
%       h0 = ||p_i - p_j||^2 - (2*r_col)^2
%       Lf^2 h + Lg(Lf h) * u + (a1+a2)*dh0 + a1*a2*h0 >= 0
%
%   'attc_vraw' — aTTC CBF using the "raw velocities" FiLM NN.
%       h = aTTC_NN(x) - tau_min,  NN surrogate of adversarial TTC.
%       Input features are [delta_p; vel_i; vel_j] (9-dim in 3D), giving
%       the NN both individual velocities directly rather than only their
%       difference.  Scalar v_max_j FiLM conditioning.
%       Lg h * u >= -(Lf h + alpha * h)
%       The Lie derivatives use dh_dvi (ego-velocity gradient) for Lg
%       and dh_dvj (other-velocity gradient) for the other-agent drift
%       contribution to Lf — see attc_vraw_lie_derivs_3d below.
%
%   'sattc_vraw' — Stochastic aTTC CBF using the vraw NN.
%       Stochastic counterpart to 'attc_vraw'.  Because conditioning is
%       scalar v_max_j (constant per pair, doesn't depend on ego state),
%       the rank-1 H_features trick from attc_nn_forward_hessian carries
%       over directly.  J_feat is 9x6 with the vel_j rows zero (vel_j has
%       no dependence on ego state); H_curv contracts the analytical
%       d²(vel_i)/d(angles)² with dh_dvi.
%       B = exp(-k*h),  k = -log(1-p)/h_0
%       Lg h * u >= -(Lf h + Ls_h - k/2*||Sigma o dh||^2) + alpha/k - beta/(k*B)
%
%   'shocbf' — Stochastic Higher-Order CBF (relative degree 2)
%       psi = Lf h0 + alpha1 * h0   (synthesised first-order function)
%       B   = exp(-k*psi),  k = -log(1-p)/psi_0
%       Lg_psi * u >= -(Lf_psi + Ls_psi - k/2*||Sigma o dpsi||^2) + alpha/k - beta/(k*B)
%       Ito correction applied to psi; fully analytical (no NN required).
%
%   QP:  min ||u_ego - u_nom||^2
%        s.t.  above constraint  +  control bounds
    na     = params.num_agents;
    u_safe = u_nom;

    for ego = 1:na
        % In pursuit-style modes, only EVADERS get CBF protection;
        % pursuers fly uninhibited.
        %   - team_pursuit: params.evader_mask explicitly identifies evaders
        %   - plain pursuit: only agent 1 is the evader
        is_pursuer = false;
        if isfield(params, 'evader_mask')
            is_pursuer = ~params.evader_mask(ego);
        elseif params.pursuit_mode
            is_pursuer = (ego > 1);
        end
        if is_pursuer
            u_safe(ego,:) = u_nom(ego,:);
            continue;
        end

        A_ineq = [];
        b_ineq = [];

        for other = 1:na
            if other == ego, continue; end

            % Pair classification: is this an evader-pursuer (adversarial)
            % pair?  ee/pp pairs use the canonical attc_tau_min /
            % attc_activate_tau / r_cbf; ep pairs use the *_adv overrides
            % if the user supplied them (NaN sentinel = fall back to
            % canonical).  Computed first so the geometric activation gate
            % below can choose between cbf_activate_dist (ee) and
            % cbf_activate_dist_adv (ep) for asymmetric gating.
            is_adv_pair = false;
            if isfield(params, 'evader_mask') && ~isempty(params.evader_mask)
                is_adv_pair = (params.evader_mask(ego) ~= params.evader_mask(other));
            elseif params.pursuit_mode
                is_adv_pair = ((ego == 1) ~= (other == 1));   % plain pursuit
            end

            % Geometric activation gate (CLASS-AWARE).  For ep pairs, use
            % cbf_activate_dist_adv if it's a finite value; otherwise use
            % the canonical cbf_activate_dist for both classes.  Hybrid
            % cbf_type auto-defaults cbf_activate_dist_adv = Inf so the
            % ep half (attc_vraw) isn't blocked by the tighter ee gate.
            if params.dim == 2
                d_pair = norm(x_curr(ego,1:2) - x_curr(other,1:2));
            else
                d_pair = norm(x_curr(ego,1:3) - x_curr(other,1:3));
            end
            if is_adv_pair && isfield(params, 'cbf_activate_dist_adv') ...
                    && ~isnan(params.cbf_activate_dist_adv)
                d_gate = params.cbf_activate_dist_adv;
            else
                d_gate = params.cbf_activate_dist;
            end
            if d_pair > d_gate
                continue;   % pair is safely far apart — skip constraint
            end

            % Effective collision radius (inflated under chance constraint)
            if params.use_chance_constraint
                r_col = compute_r_eff(x_curr, ego, other, Sigma, params);
            else
                r_col = params.r_cbf;
            end

            % The aTTC overrides feed into the CBF branches via params_pair.
            % The r_cbf_adv override mutates the local r_col directly so it
            % takes effect for HOCBF (h = d^2 - (2*r_col)^2) and aTTC alike
            % (forward-sim collision threshold).
            params_pair = params;   % MATLAB struct copy is value semantics
            if is_adv_pair
                if isfield(params, 'attc_tau_min_adv') ...
                        && ~isnan(params.attc_tau_min_adv)
                    params_pair.attc_tau_min = params.attc_tau_min_adv;
                end
                if isfield(params, 'attc_activate_tau_adv') ...
                        && ~isnan(params.attc_activate_tau_adv)
                    params_pair.attc_activate_tau = params.attc_activate_tau_adv;
                end
                if isfield(params, 'r_cbf_adv') ...
                        && ~isnan(params.r_cbf_adv)
                    r_col = params.r_cbf_adv;
                end
                if isfield(params, 'cbf_alpha1_adv') ...
                        && ~isnan(params.cbf_alpha1_adv)
                    params_pair.cbf_alpha1 = params.cbf_alpha1_adv;
                end
                if isfield(params, 'attc_alpha_adv') ...
                        && ~isnan(params.attc_alpha_adv)
                    params_pair.attc_alpha = params.attc_alpha_adv;
                end
            end

            if strcmp(params.cbf_type, 'attc_vraw')
                % aTTC-CBF using the vraw FiLM NN (features = [dp; vel_i; vel_j]).
                if params.dim == 2
                    [Lg, Lf, h_val] = attc_vraw_lie_derivs_2d( ...
                        x_curr, ego, other, u_nom(other,:), r_col, params_pair);
                else
                    [Lg, Lf, h_val] = attc_vraw_lie_derivs_3d( ...
                        x_curr, ego, other, u_nom(other,:), r_col, params_pair);
                end
                if h_val > params_pair.attc_activate_tau - params_pair.attc_tau_min
                    continue;
                end
                if norm(Lg) < 1e-10
                    continue;
                end
                rhs = -(Lf + params_pair.attc_alpha * h_val);
            elseif strcmp(params.cbf_type, 'sattc_vraw')
                % Stochastic aTTC-CBF using vraw NN (3D only).
                assert(params.dim == 3, 'sattc_vraw-CBF only implemented for 3D');
                [Lg, ~, h_val, rhs] = attc_vraw_sattc_lie_derivs_3d( ...
                    x_curr, ego, other, u_nom(other,:)', r_col, params_pair);
                if h_val > params_pair.attc_activate_tau - params_pair.attc_tau_min
                    continue;
                end
                if norm(Lg) < 1e-10
                    continue;
                end
                % rhs already fully computed in attc_vraw_sattc_lie_derivs_3d
            elseif strcmp(params.cbf_type, 'shocbf')
                % Stochastic HOCBF (3D only)
                assert(params.dim == 3, 'sHOCBF only implemented for 3D');
                [Lg, ~, ~, rhs] = hocbf_stochastic_lie_derivs_3d( ...
                    x_curr, ego, other, u_nom(other,:)', r_col, params_pair);
                if norm(Lg) < 1e-10
                    continue;
                end
                % rhs already fully computed in hocbf_stochastic_lie_derivs_3d
            elseif strcmp(params.cbf_type, 'hocbf')
                % HOCBF (relative degree 2) — must be requested EXPLICITLY.
                if params.dim == 2
                    [LgLfh, Lf2h, dh0, h0] = lie_derivs_2d( ...
                        x_curr, ego, other, u_nom(other,:), r_col);
                else
                    [LgLfh, Lf2h, dh0, h0] = lie_derivs_3d( ...
                        x_curr, ego, other, u_nom(other,:), r_col);
                end
                a1  = params_pair.cbf_alpha1;
                a2  = params.cbf_alpha2;
                Lg  = LgLfh;
                rhs = -(Lf2h + (a1+a2)*dh0 + a1*a2*h0);
            else
                % Hard fail on unknown cbf_type rather than silently falling
                % back to HOCBF.  A typo in cbf_type used to slip through
                % here and quietly produce HOCBF results that the caller
                % mistook for the type they intended.
                error('auspice_cbf:unknownCbfType', ...
                    ['Unknown cbf_type ''%s'' inside auspice_cbf dispatch.\n' ...
                     '  Valid types: hocbf, shocbf, attc_vraw, sattc_vraw.'], ...
                    params.cbf_type);
            end

            % Constraint: Lg * u >= rhs  -->  -Lg * u <= -rhs
            % For HOCBF: Lg = Lg(Lf h0), the control-affine part of h0_ddot
            A_ineq = [A_ineq; -Lg];     %#ok<AGROW>
            b_ineq = [b_ineq; -rhs];    %#ok<AGROW>
        end

        if isempty(A_ineq), continue; end

        % Control bounds
        if params.dim == 2
            lb = [-params.omega_max; -params.a_max];
            ub = [ params.omega_max;  params.a_max];
        else
            lb = [-params.omega_max; -params.nu_max; -params.a_max];
            ub = [ params.omega_max;  params.nu_max;  params.a_max];
        end

        % Solve QP
        u_ego_nom = u_nom(ego,:)';
        if params.has_quadprog
            H    = eye(params.ctrl_dim);
            f    = -u_ego_nom;
            opts = optimset('Display','off');
            [u_opt, ~, exitflag] = quadprog(H, f, A_ineq, b_ineq, ...
                                            [], [], lb, ub, u_ego_nom, opts);
            if exitflag > 0
                u_safe(ego,:) = u_opt';
            else
                u_safe(ego,:) = simple_qp_solve(u_ego_nom, A_ineq, b_ineq, lb, ub)';
            end
        else
            u_safe(ego,:) = simple_qp_solve(u_ego_nom, A_ineq, b_ineq, lb, ub)';
        end
    end
end


%% ==================== Stochastic HOCBF Lie Derivatives (3D) ====================

function [Lg_psi, Lf_psi, psi_val, rhs] = hocbf_stochastic_lie_derivs_3d(x_curr, ego, other, u_oth, r_col, params)
% Stochastic HOCBF Lie derivatives (3D Dubins).
%
%   Defines the synthesised first-order function:
%       psi = Lf h0 + alpha1 * h0  =  2*(dp.dv) + alpha1*(||dp||^2 - (2r)^2)
%
%   Applies the stochastic CBF to psi:
%       B = exp(-k * psi),  k = params.scbf_k(ego, other)
%       Lg_psi * u >= -(Lf_psi + Ls_psi - k/2*||Sigma o dpsi/dx||^2) + alpha/k - beta/(k*B)
%
%   The Ito correction Ls_psi and the gradient norm are computed analytically
%   since h0 = ||dp||^2 - (2r)^2 and its Lie derivatives are polynomial.
%
%   Returns:
%   Lg_psi  : (1x3) control-affine part  [= Lg(Lf h0), same as HOCBF]
%   Lf_psi  : scalar drift  [= Lf^2 h0 + alpha1 * Lf h0]
%   psi_val : scalar  [= Lf h0 + alpha1 * h0]
%   rhs     : scalar SCBF constraint right-hand side

    xi = x_curr(ego,:);    xj = x_curr(other,:);
    pxi = xi(1); pyi = xi(2); pzi = xi(3);
    psii = xi(4); gami = xi(5); vi = xi(6);
    pxj = xj(1); pyj = xj(2); pzj = xj(3);
    psij = xj(4); gamj = xj(5); vj = xj(6);
    u_psij = u_oth(1);  u_gamj = u_oth(2);  aj = u_oth(3);

    a1 = params.cbf_alpha1;
    a2 = params.cbf_alpha2;

    % Relative position and velocity
    dpx = pxi - pxj;  dpy = pyi - pyj;  dpz = pzi - pzj;
    dp  = [dpx; dpy; dpz];

    vxi = vi*cos(gami)*cos(psii);  vyi = vi*cos(gami)*sin(psii);  vzi = vi*sin(gami);
    vxj = vj*cos(gamj)*cos(psij);  vyj = vj*cos(gamj)*sin(psij);  vzj = vj*sin(gamj);
    dvx = vxi - vxj;  dvy = vyi - vyj;  dvz = vzi - vzj;
    dv  = [dvx; dvy; dvz];

    vel_i = [vxi; vyi; vzi];

    % h0 and its Lie derivatives
    h0  = dpx^2 + dpy^2 + dpz^2 - (2*r_col)^2;
    dh0 = 2*(dpx*dvx + dpy*dvy + dpz*dvz);   % = Lf h0

    % Synthesised function
    psi_val = dh0 + a1 * h0;

    % Drift acceleration from other agent
    d_dvx_oth = -(vj*(-sin(gamj)*u_gamj*cos(psij) - cos(gamj)*sin(psij)*u_psij) ...
                  + aj*cos(gamj)*cos(psij));
    d_dvy_oth = -(vj*(-sin(gamj)*u_gamj*sin(psij) + cos(gamj)*cos(psij)*u_psij) ...
                  + aj*cos(gamj)*sin(psij));
    d_dvz_oth = -(vj*cos(gamj)*u_gamj + aj*sin(gamj));

    Lf2h = 2*(dvx^2 + dvy^2 + dvz^2) + ...
           2*(dpx*d_dvx_oth + dpy*d_dvy_oth + dpz*d_dvz_oth);

    % Lf psi = Lf^2 h0 + alpha1 * Lf h0
    Lf_psi = Lf2h + a1 * dh0;

    % Control-affine part: Lg psi = Lg(Lf h0)  (same as HOCBF)
    Lg_psi = [2*(dpx*(-vi*cos(gami)*sin(psii)) + dpy*(vi*cos(gami)*cos(psii))), ...
              2*(dpx*(-vi*sin(gami)*cos(psii)) + dpy*(-vi*sin(gami)*sin(psii)) ...
                + dpz*(vi*cos(gami))), ...
              2*(dpx*cos(gami)*cos(psii) + dpy*cos(gami)*sin(psii) + dpz*sin(gami))];

    % ============================================================
    %  Gradient of psi w.r.t. ego state x_ego = [x,y,z,psi,gam,v]
    %  psi = 2*(dp.dv) + a1*(||dp||^2 - (2r)^2)
    %
    %  d(dp)/d(pos_ego) = I3,   d(dp)/d(angles,v) = 0
    %  d(dv)/d(pos_ego) = 0,    d(dv)/d(psi,gam,v) = J_vel_ego
    % ============================================================

    J_vel = [-vi*cos(gami)*sin(psii),  -vi*sin(gami)*cos(psii),  cos(gami)*cos(psii);
              vi*cos(gami)*cos(psii),  -vi*sin(gami)*sin(psii),  cos(gami)*sin(psii);
              0,                         vi*cos(gami),             sin(gami)];

    % dpsi/d(pos_ego) = 2*dv + 2*a1*dp
    dpsi_dpos = 2*dv + 2*a1*dp;

    % dpsi/d(angles,v) = 2 * J_vel' * dp
    dpsi_dang = 2 * J_vel' * dp;

    dpsi_dx = [dpsi_dpos; dpsi_dang];   % (6x1)

    % ============================================================
    %  Hessian diagonal of psi w.r.t. ego state (6 entries)
    %  d^2 psi / d(pos)^2 diagonal = 2*a1 (identity block)
    %  d^2 psi / d(angles,v)^2 diagonal:
    %    = 2 * dp . d^2(vel_i)/d(angle_k)^2  for each angle/speed k
    % ============================================================

    % Second derivatives of vel_i components w.r.t. (psi, gam, v):
    d2vx_dpsi2 = -vi*cos(gami)*cos(psii);
    d2vy_dpsi2 = -vi*cos(gami)*sin(psii);
    d2vz_dpsi2 =  0;

    d2vx_dgam2 = -vi*cos(gami)*cos(psii);
    d2vy_dgam2 = -vi*cos(gami)*sin(psii);
    d2vz_dgam2 = -vi*sin(gami);

    % d^2 vel / dv^2 = 0 for all components

    d2psi_dpsi2 = 2*(dpx*d2vx_dpsi2 + dpy*d2vy_dpsi2 + dpz*d2vz_dpsi2);
    d2psi_dgam2 = 2*(dpx*d2vx_dgam2 + dpy*d2vy_dgam2 + dpz*d2vz_dgam2);
    d2psi_dv2   = 0;

    hess_diag = [2*a1; 2*a1; 2*a1; d2psi_dpsi2; d2psi_dgam2; d2psi_dv2];

    % ============================================================
    %  Ito diffusion term:  Ls_psi = 1/2 * sum(sigma_i^2 * hess_diag_i)
    % ============================================================
    sigma2  = params.noise_std(:) .^ 2;   % (6x1)
    Ls_psi  = 0.5 * sum(sigma2 .* hess_diag);

    % ||Sigma o dpsi/dx||^2
    norm_sq = sum(sigma2 .* dpsi_dx .^ 2);

    % ============================================================
    %  Full SCBF constraint RHS
    %  Dividing LgB*u + LfB + LsB <= -alpha*B + beta  by -k*B:
    %    Lg_psi * u >= -(Lf_psi + Ls_psi - k/2*norm_sq) + alpha/k - beta/(k*B)
    % ============================================================
    k   = params.scbf_k(ego, other);
    B   = exp(-k * psi_val);
    rhs = -(Lf_psi + Ls_psi - 0.5*k*norm_sq) ...
          + params.scbf_alpha / k ...
          - params.scbf_beta  / (k * B);
end


%% ==================== HOCBF Lie Derivatives ====================

function [LgLfh, Lf2h, dh0, h0] = lie_derivs_2d(x_curr, ego, other, u_oth, r_col)
% 2D Dubins Lie derivatives for HOCBF.
%   LgLfh = Lg(Lf h0): control-affine part of the second derivative of h0
%   Lf2h  = Lf^2 h0:   drift part of the second derivative of h0
%   dh0   = Lf h0:     first time derivative of h0 (pure drift, since Lg h0 = 0)
%   h0    = barrier value
%   r_col = effective collision radius (nominal r, or inflated for chance constraint)
    xi = x_curr(ego,:);    xj = x_curr(other,:);
    pxi = xi(1); pyi = xi(2); thi = xi(3); vi = xi(4);
    pxj = xj(1); pyj = xj(2); thj = xj(3); vj = xj(4);
    omj = u_oth(1);  aj = u_oth(2);

    dpx = pxi - pxj;
    dpy = pyi - pyj;
    dvx = vi*cos(thi) - vj*cos(thj);
    dvy = vi*sin(thi) - vj*sin(thj);

    % h0, dh0
    h0  = dpx^2 + dpy^2 - (2*r_col)^2;
    dh0 = 2*(dpx*dvx + dpy*dvy);

    % Drift acceleration from other agent
    d_dvx_oth = -(aj*cos(thj) - vj*omj*sin(thj));
    d_dvy_oth = -(aj*sin(thj) + vj*omj*cos(thj));

    Lf2h = 2*(dvx^2 + dvy^2) + 2*(dpx*d_dvx_oth + dpy*d_dvy_oth);

    % Control-affine terms for ego  u = [omega_i, a_i]
    LgLfh = [2*(dpx*(-vi*sin(thi)) + dpy*(vi*cos(thi))), ...
             2*(dpx*cos(thi)        + dpy*sin(thi))];
end

%% Lie Derivatives 3D
function [LgLfh, Lf2h, dh0, h0] = lie_derivs_3d(x_curr, ego, other, u_oth, r_col)
% 3D Dubins Lie derivatives for HOCBF.
%   LgLfh = Lg(Lf h0): control-affine part of the second derivative of h0
%   Lf2h  = Lf^2 h0:   drift part of the second derivative of h0
%   dh0   = Lf h0:     first time derivative of h0 (pure drift, since Lg h0 = 0)
%   h0    = barrier value
%   r_col = effective collision radius (nominal r, or inflated for chance constraint)
    xi = x_curr(ego,:);    xj = x_curr(other,:);
    pxi = xi(1); pyi = xi(2); pzi = xi(3);
    psii = xi(4); gami = xi(5); vi = xi(6);
    pxj = xj(1); pyj = xj(2); pzj = xj(3);
    psij = xj(4); gamj = xj(5); vj = xj(6);
    u_psij = u_oth(1);  u_gamj = u_oth(2);  aj = u_oth(3);

    dpx = pxi - pxj;  dpy = pyi - pyj;  dpz = pzi - pzj;
    dvx = vi*cos(gami)*cos(psii) - vj*cos(gamj)*cos(psij);
    dvy = vi*cos(gami)*sin(psii) - vj*cos(gamj)*sin(psij);
    dvz = vi*sin(gami)           - vj*sin(gamj);

    h0  = dpx^2 + dpy^2 + dpz^2 - (2*r_col)^2;
    dh0 = 2*(dpx*dvx + dpy*dvy + dpz*dvz);

    % Drift acceleration from other agent (heading rates + acceleration)
    d_dvx_oth = -(vj*(-sin(gamj)*u_gamj*cos(psij) - cos(gamj)*sin(psij)*u_psij) ...
                  + aj*cos(gamj)*cos(psij));
    d_dvy_oth = -(vj*(-sin(gamj)*u_gamj*sin(psij) + cos(gamj)*cos(psij)*u_psij) ...
                  + aj*cos(gamj)*sin(psij));
    d_dvz_oth = -(vj*cos(gamj)*u_gamj + aj*sin(gamj));

    Lf2h = 2*(dvx^2 + dvy^2 + dvz^2) + ...
           2*(dpx*d_dvx_oth + dpy*d_dvy_oth + dpz*d_dvz_oth);

    % Control-affine terms for ego  u = [u_psi_i, u_gamma_i, a_i]
    LgLfh = [2*(dpx*(-vi*cos(gami)*sin(psii)) + dpy*(vi*cos(gami)*cos(psii))), ...
             2*(dpx*(-vi*sin(gami)*cos(psii)) + dpy*(-vi*sin(gami)*sin(psii)) ...
               + dpz*(vi*cos(gami))), ...
             2*(dpx*cos(gami)*cos(psii) + dpy*cos(gami)*sin(psii) + dpz*sin(gami))];
end


%% ==================== aTTC-CBF: NN Forward (delegated) ====================
% The forward + backward pass `attc_nn_forward(...)` lives in the standalone
% file attc_nn_forward.m (ditto for the cond2 / hessian variants).  When the
% Lie-derivative helpers below call attc_nn_forward / attc_nn_forward_*,
% MATLAB resolves to those standalone files (since this function file no
% longer defines a local copy).  Single source of truth — including for the
% optional inference-only SoftCap (params.attc_inference_cap).


%% ==================== aTTC-vraw CBF Lie Derivatives (3D) ====================

function [Lg_h, Lf_h, h_val] = attc_vraw_lie_derivs_3d(x_curr, ego, other, u_oth, r_col, params)
% 3D aTTC-CBF Lie derivatives using the VRAW NN surrogate.
%
%   Features fed to the NN: [delta_p; vel_i; vel_j]   (9-dim in 3D)
%   Conditioning: scalar v_max_j (same as cond1).
%
%   Differs from attc_lie_derivs_3d (which uses delta_v) in that the
%   gradient w.r.t. the j-velocity feature, dh_dvj, is INDEPENDENT of the
%   gradient w.r.t. the i-velocity, dh_dvi.  In the delta_v formulation,
%   d(attc)/d(vel_i) = +dh_dv  and  d(attc)/d(vel_j) = -dh_dv  are
%   *constrained* to be negatives of each other by the chain rule
%   (since dv = vel_i - vel_j).  vraw lifts that constraint — the NN
%   learns arbitrary per-velocity sensitivities.
%
%   Consequences for the Lie derivatives (compared to attc_lie_derivs_3d):
%     - Lg_h uses dh_dvi (NOT dh_dv) — same algebraic form, different gradient
%     - Lf_h uses dh_dvj (NOT dh_dv) for the other-agent drift channel,
%       and the sign flip on `dnu_dpsi_j` etc. that was needed in the
%       delta_v formulation goes away because we use the direct
%       d(vel_j)/d(state_j) derivatives here.
%
%   Returns:
%   Lg_h  : (1x3) row vector [dh/du_psi, dh/du_gamma, dh/da] for ego
%   Lf_h  : scalar drift Lie derivative
%   h_val : scalar barrier value (aTTC_NN - tau_min)

    xi = x_curr(ego,:);    xj = x_curr(other,:);
    psi_i = xi(4); gam_i = xi(5); v_i = xi(6);
    psi_j = xj(4); gam_j = xj(5); v_j = xj(6);
    u_psi_j = u_oth(1);  u_gam_j = u_oth(2);  a_j = u_oth(3);

    % Relative position and INDIVIDUAL velocities (i-j convention for dp)
    dp = xi(1:3)' - xj(1:3)';

    vx_i = v_i*cos(gam_i)*cos(psi_i);
    vy_i = v_i*cos(gam_i)*sin(psi_i);
    vz_i = v_i*sin(gam_i);
    vx_j = v_j*cos(gam_j)*cos(psi_j);
    vy_j = v_j*cos(gam_j)*sin(psi_j);
    vz_j = v_j*sin(gam_j);
    vel_i = [vx_i; vy_i; vz_i];
    vel_j = [vx_j; vy_j; vz_j];

    % vraw features: [dp; vel_i; vel_j]  (9-dim)
    features_nn = [dp; vel_i; vel_j];

    % Conditioning: scalar v_max_j (same as cond1)
    vm = params.v_max(:);
    v_max_other = vm(min(other, length(vm)));

    [attc_val, grad_nn] = attc_nn_forward(features_nn, params.attc_nn, ...
                                           params.attc_grad_clip, v_max_other);

    dh_dp  = grad_nn(1:3);   % d(attc)/d(dp)
    dh_dvi = grad_nn(4:6);   % d(attc)/d(vel_i)   (independent gradient)
    dh_dvj = grad_nn(7:9);   % d(attc)/d(vel_j)   (independent gradient)

    h_val = attc_val - params.attc_tau_min;

    % --- Velocity Jacobians (DIRECT, positive sign — vraw uses individual vels) ---
    % d(vel_i)/d(psi_i, gam_i, v_i)
    dvi_dpsi_i = [-v_i*cos(gam_i)*sin(psi_i);  v_i*cos(gam_i)*cos(psi_i); 0];
    dvi_dgam_i = [-v_i*sin(gam_i)*cos(psi_i); -v_i*sin(gam_i)*sin(psi_i); v_i*cos(gam_i)];
    dvi_dv_i   = [ cos(gam_i)*cos(psi_i);      cos(gam_i)*sin(psi_i);     sin(gam_i)];

    % d(vel_j)/d(psi_j, gam_j, v_j)  — note POSITIVE sign (no flip)
    dvj_dpsi_j = [-v_j*cos(gam_j)*sin(psi_j);  v_j*cos(gam_j)*cos(psi_j); 0];
    dvj_dgam_j = [-v_j*sin(gam_j)*cos(psi_j); -v_j*sin(gam_j)*sin(psi_j); v_j*cos(gam_j)];
    dvj_dv_j   = [ cos(gam_j)*cos(psi_j);      cos(gam_j)*sin(psi_j);     sin(gam_j)];

    % --- Lg(h): control-affine part for ego ---
    %   vel_j has zero gradient w.r.t. ego state, so only dh_dvi contributes.
    Lg_h = [dh_dvi' * dvi_dpsi_i, ...
            dh_dvi' * dvi_dgam_i, ...
            dh_dvi' * dvi_dv_i];

    % --- Lf(h): drift part ---
    %   Position drift: dh_dp · (vel_i - vel_j).
    Lf_pos = dh_dp' * (vel_i - vel_j);

    %   Other agent's controls (treated as known disturbance) act on vel_j.
    %   Note: NO sign flip — dh_dvj * d(vel_j)/d(state_j) is the direct
    %   contribution.  In the delta_v formulation we used `-dh_dv` here
    %   implicitly via the negated `dnu_dpsi_j` definition.
    Lf_other_ctrl = dh_dvj' * dvj_dpsi_j * u_psi_j ...
                  + dh_dvj' * dvj_dgam_j * u_gam_j ...
                  + dh_dvj' * dvj_dv_j   * a_j;

    Lf_h = Lf_pos + Lf_other_ctrl;
end


%% ==================== aTTC-vraw CBF Lie Derivatives (2D) ====================

function [Lg_h, Lf_h, h_val] = attc_vraw_lie_derivs_2d(x_curr, ego, other, u_oth, r_col, params)
% 2D aTTC-CBF Lie derivatives using the VRAW NN surrogate.
%   State: [px, py, theta, v], Control: [omega, a].
%   Features fed to the NN: [delta_p; vel_i; vel_j]  (6-dim in 2D).
%
%   Returns:
%   Lg_h  : (1x2) row vector [dh/d_omega, dh/d_a] for ego
%   Lf_h  : scalar drift
%   h_val : scalar barrier value (aTTC_NN - tau_min)

    xi = x_curr(ego,:);    xj = x_curr(other,:);
    th_i = xi(3); v_i = xi(4);
    th_j = xj(3); v_j = xj(4);
    om_j = u_oth(1);  a_j = u_oth(2);

    dp = xi(1:2)' - xj(1:2)';
    vx_i = v_i*cos(th_i);  vy_i = v_i*sin(th_i);
    vx_j = v_j*cos(th_j);  vy_j = v_j*sin(th_j);
    vel_i = [vx_i; vy_i];
    vel_j = [vx_j; vy_j];

    features_nn = [dp; vel_i; vel_j];   % 6-dim

    vm = params.v_max(:);
    v_max_other = vm(min(other, length(vm)));

    [attc_val, grad_nn] = attc_nn_forward(features_nn, params.attc_nn, ...
                                           params.attc_grad_clip, v_max_other);

    dh_dp  = grad_nn(1:2);
    dh_dvi = grad_nn(3:4);
    dh_dvj = grad_nn(5:6);

    h_val = attc_val - params.attc_tau_min;

    % --- Lg(h) ---
    dvi_dth_i = [-v_i*sin(th_i); v_i*cos(th_i)];     % d(vel_i)/d(theta_i)
    dvi_dv_i  = [ cos(th_i);     sin(th_i)];         % d(vel_i)/d(v_i)

    Lg_h = [dh_dvi' * dvi_dth_i, ...
            dh_dvi' * dvi_dv_i];

    % --- Lf(h) ---
    Lf_pos = dh_dp' * (vel_i - vel_j);

    dvj_dth_j = [-v_j*sin(th_j); v_j*cos(th_j)];
    dvj_dv_j  = [ cos(th_j);     sin(th_j)];

    Lf_other_ctrl = dh_dvj' * dvj_dth_j * om_j ...
                  + dh_dvj' * dvj_dv_j  * a_j;

    Lf_h = Lf_pos + Lf_other_ctrl;
end


%% ==================== Stochastic aTTC-vraw Lie Derivatives (3D) ====================

function [Lg_h, Lf_h, h_val, rhs] = attc_vraw_sattc_lie_derivs_3d(x_curr, ego, other, u_oth, r_col, params)
% 3D stochastic aTTC-CBF Lie derivatives using the VRAW NN surrogate.
%
%   Mirrors attc_sattc_lie_derivs_3d but with vraw features
%   [delta_p; vel_i; vel_j] (9-dim).  Conditioning is still scalar
%   v_max_j and does NOT depend on ego state, so the rank-1
%   H_features trick from attc_nn_forward_hessian works directly —
%   no FD machinery needed (unlike sattc_c2).
%
%   Two structural differences from the delta_v-based sattc:
%     (1) Lg_h and the H_curv block use dh_dvi (ego-velocity gradient)
%         instead of dh_dv.  No sign flip on the i-side.
%     (2) Lf_h's other-agent-control disturbance uses dh_dvj directly
%         (no sign flip from the implicit chain through delta_v).
%     (3) J_feat is 9x6 with the bottom 3 rows (vel_j) zero, since
%         vel_j has no dependence on ego state.
%
%   Returns:
%   Lg_h   : (1x3) control-affine row vector for ego
%   Lf_h   : scalar drift
%   h_val  : barrier value (aTTC_NN - tau_min)
%   rhs    : full SCBF constraint RHS

    xi = x_curr(ego,:);    xj = x_curr(other,:);
    psi_i = xi(4); gam_i = xi(5); v_i = xi(6);
    psi_j = xj(4); gam_j = xj(5); v_j = xj(6);
    u_psi_j = u_oth(1);  u_gam_j = u_oth(2);  a_j = u_oth(3);

    % Relative position and INDIVIDUAL velocities
    dp = xi(1:3)' - xj(1:3)';

    vx_i = v_i*cos(gam_i)*cos(psi_i);
    vy_i = v_i*cos(gam_i)*sin(psi_i);
    vz_i = v_i*sin(gam_i);
    vx_j = v_j*cos(gam_j)*cos(psi_j);
    vy_j = v_j*cos(gam_j)*sin(psi_j);
    vz_j = v_j*sin(gam_j);
    vel_i = [vx_i; vy_i; vz_i];
    vel_j = [vx_j; vy_j; vz_j];

    % vraw features: [dp; vel_i; vel_j]  (9-dim)
    features_nn = [dp; vel_i; vel_j];
    vm = params.v_max(:);
    v_max_other = vm(min(other, length(vm)));

    % NN forward + grad + 9x9 feature-space Hessian (rank-1, fixed cond)
    [attc_val, grad_nn, H_features] = attc_nn_forward_hessian( ...
        features_nn, params.attc_nn, params.attc_grad_clip, v_max_other);

    dh_dp  = grad_nn(1:3);   % d(attc)/d(dp)
    dh_dvi = grad_nn(4:6);   % d(attc)/d(vel_i)
    dh_dvj = grad_nn(7:9);   % d(attc)/d(vel_j)

    h_val = attc_val - params.attc_tau_min;

    % --- Velocity Jacobians (direct, positive sign) ---
    J_vel_i = [-v_i*cos(gam_i)*sin(psi_i),  -v_i*sin(gam_i)*cos(psi_i),  cos(gam_i)*cos(psi_i);
                v_i*cos(gam_i)*cos(psi_i),  -v_i*sin(gam_i)*sin(psi_i),  cos(gam_i)*sin(psi_i);
                0,                            v_i*cos(gam_i),              sin(gam_i)];

    dvi_dpsi_i = J_vel_i(:,1);
    dvi_dgam_i = J_vel_i(:,2);
    dvi_dv_i   = J_vel_i(:,3);

    dvj_dpsi_j = [-v_j*cos(gam_j)*sin(psi_j);  v_j*cos(gam_j)*cos(psi_j); 0];
    dvj_dgam_j = [-v_j*sin(gam_j)*cos(psi_j); -v_j*sin(gam_j)*sin(psi_j); v_j*cos(gam_j)];
    dvj_dv_j   = [ cos(gam_j)*cos(psi_j);      cos(gam_j)*sin(psi_j);     sin(gam_j)];

    % --- Lg(h): only dh_dvi contributes (vel_j has no ego-state gradient) ---
    Lg_h = [dh_dvi' * dvi_dpsi_i, ...
            dh_dvi' * dvi_dgam_i, ...
            dh_dvi' * dvi_dv_i];

    % --- Lf(h): position drift + other-agent-control disturbance via dh_dvj ---
    Lf_pos = dh_dp' * (vel_i - vel_j);

    Lf_other_ctrl = dh_dvj' * dvj_dpsi_j * u_psi_j ...
                  + dh_dvj' * dvj_dgam_j * u_gam_j ...
                  + dh_dvj' * dvj_dv_j   * a_j;

    Lf_h = Lf_pos + Lf_other_ctrl;

    % ============================================================
    %  Stochastic diffusion term: Ls_h = 1/2 Tr(Σ² · H_full)
    %  Build the 6x6 Hessian of h w.r.t. ego state.
    % ============================================================

    % Feature Jacobian J_feat (9x6) for ego state [x,y,z,psi,gam,v]:
    %   - dp rows  : [I_3, 0_3]
    %   - vel_i rows : [0_3, J_vel_i]
    %   - vel_j rows : [0_3, 0_3]   ← vel_j doesn't depend on ego state
    J_feat = [eye(3),     zeros(3,3);
              zeros(3,3), J_vel_i;
              zeros(3,3), zeros(3,3)];

    % NN-curvature term: J_feat' * H_features * J_feat   (9x9 → 6x6)
    H_nn_term = J_feat' * H_features * J_feat;

    % Analytical curvature of vel_i w.r.t. (psi_i, gam_i, v_i), contracted
    % with dh_dvi (positions 4-6 of grad_raw, NOT dh_dv as in sattc).
    H_vx = [-v_i*cos(gam_i)*cos(psi_i),  v_i*sin(gam_i)*sin(psi_i), -cos(gam_i)*sin(psi_i);
              v_i*sin(gam_i)*sin(psi_i), -v_i*cos(gam_i)*cos(psi_i), -sin(gam_i)*cos(psi_i);
             -cos(gam_i)*sin(psi_i),     -sin(gam_i)*cos(psi_i),       0];
    H_vy = [-v_i*cos(gam_i)*sin(psi_i), -v_i*sin(gam_i)*cos(psi_i),  cos(gam_i)*cos(psi_i);
             -v_i*sin(gam_i)*cos(psi_i), -v_i*cos(gam_i)*sin(psi_i), -sin(gam_i)*sin(psi_i);
              cos(gam_i)*cos(psi_i),     -sin(gam_i)*sin(psi_i),       0];
    H_vz = [0,      0,             0;
             0, -v_i*sin(gam_i),  cos(gam_i);
             0,   cos(gam_i),     0];
    curv_block = dh_dvi(1)*H_vx + dh_dvi(2)*H_vy + dh_dvi(3)*H_vz;
    H_curv = zeros(6, 6);
    H_curv(4:6, 4:6) = curv_block;

    % Note: vel_j features have ZERO ego-state derivatives, so they
    % contribute NOTHING to the ego-state Hessian.  No cross / cond
    % terms either (cond is constant).

    H_full = H_nn_term + H_curv;

    % --- Itô diffusion ---
    sigma  = params.noise_std(:);
    sigma2 = sigma .^ 2;
    Ls_h_trace = 0.5 * sum(sigma2 .* diag(H_full));

    % Full ego-state gradient: dh/dx = J_feat' · grad_raw  (6x1)
    dh_dx_ego = J_feat' * grad_nn;
    norm_sq   = sum(sigma2 .* dh_dx_ego .^ 2);

    % --- SCBF constraint RHS ---
    k = params.scbf_k(ego, other);
    B = exp(-k * h_val);
    rhs = -(Lf_h + Ls_h_trace - 0.5*k*norm_sq) ...
          + params.scbf_alpha / k ...
          - params.scbf_beta  / (k * B);
end




%% ==================== Fallback QP solver ====================

function u = simple_qp_solve(u_nom, A, b, lb, ub)
% Projected-gradient fallback when quadprog is unavailable.
% Handles multiple linear inequality constraints with box bounds.
    u = max(lb, min(ub, u_nom));       % clip to box

    max_iter = 50;
    for iter = 1:max_iter
        violated = A*u - b;
        [worst, idx] = max(violated);
        if worst <= 1e-8
            break;                      % all constraints satisfied
        end
        % Project onto the most-violated constraint hyperplane
        ai = A(idx,:)';
        u  = u - (violated(idx) / (ai'*ai)) * ai;
        u  = max(lb, min(ub, u));       % re-clip
    end
end
