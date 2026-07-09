function results = auspice_sim(varargin)
%AUSPICE_SIM  Multi-agent Dubins simulation with TTC and CBF safety filter.
%
%   results = AUSPICE_SIM() runs a default 2-agent 2D scenario.
%
%   results = AUSPICE_SIM('Name', Value, ...) with options:
%       'dim'          - 2 or 3                        (default: 2)
%       'cbf_type'     - CBF type: 'none' | 'hocbf' | 'shocbf' | 'attc_vraw' | 'sattc_vraw'
%                        (default: 'none')
%                          'none'       no safety filter (nominal control only)
%                          'hocbf'      higher-order distance-based CBF baseline
%                          'shocbf'     stochastic HOCBF (chance-constrained)
%                          'attc_vraw'  aTTC-CBF: neural-network surrogate of the
%                                       adversarial time-to-collision used as the
%                                       barrier function (the method of the paper)
%                          'sattc_vraw' stochastic aTTC-CBF (chance-constrained)
%       'num_agents'   - number of agents              (default: 2)
%       'T_sim'        - simulation time [s]            (default: 1000)
%       'dt'           - integration timestep [s]      (default: 0.1)
%       'r_cbf'        - CBF barrier radius [km]        (default: 1).
%                        Used for evader-evader (and all-pairs when no
%                        evader_mask).
%       'r_cbf_adv'    - CBF barrier radius for ADVERSARIAL (evader-pursuer)
%                        pairs.  NaN -> fall back to r_cbf.  Same gating as
%                        attc_tau_min_adv: only takes effect when an
%                        evader_mask is present.  (default: NaN)
%       'r_ttc'        - TTC collision radius [km]     (default: 0)
%       'tau_crit'     - critical TTC threshold [s]    (default: 5.0)
%       'ttc_horizon'  - TTC display cap [s]           (default: 60)
%       'attc_dt'      - aTTC forward-sim timestep [s] (default: 0.2, auto-clamped to r_ttc/v_max)
%       'attc_horizon' - aTTC forward-sim horizon [s]  (default: ttc_horizon)
%       'omega_max'    - max turn rate [rad/s]         (default: 0.4)
%       'a_max'        - max accel [km/s^2] (default: 0.05)
%       'nu_max'       - max gamma rate (3D) [rad/s]   (default: 0.1)
%       'v_max'        - max speed [km/s]              (default: 0.5)
%       'cbf_alpha1'   - HOCBF parameter               (default: 0.1)
%       'cbf_alpha2'   - HOCBF parameter               (default: 1.0)
%       'cbf_activate_dist' - CBF only active below     (default: 4*r_cbf
%                              for HOCBF/sHOCBF, Inf for the aTTC variants;
%                              the aTTC gate already encodes distance +
%                              approach rate via the predicted aTTC)
%       'mission'      - 'waypoints_random' | 'waypoints_random_sync' | 'waypoints_spiral'
%                        'escape' | 'head_on' | 'pinwheel' | 'gauntlet' | 'random_sphere'
%                        | 'head_on_ping_pong' | 'pursuit' | 'team_pursuit'
%                        | 'random_sphere_pursuit' | 'formation' | 'formation_pursuit'
%                        | 'formation_pursuit_vis'
%                        (default: 'waypoints_random')
%                        waypoints_random_sync: same as waypoints_random but agents wait at each WP until all have arrived
%                        head_on:  N agents on circle, each aims at antipodal point; all paths cross at origin simultaneously
%                        pinwheel: floor(N/2) agents fly east, ceil(N/2) fly north; T-bone crossings at interior grid points
%                        gauntlet: floor(N/2) slow "wall" agents (v=0.05) + ceil(N/2) fast "runners" overtaking from behind
%                        random_sphere: agents start on a sphere (circle in 2D) and fly across to the other side;
%                                       each waypoint is on the hemisphere opposite the previous one, creating
%                                       rich crossing geometries for NN training data
%                        head_on_ping_pong: paired agents on opposite poles of a sphere, flying
%                                          back and forth head-on forever.  Odd agents get a random
%                                          start, even agents start diametrically opposite.
%                        pursuit:      agent 1 = evader (waypoint), agents 2..N = pursuers chasing it.
%                                          Pursuer speed = pursuit_advantage * v_desired.  No CBF on pursuers.
%                        team_pursuit: agents 1..(N-K) = evaders following synced waypoints
%                                          (like waypoints_random_sync); agents (N-K+1)..N = K pursuers.
%                                          Each pursuer is initially assigned a uniform-random
%                                          evader and chases that one (sticky); on collision
%                                          (dist < 2*r_col) it re-picks a new random evader
%                                          excluding the just-caught target.  K = num_pursuers
%                                          (default 1).  Evaders get full CBF; pursuers fly
%                                          uninhibited.
%                        random_sphere_pursuit: same role split as team_pursuit, but each
%                                          evader follows a random_sphere-style waypoint
%                                          sequence (sphere of radius v_des*100/6 km, first
%                                          WP antipodal, subsequent WPs on opposite hemisphere)
%                                          rather than synced shared waypoints.  Pursuers
%                                          start at random points on the same sphere.  Sticky
%                                          targeting + attc_*_adv overrides apply identically.
%                        formation:    agent 1 = leader (random waypoints); agents 2..N = followers
%                                          tracking body-frame V-formation slots relative to leader's
%                                          (yaw, pitch) pose.  All agents = evaders, full CBF, uniform
%                                          v_max and v_desired.  Formation distorts during sharp turns.
%                        formation_pursuit: same formation team as 'formation' for agents 1..(N-K),
%                                          plus K pursuers (agents N-K+1..N) using the same sticky-
%                                          target scheme as team_pursuit.  K = num_pursuers (default 1).
%       'num_pursuers' - team_pursuit / random_sphere_pursuit / formation_pursuit only:
%                        number of pursuers (last K agents). (default: 1)
%       'formation_type'          - formation/formation_pursuit only: 'v'   (default: 'v')
%       'formation_stagger_angle' - V half-angle [rad]                       (default: pi/4)
%       'formation_spacing'       - slot spacing [km]                        (default: 4*r_ttc)
%       'v_desired'    - cruise speed [km/s], scalar (broadcast) or per-agent vector [na x 1] (default: 0.25)
%       'waypoints'    - cell array {na x 1}, each [nwp x 2 or 3] (default: auto)
%       'wp_capture_radius' - switch to next WP when within this [km] (default: 1)
%       'attc_loss'       - loss tag baked into the NN dir name. Default 'iw_huber3'.
%       'attc_data_tag'   - data tag baked into the NN dir name. Default 'data1'.
%       The NN dir for attc_vraw / sattc_vraw is built as:
%         '../Results/nn_bs128_h128-128-64_vraw_<attc_loss>_<attc_data_tag>'
%       and must contain 'attc_nn_weights.mat' (see Python_Code/
%       export_attc_weights.py, which writes it after training).
%       'attc_tau_min' - aTTC-CBF safety threshold [s]:
%                        h = aTTC_NN - tau_min, CBF enforces h >= 0
%                        (default: 5.0).  Used for evader-evader pairs
%                        (and for all pairs when evader_mask is unset).
%       'attc_alpha'   - aTTC-CBF class-K gain           (default: 0.1)
%       'attc_activate_tau' - only impose aTTC constraint
%                        when aTTC < this [s]             (default: 30.0).
%                        Used for evader-evader pairs (or all pairs without
%                        evader_mask).
%       'attc_tau_min_adv' - aTTC-CBF safety threshold [s] for ADVERSARIAL
%                        (evader-pursuer) pairs.  NaN -> fall back to
%                        attc_tau_min.  Only meaningful when evader_mask is
%                        set (team_pursuit, formation_pursuit, or pursuit).
%                        (default: NaN)
%       'attc_activate_tau_adv' - aTTC activation threshold [s] for
%                        adversarial pairs.  NaN -> fall back to
%                        attc_activate_tau.  (default: NaN)
%       'attc_grad_clip' - max NN gradient norm          (default: 100.0)
%       'attc_dt'      - aTTC forward-sim timestep [s]   (default: 0.2, auto-clamped to r_ttc/v_max)
%       'attc_horizon' - aTTC forward-sim horizon [s]    (default: ttc_horizon)
%       'attc_type'    - aTTC mode: 'bang','prop',
%                        'pursuit_ego','pursuit_min'      (default: 'pursuit_ego')
%                        Note: NN-based CBFs were trained on pursuit_ego;
%                        the simulator default matches that so the truth
%                        in res.attc is on the same scale as NN predictions.
%       'compute_ttc'  - compute TTC/STTC/TTC2/STTC2    (default: true)
%       'compute_attc' - compute adversarial TTC         (default: true)
%       'epsilon_ttc'  - small eps for TTC computation   (default: 1e-4)
%       'noise_std_mult'- scalar multiplier on default noise_std vector (default: 1)
%                        2D default: [0.01, 0.01, 0.01, 0.005]
%                        3D default: [0.01, 0.01, 0.01, 0.01, 0.005, 0.0]
%       'use_chance_constraint' - tighten CBF by epsilon (default: false)
%       'epsilon'      - max collision probability       (default: 0.05)
%       'use_kalman_filter' - KF measurement update      (default: false)
%       'obs_noise_std'- meas. noise std per pos [km]    (default: [0.01,0.01] or [0.01,0.01,0.01])
%       'x0'           - initial states [na x sd]        (default: auto)
%       'policy'       - @(x_all, id, t, p) -> u         (default: waypoint tracking)
%
%   Example – compare CBF on vs off:
%       r1 = auspice_sim('cbf_type', 'none');
%       r2 = auspice_sim('cbf_type', 'attc_vraw', 'attc_loss', 'iw_huber3', 'attc_data_tag', 'data1');
%
%   Dynamics (2D Dubins, Eq. 4 in AUSPICE document):
%       dx/dt = [v*cos(theta); v*sin(theta); omega; a]
%
%   Dynamics (3D Dubins, Eq. 2 in AUSPICE document):
%       dx/dt = [v*cos(gamma)*cos(psi); v*cos(gamma)*sin(psi);
%               v*sin(gamma); u_psi; u_gamma; 0]
%
%   TTC:  analytic formula assuming constant-velocity propagation.
%   aTTC: adversarial TTC — forward-sim with crash-seeking bang-bang controls.
%   CBF:  higher-order CBF (relative degree 2) with distance-based barrier
%         h(x) = ||p_i - p_j||^2 - (2r_cbf)^2, solved via QP.

%% ==================== PARSE INPUTS ====================
ip = inputParser;
addParameter(ip, 'dim',        2);
addParameter(ip, 'use_cbf',    []);     % DEPRECATED: use cbf_type='none' instead
addParameter(ip, 'num_agents', 2);
addParameter(ip, 'T_sim',      1000);
addParameter(ip, 'dt',         0.1);
addParameter(ip, 'r_cbf',      1);
% Adversarial (evader-pursuer) override on the CBF safety radius.  NaN
% sentinel -> fall back to r_cbf for all pairs.  Only meaningful when an
% evader_mask exists (team_pursuit, formation_pursuit, random_sphere_pursuit,
% pursuit).  Affects HOCBF directly (h = d^2 - (2*r_col)^2) and aTTC variants
% indirectly (forward-sim collision threshold).
addParameter(ip, 'r_cbf_adv',  NaN);
addParameter(ip, 'r_ttc',      0.1);
addParameter(ip, 'tau_crit',   5.0);
addParameter(ip, 'ttc_horizon',60);
addParameter(ip, 'omega_max',  0.4);
addParameter(ip, 'a_max',      0.05);
addParameter(ip, 'nu_max',     0.1);
addParameter(ip, 'v_max',      0.5);
addParameter(ip, 'cbf_alpha1', 0.1);
% Adversarial (evader-pursuer) override on the HOCBF class-K gain α₁.
% NaN sentinel -> ep pairs use cbf_alpha1.  Used by hocbf and shocbf;
% no-op for the aTTC types.
addParameter(ip, 'cbf_alpha1_adv', NaN);
addParameter(ip, 'cbf_alpha2', 1.0);
addParameter(ip, 'cbf_activate_dist', []);
% Adversarial (evader-pursuer) override on the geometric activation gate.
% NaN sentinel -> use cbf_activate_dist for ep pairs as well.
addParameter(ip, 'cbf_activate_dist_adv', NaN);
addParameter(ip, 'cbf_type',    'none');    % 'none','hocbf','shocbf','attc_vraw','sattc_vraw'
% aTTC-CBF parameters
% NN weights directory is built from (attc_loss, attc_data_tag):
%   '../Results/nn_bs128_h128-128-64_vraw_<attc_loss>_<attc_data_tag>'
addParameter(ip, 'attc_loss',       'iw_huber3');
addParameter(ip, 'attc_data_tag',   'data1');
addParameter(ip, 'attc_tau_min',     5.0);   % safety threshold (s): h = aTTC_NN - tau_min
addParameter(ip, 'attc_alpha',       0.1);   % class-K gain for CBF constraint
% Adversarial (ep) override on the aTTC class-K gain.  NaN sentinel -> ep
% pairs use attc_alpha.  Used by attc_vraw / sattc_vraw.
addParameter(ip, 'attc_alpha_adv',   NaN);
addParameter(ip, 'attc_activate_tau',30.0);   % only impose constraint when aTTC < this
% Adversarial (evader-pursuer) overrides.  NaN sentinel -> fall back to the
% canonical attc_tau_min / attc_activate_tau.  Only meaningful when an
% evader_mask exists (team_pursuit, formation_pursuit, plain pursuit).
addParameter(ip, 'attc_tau_min_adv',      NaN);
addParameter(ip, 'attc_activate_tau_adv', NaN);
addParameter(ip, 'attc_grad_clip', 100.0);    % max gradient norm
% Stochastic aTTC-CBF (saTTC) parameters
addParameter(ip, 'scbf_p',     0.9);    % desired probability of safety [0,1)
addParameter(ip, 'scbf_alpha', 0.1);    % class-K gain for SCBF constraint
addParameter(ip, 'scbf_beta',  0.01);   % slack to prevent infeasibility
addParameter(ip, 'mission', 'waypoints_random');  % 'waypoints_random' | 'waypoints_spiral' | 'escape'
addParameter(ip, 'v_desired',  0.25);
addParameter(ip, 'waypoints',  {});
addParameter(ip, 'wp_capture_radius', 1);
addParameter(ip, 'epsilon_ttc', 1e-4);
addParameter(ip, 'attc_dt',      0.2);       % aTTC forward-sim timestep [s]
addParameter(ip, 'attc_horizon', []);         % aTTC horizon [s] (defaults to ttc_horizon)
addParameter(ip, 'compute_ttc',  true);       % compute TTC/STTC/TTC2/STTC2
addParameter(ip, 'compute_attc', true);       % compute adversarial TTC
addParameter(ip, 'attc_type',    'pursuit_ego');  % 'bang','prop','pursuit_ego','pursuit_min'  (default matches NN training)
addParameter(ip, 'noise_std_mult', 1);
addParameter(ip, 'use_chance_constraint', false);
addParameter(ip, 'epsilon',    0.05);
addParameter(ip, 'use_kalman_filter', false);
addParameter(ip, 'obs_noise_std', []);
addParameter(ip, 'x0',         []);
addParameter(ip, 'policy',     []);
addParameter(ip, 'pursuit_advantage', 1.1);  % pursuer speed = pursuit_advantage * v_max
addParameter(ip, 'num_pursuers',      1);    % team_pursuit: last K agents = pursuers
% Sphere mission sizing: R_sphere = v_desired * crossing_time / 2.  Used by
% random_sphere, random_sphere_pursuit, and head_on_ping_pong.  Default 30 s
% (so for v_desired=0.25, v_max=0.5: R_sphere=3.75 km, diameter=7.5 km).
addParameter(ip, 'crossing_time',     50);
% Formation flight parameters (mission='formation' or 'formation_pursuit')
addParameter(ip, 'formation_type',          'v');     % only 'v' supported initially
addParameter(ip, 'formation_stagger_angle', pi/4);    % half-angle of the V [rad]
addParameter(ip, 'formation_spacing',       NaN);     % slot spacing [km]; NaN -> 4*r_ttc
% Spread of the random waypoint grid for waypoints_random,
% waypoints_random_sync, team_pursuit, and formation* missions.  Defines
% the X-axis full width; Y and Z half-widths follow the historic aspect
% ratio (0.8*spread and 0.4*spread).  NaN sentinel -> per-mission historic
% default (250 for waypoints_random[_sync]; 50 for team_pursuit /
% formation[_pursuit]).
addParameter(ip, 'waypoint_spread',         NaN);
parse(ip, varargin{:});
params = ip.Results;

% Default formation_spacing = 4 * r_ttc (filled in here so user-supplied r_ttc propagates).
if isnan(params.formation_spacing)
    params.formation_spacing = 4 * params.r_ttc;
end

% Backward compatibility: use_cbf=true/false -> cbf_type
if ~isempty(params.use_cbf)
    if params.use_cbf && strcmp(params.cbf_type, 'none')
        params.cbf_type = 'hocbf';   % legacy default when use_cbf=true
    elseif ~params.use_cbf
        params.cbf_type = 'none';
    end
end
params.use_cbf = ~strcmp(params.cbf_type, 'none');  % derived flag for legacy code

% Validate cbf_type up front — fail loudly on typos rather than silently
% falling through to HOCBF later.
%
valid_cbf_types = {'none','hocbf','shocbf','attc_vraw','sattc_vraw'};
if ~any(strcmp(params.cbf_type, valid_cbf_types))
    error('auspice_sim:unknownCbfType', ...
        ['Unknown cbf_type ''%s''.\n  Valid options: %s'], ...
        params.cbf_type, strjoin(valid_cbf_types, ', '));
end

% Build the NN weights directory from (attc_loss, attc_data_tag).  Used by
% attc_vraw / sattc_vraw; other cbf types ignore this.
params.attc_model_dir = sprintf( ...
    '../Results/nn_bs128_h128-128-64_vraw_%s_%s', ...
    params.attc_loss, params.attc_data_tag);

% State / control dimensions
if params.dim == 2
    params.state_dim = 4;   % [px, py, theta, v]
    params.ctrl_dim  = 2;   % [omega, a]
elseif params.dim == 3
    params.state_dim = 6;   % [px, py, pz, psi, gamma, v]
    params.ctrl_dim  = 3;   % [u_psi, u_gamma, a]
else
    error('dim must be 2 or 3');
end

na = params.num_agents;
sd = params.state_dim;
cd = params.ctrl_dim;

% Broadcast scalar v_desired to per-agent vector
if isscalar(params.v_desired)
    params.v_desired = repmat(params.v_desired, na, 1);
end

% Pursuit mode flag (set in mission block below, default off)
params.pursuit_mode = false;

% Default CBF activation distance
if isempty(params.cbf_activate_dist)
    if any(strcmp(params.cbf_type, {'attc_vraw','sattc_vraw'}))
        % aTTC-based CBFs already gate on time-to-collision via
        % attc_activate_tau, which inherently encodes both distance AND
        % approach rate.  Default to Inf so only the aTTC gate is in
        % effect.  User can still override with a finite value to keep
        % distant pairs out of the QP for compute reasons.
        params.cbf_activate_dist = Inf;
    else
        params.cbf_activate_dist = 2 * (2*params.r_cbf);
    end
end

% NN-based CBFs (attc_vraw / sattc_vraw): the vraw NN uses 3*dim features
% [dp; vel_i; vel_j] with scalar v_max conditioning (FiLM).
if params.use_cbf && any(strcmp(params.cbf_type, ...
        {'attc_vraw','sattc_vraw'}))
    is_stochastic = strcmp(params.cbf_type, 'sattc_vraw');
    cbf_label = upper(params.cbf_type);

    model_dir = params.attc_model_dir;
    % Resolve relative path from Matlab_code/ directory
    if ~(startsWith(model_dir, '/') || startsWith(model_dir, '~') || ...
         (length(model_dir) >= 2 && model_dir(2) == ':'))
        model_dir = fullfile(fileparts(mfilename('fullpath')), model_dir);
    end
    mat_file = fullfile(model_dir, 'attc_nn_weights.mat');
    if ~exist(mat_file, 'file')
        error('%s-CBF: weight file not found: %s\nRun export_attc_weights.py first.', ...
            cbf_label, mat_file);
    end
    params.attc_nn = load(mat_file);
    % Bias vectors → columns (main pathway)
    for k = 1:4
        bname = sprintf('b%d', k);
        params.attc_nn.(bname) = params.attc_nn.(bname)(:);
    end
    params.attc_nn.feat_mean = params.attc_nn.feat_mean(:);
    params.attc_nn.feat_std  = params.attc_nn.feat_std(:);
    % FiLM conditioning weights — required
    assert(isfield(params.attc_nn, 'film') && params.attc_nn.film > 0.5, ...
        '%s-CBF: model must be FiLM-conditioned. Loaded weights from %s are not FiLM.', ...
        cbf_label, model_dir);
    for k = 1:3
        cbname = sprintf('cond_b%d', k);
        params.attc_nn.(cbname) = params.attc_nn.(cbname)(:);
    end
    params.attc_nn.film_sizes = params.attc_nn.film_sizes(:)';

    fprintf('  %s-CBF: FiLM model (v_max range [%.3f, %.3f])\n', ...
        cbf_label, params.attc_nn.v_max_min, params.attc_nn.v_max_max);

    % Validate input feature dim against NN's W1.  Vraw NN expects 3*dim
    % features [dp; vel_i; vel_j].
    expected_input = size(params.attc_nn.W1, 1);
    assert(expected_input == 3*params.dim, ...
        ['%s-CBF: model input dim is %d but expected %d ' ...
         '(=3*dim for [dp; vel_i; vel_j]).  Wrong model dir?'], ...
        cbf_label, expected_input, 3*params.dim);
    fprintf('  %s-CBF: loaded model from %s  (input_dim=%d)\n', ...
        cbf_label, model_dir, expected_input);

    % Final diagnostic line: stochastic CBFs care about (p, alpha, beta);
    % deterministic ones care about (alpha, activate_tau).
    if is_stochastic
        fprintf('  %s-CBF: p=%.2f, alpha=%.2f, beta=%.4f, tau_min=%.1fs\n', ...
            cbf_label, params.scbf_p, params.scbf_alpha, ...
            params.scbf_beta, params.attc_tau_min);
    else
        fprintf('  %s-CBF: tau_min=%.1fs, alpha=%.1f, activate_tau=%.1fs\n', ...
            cbf_label, params.attc_tau_min, params.attc_alpha, ...
            params.attc_activate_tau);
    end
end

% aTTC defaults and safety clamp
if isempty(params.attc_horizon)
    params.attc_horizon = params.ttc_horizon;
end
if params.r_ttc > 0
    max_safe_dt = params.r_ttc / max(params.v_max);
    if params.attc_dt > max_safe_dt
        warning('auspice_sim:attcDt', ...
            'attc_dt (%.3g) > r_ttc/v_max (%.3g); clamping.', ...
            params.attc_dt, max_safe_dt);
        params.attc_dt = max_safe_dt;
    end
end

% Noise standard deviations: scalar multiplier × default vector
if params.dim == 2
    noise_std_base = [0.01, 0.01, 0.01, 0.005];             % [px(km), py(km), theta(rad), v(km/s)]
else
    noise_std_base = [0.01, 0.01, 0.01, 0.01, 0.005, 0.005]; % [px,py,pz(km),psi,gam(rad),v(km/s)]
end
params.noise_std = params.noise_std_mult * noise_std_base;
params.Q = diag(params.noise_std.^2);   % process noise covariance

% Kalman filter: observation model & measurement noise
if params.use_kalman_filter
    if isempty(params.obs_noise_std)
        if params.dim == 2
            params.obs_noise_std = [0.01, 0.01];         % GPS-like ~10m = 0.01 km std
        else
            params.obs_noise_std = [0.01, 0.01, 0.01];
        end
    end
    params.R_obs = diag(params.obs_noise_std.^2);
    if params.dim == 2
        params.H = [1 0 0 0; 0 1 0 0];                  % observe [px, py]
    else
        params.H = [eye(3), zeros(3,3)];                  % observe [px, py, pz]
    end
end

% Chance constraint: kappa = Phi^{-1}(1-epsilon)  (no toolbox needed)
if params.use_chance_constraint
    params.kappa = sqrt(2) * erfinv(2*(1-params.epsilon) - 1);
end

% Check for quadprog
params.has_quadprog = (exist('quadprog','file') == 2);
if params.use_cbf && ~params.has_quadprog
    warning('quadprog not found (Optimization Toolbox). Using fallback QP solver.');
end

%% ==================== WAYPOINTS & INITIAL CONDITIONS ====================

% Generate waypoints based on mission type
if isempty(params.waypoints)
    switch lower(params.mission)

        case {'waypoints_random', 'waypoints_random_sync'}
            n_wp = 8;
            spread = params.waypoint_spread;
            if isnan(spread), spread = 250; end   % historic default
            shared_wp = generate_random_waypoint_grid(n_wp, params.dim, spread);
            params.waypoints = cell(na, 1);
            for i = 1:na, params.waypoints{i} = shared_wp; end

        case 'waypoints_spiral'
            n_wp = 4;
            % Archimedean spiral: radius grows linearly, total angular
            % sweep stays under 360 deg so straight-line segments between
            % consecutive waypoints never cross each other.
            r_start = 30;            % first WP distance from origin [km]
            r_end   = 200;           % last WP distance [km]
            theta0  = 2*pi*rand;     % random starting angle
            d_theta = 0.85*2*pi / max(n_wp-1, 1);  % ~306 deg total sweep

            radii  = linspace(r_start, r_end, n_wp)';
            angles = theta0 + (0:n_wp-1)' * d_theta;

            if params.dim == 2
                shared_wp = [radii .* cos(angles), ...
                             radii .* sin(angles)];
            else
                zz = linspace(8, 12, n_wp)';      % altitude range [km]
                shared_wp = [radii .* cos(angles), ...
                             radii .* sin(angles), zz];
            end
            params.waypoints = cell(na, 1);
            for i = 1:na, params.waypoints{i} = shared_wp; end

        case 'escape'
            % Each agent gets its own target on circle/sphere of radius 30r
            r_final      = 30 * params.r_cbf;
            angles_final = (0:na-1)' * (2*pi / na);
            if params.dim == 2
                escape_wp = [r_final*cos(angles_final), r_final*sin(angles_final)];
            else  % equatorial circle at cruise altitude 10 km
                escape_wp = [r_final*cos(angles_final), r_final*sin(angles_final), 10*ones(na,1)];
            end
            % Odd agents: final WP up +30 km; even agents: down -30 km
            % (y in 2D, z in 3D)
            params.waypoints = cell(na, 1);
            for i = 1:na
                wp_i = escape_wp(i,:);
                if mod(i,2) == 1
                    if params.dim == 2, wp_i(2) = wp_i(2) + 30;
                    else,               wp_i(3) = wp_i(3) + 30; end
                else
                    if params.dim == 2, wp_i(2) = wp_i(2) - 30;
                    else,               wp_i(3) = wp_i(3) - 30; end
                end
                params.waypoints{i} = wp_i;
            end

        case 'head_on'
            % Agents at evenly-spaced angles on a circle, each targeting the antipodal
            % point.  All N paths pass through the origin simultaneously.
            R_ic      = 20 * params.r_cbf;
            angles_ic = (0:na-1)' * (2*pi / na);
            params.waypoints = cell(na, 1);
            for i = 1:na
                ang_wp = angles_ic(i) + pi;   % antipodal
                if params.dim == 2
                    params.waypoints{i} = [R_ic*cos(ang_wp), R_ic*sin(ang_wp)];
                else
                    params.waypoints{i} = [R_ic*cos(ang_wp), R_ic*sin(ang_wp), 10];
                end
            end

        case 'pinwheel'
            % Two perpendicular traffic streams: floor(N/2) agents flying east,
            % ceil(N/2) agents flying north.  Each east agent crosses every north
            % agent at a distinct interior point with a T-bone geometry.
            % Crossing times are equal for same-offset pairs (guaranteed encounters).
            R_ic = 20 * params.r_cbf;
            na_A = floor(na/2);     % east-flying group
            na_B = na - na_A;       % north-flying group
            y_A  = linspace(-R_ic/2, R_ic/2, max(na_A, 1));  % y-offset of each east agent
            x_B  = linspace(-R_ic/2, R_ic/2, max(na_B, 1));  % x-offset of each north agent
            params.waypoints = cell(na, 1);
            for i = 1:na_A
                if params.dim == 2
                    params.waypoints{i}      = [R_ic, y_A(i)];       % east target
                else
                    params.waypoints{i}      = [R_ic, y_A(i), 10];
                end
            end
            for i = 1:na_B
                if params.dim == 2
                    params.waypoints{na_A+i} = [x_B(i), R_ic];       % north target
                else
                    params.waypoints{na_A+i} = [x_B(i), R_ic, 10];
                end
            end

        case 'gauntlet'
            % Half the agents form a slow-moving "wall"; the other half are
            % fast "runners" that overtake from behind.
            na_wall = floor(na/2);
            na_run  = na - na_wall;
            R_ic    = 20 * params.r_cbf;
            wall_spacing = 3 * params.r_cbf;
            wall_spread  = wall_spacing * (na_wall - 1) / 2;

            y_wall = linspace(-wall_spread, wall_spread, max(na_wall, 1)) + params.r_cbf;
            y_run  = linspace(-wall_spread, wall_spread, max(na_run,  1)) - params.r_cbf;

            params.waypoints = cell(na, 1);
            for i = 1:na_wall
                if params.dim == 2
                    params.waypoints{i} = [R_ic, y_wall(i)];
                else
                    params.waypoints{i} = [R_ic, y_wall(i), 10 + params.r_cbf];
                end
            end
            for i = 1:na_run
                if params.dim == 2
                    params.waypoints{na_wall+i} = [R_ic, y_run(i)];
                else
                    params.waypoints{na_wall+i} = [R_ic, y_run(i), 10 - params.r_cbf];
                end
            end

            % Set per-agent speeds: wall = 0.05, runners = original v_desired
            v_run = params.v_desired(1);   % original (uniform) speed
            params.v_desired(1:na_wall)       = 0.05;
            params.v_desired(na_wall+1:end)   = v_run;

        case 'random_sphere'
            % Agents start on a sphere (2D: circle) of diameter
            % v_desired * crossing_time, giving ~crossing_time seconds to
            % fly across.  First WP is the antipodal point; subsequent WPs
            % are random on the opposite hemisphere from the previous WP.
            % Per-agent geometry is generated by random_sphere_wp_sequence
            % (local helper) so this case shares its logic with
            % random_sphere_pursuit.
            R_sphere = params.v_desired(1) * params.crossing_time / 2;
            n_wp     = 2500;

            params.sphere_radius = R_sphere;
            params.waypoints     = cell(na, 1);
            params.sphere_ic     = zeros(na, params.dim);
            for i = 1:na
                [ic_i, wps_i] = random_sphere_wp_sequence(R_sphere, params.dim, n_wp);
                params.sphere_ic(i,:) = ic_i;
                params.waypoints{i}   = wps_i;
            end

        case 'head_on_ping_pong'
            % Paired agents on opposite poles of a sphere, flying head-on
            % back and forth.  Odd-numbered agents get a random start on
            % the sphere; even-numbered agents start at the antipodal point
            % of the preceding odd agent.  Waypoints just alternate between
            % the two poles so agents bounce forever.
            R_sphere = params.v_desired(1) * params.crossing_time / 2;
            n_wp     = 2500;

            params.sphere_radius = R_sphere;
            params.waypoints     = cell(na, 1);

            if params.dim == 2
                params.sphere_ic = zeros(na, 2);
                for i = 1:na
                    if mod(i, 2) == 1
                        % Odd agent: random start on circle
                        theta_start = 2*pi * rand;
                        params.sphere_ic(i,:) = R_sphere * [cos(theta_start), sin(theta_start)];
                    else
                        % Even agent: antipodal to previous odd agent
                        params.sphere_ic(i,:) = -params.sphere_ic(i-1,:);
                    end

                    % Waypoints: alternate between own start and partner's start
                    if mod(i, 2) == 1
                        pole_A = params.sphere_ic(i,:);       % own start
                    else
                        pole_A = params.sphere_ic(i,:);       % own start (antipodal)
                    end
                    % partner is the other member of the pair
                    if mod(i, 2) == 1
                        pole_B = -pole_A;                     % antipodal = partner's start
                    else
                        pole_B = params.sphere_ic(i-1,:);     % partner's start
                    end

                    wps = zeros(n_wp, 2);
                    for w = 1:n_wp
                        if mod(w, 2) == 1
                            wps(w,:) = pole_B;  % fly to partner's pole
                        else
                            wps(w,:) = pole_A;  % fly back home
                        end
                    end
                    params.waypoints{i} = wps;
                end
            else
                params.sphere_ic = zeros(na, 3);
                for i = 1:na
                    if mod(i, 2) == 1
                        % Odd agent: random start on sphere
                        d_start = randn(1, 3);
                        d_start = d_start / norm(d_start);
                        params.sphere_ic(i,:) = R_sphere * d_start;
                    else
                        % Even agent: antipodal to previous odd agent
                        params.sphere_ic(i,:) = -params.sphere_ic(i-1,:);
                    end

                    pole_A = params.sphere_ic(i,:);
                    if mod(i, 2) == 1
                        pole_B = -pole_A;
                    else
                        pole_B = params.sphere_ic(i-1,:);
                    end

                    wps = zeros(n_wp, 3);
                    for w = 1:n_wp
                        if mod(w, 2) == 1
                            wps(w,:) = pole_B;
                        else
                            wps(w,:) = pole_A;
                        end
                    end
                    params.waypoints{i} = wps;
                end
            end

        case 'pursuit'
            % Agent 1 flies toward a distant waypoint; all other agents
            % pursue agent 1 (their waypoints are updated each timestep in
            % the main loop).  Only agent 1 gets a CBF; pursuers fly
            % uninhibited at pursuit_advantage * v_desired.
            params.pursuit_mode = true;
            params.sync_waypoints = false;

            % Pursuer speeds: v_max scaled by pursuit_advantage
            v_max_orig = params.v_max;
            v_max_pursuer = params.pursuit_advantage * v_max_orig;
            params.v_max = [v_max_orig; repmat(v_max_pursuer, na-1, 1)];

            % Desired speeds: pursuers cruise at their v_max, evader unchanged
            v_evader = params.v_desired(1);
            params.v_desired = [v_evader; repmat(v_max_pursuer, na-1, 1)];

            % Agent 1 waypoint: 10*v_max ahead along +x (close enough to orbit)
            wp_dist = 10 * v_max_orig;
            if params.dim == 2
                params.waypoints{1} = [wp_dist, 0];
            else
                params.waypoints{1} = [wp_dist, 0, 10];
            end

            % Pursuer waypoints: initially agent 1's start (origin).
            % These get overwritten each timestep in the main loop.
            for i = 2:na
                if params.dim == 2
                    params.waypoints{i} = [0, 0];
                else
                    params.waypoints{i} = [0, 0, 10];
                end
            end

        case 'team_pursuit'
            % First (na - num_pursuers) agents are EVADERS following
            % synchronized random waypoints (like waypoints_random_sync).
            % Last num_pursuers agents are PURSUERS that each chase their
            % nearest evader; their waypoints get rewritten every timestep
            % in the main loop.  Pursuers fly at pursuit_advantage*v_desired
            % and get NO CBF (matches plain pursuit semantics).
            n_pur = params.num_pursuers;
            n_eva = na - n_pur;
            if n_pur < 1
                error('team_pursuit requires num_pursuers >= 1, got %d', n_pur);
            end
            if n_eva < 1
                error(['team_pursuit requires at least 1 evader: ' ...
                       'num_agents=%d, num_pursuers=%d'], na, n_pur);
            end

            params.pursuit_mode = true;
            params.evader_mask  = false(na, 1);
            params.evader_mask(1:n_eva) = true;

            % Pursuer speeds: scaled by pursuit_advantage like plain pursuit
            v_max_orig    = params.v_max;
            v_max_pursuer = params.pursuit_advantage * v_max_orig;
            v_evader      = params.v_desired(1);
            params.v_max     = repmat(v_max_orig, na, 1);
            params.v_desired = repmat(v_evader,  na, 1);
            params.v_max(~params.evader_mask)     = v_max_pursuer;
            params.v_desired(~params.evader_mask) = v_max_pursuer;

            % Evader waypoints: shared random list (matches waypoints_random_sync)
            n_wp = 8;
            spread = params.waypoint_spread;
            if isnan(spread), spread = 50; end    % historic default
            shared_wp = generate_random_waypoint_grid(n_wp, params.dim, spread);
            params.waypoints = cell(na, 1);
            for i = 1:n_eva
                params.waypoints{i} = shared_wp;
            end
            % Pursuer waypoints: dummy origin point (gets overwritten in main loop)
            for i = (n_eva+1):na
                if params.dim == 2
                    params.waypoints{i} = [0, 0];
                else
                    params.waypoints{i} = [0, 0, 10];
                end
            end

        case 'random_sphere_pursuit'
            % Same role split as team_pursuit, but each evader follows a
            % random_sphere-style waypoint sequence (own start -> antipodal
            % -> opposite-hemisphere randoms) instead of a synced shared
            % list.  Pursuers start at random points on the same sphere.
            n_pur = params.num_pursuers;
            n_eva = na - n_pur;
            if n_pur < 1
                error(['random_sphere_pursuit requires num_pursuers >= 1, ' ...
                       'got %d'], n_pur);
            end
            if n_eva < 1
                error(['random_sphere_pursuit requires at least 1 evader: ' ...
                       'num_agents=%d, num_pursuers=%d'], na, n_pur);
            end

            params.pursuit_mode = true;
            params.evader_mask  = false(na, 1);
            params.evader_mask(1:n_eva) = true;

            % Pursuer speed scaling (same as team_pursuit / pursuit)
            v_max_orig    = params.v_max;
            v_max_pursuer = params.pursuit_advantage * v_max_orig;
            v_evader      = params.v_desired(1);
            params.v_max     = repmat(v_max_orig, na, 1);
            params.v_desired = repmat(v_evader,  na, 1);
            params.v_max(~params.evader_mask)     = v_max_pursuer;
            params.v_desired(~params.evader_mask) = v_max_pursuer;

            % Sphere geometry — shared with plain random_sphere via the
            % random_sphere_wp_sequence / random_sphere_point helpers below.
            R_sphere = v_evader * params.crossing_time / 2;
            n_wp     = 2500;
            params.sphere_radius = R_sphere;
            params.waypoints     = cell(na, 1);
            params.sphere_ic     = zeros(na, params.dim);
            if params.dim == 2
                pursuer_dummy_wp = [0, 0];
            else
                pursuer_dummy_wp = [0, 0, 10];
            end
            % Evaders: own random start + random_sphere WP sequence
            for i = 1:n_eva
                [ic_i, wps_i] = random_sphere_wp_sequence(R_sphere, params.dim, n_wp);
                params.sphere_ic(i,:) = ic_i;
                params.waypoints{i}   = wps_i;
            end
            % Pursuers: random sphere starts, single dummy WP — the
            % sticky-targeting block in the main loop rewrites it each step.
            for i = (n_eva+1):na
                params.sphere_ic(i,:) = random_sphere_point(R_sphere, params.dim);
                params.waypoints{i}   = pursuer_dummy_wp;
            end

        case {'formation', 'formation_pursuit', 'formation_pursuit_vis'}
            % Formation flight.
            %   agent 1            = leader, follows random waypoints (or a
            %                        single far-off waypoint for _vis)
            %   agents 2..n_eva    = followers, track body-frame V slots
            %                        relative to leader's (yaw, pitch) pose
            %   agents n_eva+1..na = pursuers (formation_pursuit* only),
            %                        each chasing nearest formation member
            % All formation members share v_desired and v_max (same airframe).
            is_pursuit_variant = strcmp(params.mission, 'formation_pursuit') ...
                              || strcmp(params.mission, 'formation_pursuit_vis');
            if is_pursuit_variant
                n_pur = params.num_pursuers;
                if n_pur < 1
                    error(['%s requires num_pursuers >= 1, got %d'], ...
                          params.mission, n_pur);
                end
            else
                n_pur = 0;
            end
            n_eva = na - n_pur;
            if n_eva < 2
                error(['%s requires at least 2 formation members ' ...
                       '(1 leader + >=1 follower): num_agents=%d, ' ...
                       'num_pursuers=%d'], params.mission, na, n_pur);
            end

            % Validate formation_type (only 'v' supported for now)
            if ~strcmp(params.formation_type, 'v')
                error(['formation_type=''%s'' not supported. ' ...
                       'Supported: ''v'''], params.formation_type);
            end

            params.evader_mask = false(na, 1);
            params.evader_mask(1:n_eva) = true;
            params.pursuit_mode   = is_pursuit_variant;
            params.sync_waypoints = false;

            % Speeds: formation team uniform; pursuers (if any) get
            % pursuit_advantage * v_max like plain pursuit / team_pursuit.
            v_max_orig = params.v_max;
            v_evader   = params.v_desired(1);
            params.v_max     = repmat(v_max_orig, na, 1);
            params.v_desired = repmat(v_evader,   na, 1);
            if is_pursuit_variant
                v_max_pursuer = params.pursuit_advantage * v_max_orig;
                params.v_max(~params.evader_mask)     = v_max_pursuer;
                params.v_desired(~params.evader_mask) = v_max_pursuer;
            end

            % Leader's waypoint list
            if strcmp(params.mission, 'formation_pursuit_vis')
                % Vis mission: single far-off waypoint straight ahead in +x.
                % Far enough that the leader never reaches it in the vis
                % window and just flies a straight line.
                if params.dim == 2
                    params.waypoints{1} = [500, 0];
                else
                    params.waypoints{1} = [500, 0, 10];
                end
            else
                % Random, same scheme as waypoints_random
                n_wp = 8;
                spread = params.waypoint_spread;
                if isnan(spread), spread = 50; end    % historic default
                params.waypoints{1} = generate_random_waypoint_grid(n_wp, params.dim, spread);
            end

            % Followers: dummy single-row waypoint, rewritten every step
            % from leader's pose + body-frame slot offset.
            for i = 2:n_eva
                if params.dim == 2
                    params.waypoints{i} = [0, 0];
                else
                    params.waypoints{i} = [0, 0, 10];
                end
            end

            % Pursuers: dummy single-row waypoint, rewritten by the
            % pursuer-targeting block in the main loop (formation_pursuit only).
            for i = (n_eva+1):na
                if params.dim == 2
                    params.waypoints{i} = [0, 0];
                else
                    params.waypoints{i} = [0, 0, 10];
                end
            end

        otherwise
            error(['mission must be one of: waypoints_random, ' ...
                   'waypoints_random_sync, waypoints_spiral, escape, head_on, ' ...
                   'pinwheel, gauntlet, random_sphere, head_on_ping_pong, ' ...
                   'pursuit, team_pursuit, random_sphere_pursuit, ' ...
                   'formation, formation_pursuit, formation_pursuit_vis']);
    end
end

% Per-agent waypoint index (all start targeting waypoint 1)
params.wp_idx = ones(na, 1);

% Per-agent waypoint advance counter — increments by 1 every time wp_idx
% advances (including modular wrap-around).  For agents whose waypoint list
% is a single dummy row (pursuers, formation followers), this stays at 0
% because the advance code's `mod(idx_wp, 1) + 1 = 1` never changes wp_idx.
% Saved in results.wp_advance_count at the end of the sim.
params.wp_advance_count = zeros(na, 1);

% Synchronized waypoint advancement
params.sync_waypoints = strcmp(params.mission, 'waypoints_random_sync') ...
                      || strcmp(params.mission, 'team_pursuit');
params.wp_arrived     = false(na, 1);

% Default initial conditions
if isempty(params.x0)
    if strcmp(params.mission, 'escape')
        % --- Escape ICs: random cluster in circle/sphere of radius 10r ---
        r_ic = 2 * params.r_cbf;
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na
                rand_r  = r_ic * sqrt(rand);
                rand_th = 2*pi * rand;
                px = rand_r * cos(rand_th);
                py = rand_r * sin(rand_th);
                theta_i = 2*pi * rand;               % random initial heading
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na
                rand_r  = r_ic * rand^(1/3);         % uniform-in-ball radius
                rand_el = asin(2*rand - 1);           % uniform-on-sphere elevation
                rand_az = 2*pi * rand;
                px = rand_r * cos(rand_el) * cos(rand_az);
                py = rand_r * cos(rand_el) * sin(rand_az);
                pz = 10 + rand_r * sin(rand_el);      % 10 km base altitude
                psi_i = 2*pi * rand;                  % random azimuth heading
                gam_i = (rand - 0.5) * pi/3;          % random pitch in [-30, +30] deg
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end
    elseif strcmp(params.mission, 'head_on')
        % --- Circle ICs: agents at evenly-spaced points on a circle, each
        %     facing its first waypoint (the antipodal point).
        R_ic      = 20 * params.r_cbf;
        angles_ic = (0:na-1)' * (2*pi / na);
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na
                px = R_ic * cos(angles_ic(i));
                py = R_ic * sin(angles_ic(i));
                wp1 = params.waypoints{i}(1,:);
                theta_i = atan2(wp1(2) - py, wp1(1) - px);
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na
                px = R_ic * cos(angles_ic(i));
                py = R_ic * sin(angles_ic(i));
                pz = 10;
                wp1 = params.waypoints{i}(1,:);
                dx = wp1(1)-px; dy = wp1(2)-py; dz = wp1(3)-pz;
                psi_i = atan2(dy, dx);
                gam_i = atan2(dz, sqrt(dx^2 + dy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'pinwheel')
        % --- Perpendicular stream ICs:
        %     east group starts at x=-R_ic, north group at y=-R_ic.
        %     Headings are exact (east=0, north=pi/2) so there is no turning delay.
        R_ic = 20 * params.r_cbf;
        na_A = floor(na/2);
        na_B = na - na_A;
        y_A  = linspace(-R_ic/2, R_ic/2, max(na_A, 1));
        x_B  = linspace(-R_ic/2, R_ic/2, max(na_B, 1));
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na_A
                params.x0(i,:)      = [-R_ic, y_A(i), 0,    params.v_desired(i)];  % heading east
            end
            for i = 1:na_B
                params.x0(na_A+i,:) = [x_B(i), -R_ic, pi/2, params.v_desired(na_A+i)];  % heading north
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na_A
                params.x0(i,:)      = [-R_ic, y_A(i), 10, 0,    0, params.v_desired(i)];
            end
            for i = 1:na_B
                params.x0(na_A+i,:) = [x_B(i), -R_ic, 10, pi/2, 0, params.v_desired(na_A+i)];
            end
        end
    elseif strcmp(params.mission, 'gauntlet')
        % --- Gauntlet ICs: wall at x=0 (slow), runners at x=-R_ic (fast) ---
        R_ic    = 20 * params.r_cbf;
        na_wall = floor(na/2);
        na_run  = na - na_wall;
        wall_spacing = 3 * params.r_cbf;
        wall_spread  = wall_spacing * (na_wall - 1) / 2;

        y_wall = linspace(-wall_spread, wall_spread, max(na_wall, 1)) + params.r_cbf;
        y_run  = linspace(-wall_spread, wall_spread, max(na_run,  1)) - params.r_cbf;

        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na_wall
                params.x0(i,:) = [0, y_wall(i), 0, params.v_desired(i)];
            end
            for i = 1:na_run
                params.x0(na_wall+i,:) = [-R_ic, y_run(i), 0, params.v_desired(na_wall+i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na_wall
                params.x0(i,:) = [0, y_wall(i), 10 + params.r_cbf, 0, 0, params.v_desired(i)];
            end
            for i = 1:na_run
                params.x0(na_wall+i,:) = [-R_ic, y_run(i), 10 - params.r_cbf, 0, 0, params.v_desired(na_wall+i)];
            end
        end

    elseif strcmp(params.mission, 'random_sphere_pursuit')
        % --- Random-sphere-pursuit ICs ---
        %   Evaders: same as random_sphere — sphere_ic point heading toward
        %   their own waypoint 1 (antipodal point).
        %   Pursuers: sphere_ic point heading toward the origin.  The
        %   sticky-targeting block in the main loop will rewrite each
        %   pursuer's waypoint on step 1, so this initial heading is just
        %   a sensible starting orientation.
        n_eva = sum(params.evader_mask);
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                if i <= n_eva
                    wp1 = params.waypoints{i}(1,:);
                    th  = atan2(wp1(2) - py, wp1(1) - px);
                else
                    th  = atan2(-py, -px);   % toward origin
                end
                params.x0(i,:) = [px, py, th, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                pz = params.sphere_ic(i, 3);
                if i <= n_eva
                    wp1 = params.waypoints{i}(1,:);
                    dx = wp1(1) - px;  dy = wp1(2) - py;  dz = wp1(3) - pz;
                else
                    dx = 0 - px;       dy = 0 - py;       dz = 0 - pz;
                end
                psi_i = atan2(dy, dx);
                gam_i = atan2(dz, sqrt(dx^2 + dy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'random_sphere')
        % --- Random-sphere ICs: each agent starts on the sphere surface,
        %     heading toward its first waypoint (the antipodal point).
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                wp1 = params.waypoints{i}(1,:);
                theta_i = atan2(wp1(2) - py, wp1(1) - px);
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                pz = params.sphere_ic(i, 3);
                wp1 = params.waypoints{i}(1,:);
                dx = wp1(1)-px; dy = wp1(2)-py; dz = wp1(3)-pz;
                psi_i = atan2(dy, dx);
                gam_i = atan2(dz, sqrt(dx^2 + dy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'head_on_ping_pong')
        % --- Head-on ping-pong ICs: same as random_sphere IC logic.
        %     Each agent starts on the sphere heading toward its first WP.
        if params.dim == 2
            params.x0 = zeros(na, 4);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                wp1 = params.waypoints{i}(1,:);
                theta_i = atan2(wp1(2) - py, wp1(1) - px);
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            for i = 1:na
                px = params.sphere_ic(i, 1);
                py = params.sphere_ic(i, 2);
                pz = params.sphere_ic(i, 3);
                wp1 = params.waypoints{i}(1,:);
                dx = wp1(1)-px; dy = wp1(2)-py; dz = wp1(3)-pz;
                psi_i = atan2(dy, dx);
                gam_i = atan2(dz, sqrt(dx^2 + dy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'pursuit')
        % --- Pursuit ICs: Agent 1 at origin heading toward its waypoint.
        %     Agents 2-N on a hemisphere *behind* agent 1 (negative x),
        %     at distance 10*v_evader, random angular spread, heading
        %     toward agent 1.
        R_ic = 10 * params.v_desired(1);   % use evader speed as length scale
        if params.dim == 2
            params.x0 = zeros(na, 4);
            % Agent 1: origin, heading +x toward waypoint
            wp1 = params.waypoints{1}(1,:);
            theta1 = atan2(wp1(2), wp1(1));
            params.x0(1,:) = [0, 0, theta1, params.v_desired(1)];
            % Pursuers: hemisphere behind agent 1 (x < 0)
            for i = 2:na
                % Angle in [-pi/2, pi/2] centered on -x direction (pi)
                ang = pi + (rand - 0.5) * pi;   % range [pi/2, 3*pi/2]
                px = R_ic * cos(ang);
                py = R_ic * sin(ang);
                theta_i = atan2(0 - py, 0 - px);  % heading toward origin (agent 1)
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            % Agent 1: origin at base altitude, heading +x
            wp1 = params.waypoints{1}(1,:);
            dx = wp1(1); dy = wp1(2); dz = wp1(3) - 10;
            psi1 = atan2(dy, dx);
            gam1 = atan2(dz, sqrt(dx^2 + dy^2));
            params.x0(1,:) = [0, 0, 10, psi1, gam1, params.v_desired(1)];
            % Pursuers: hemisphere behind agent 1 (x < 0)
            for i = 2:na
                % Random point on hemisphere behind evader
                az = pi + (rand - 0.5) * pi;          % azimuth in [pi/2, 3*pi/2]
                el = (rand - 0.5) * pi/2;              % elevation in [-pi/4, pi/4]
                px = R_ic * cos(el) * cos(az);
                py = R_ic * cos(el) * sin(az);
                pz = 10 + R_ic * sin(el);
                % Head toward agent 1 (origin, alt 10)
                ddx = 0 - px; ddy = 0 - py; ddz = 10 - pz;
                psi_i = atan2(ddy, ddx);
                gam_i = atan2(ddz, sqrt(ddx^2 + ddy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'team_pursuit')
        % --- Team-pursuit ICs ---
        %   Evaders (agents 1..n_eva) on a centered grid pointed at the
        %   shared first waypoint (matches waypoints_random_sync layout).
        %   Pursuers (agents n_eva+1..na) on a hemisphere behind the
        %   evader cluster (negative-x side), each heading toward the
        %   evader cluster centroid (origin).
        n_eva = sum(params.evader_mask);
        n_pur = na - n_eva;
        grid_spacing = 5 * params.r_cbf;
        R_pur = 10 * params.v_desired(1);     % pursuer distance scale
        wp1   = params.waypoints{1}(1,:);

        if params.dim == 2
            params.x0 = zeros(na, 4);
            % Evaders on a grid pointed toward wp1
            n_side = ceil(sqrt(n_eva));
            theta_nom = atan2(wp1(2), wp1(1));
            for i = 1:n_eva
                row = floor((i-1) / n_side);
                col = mod(i-1, n_side);
                px = (col - (n_side-1)/2) * grid_spacing;
                py = (row - (n_side-1)/2) * grid_spacing;
                params.x0(i,:) = [px, py, theta_nom, params.v_desired(i)];
            end
            % Pursuers on hemisphere behind (-x), heading toward origin
            for i = (n_eva+1):na
                ang = pi + (rand - 0.5) * pi;       % [pi/2, 3*pi/2]
                px = R_pur * cos(ang);
                py = R_pur * sin(ang);
                theta_i = atan2(0 - py, 0 - px);
                params.x0(i,:) = [px, py, theta_i, params.v_desired(i)];
            end
        else
            params.x0 = zeros(na, 6);
            n_side = ceil(n_eva^(1/3));
            dx = wp1(1); dy = wp1(2); dz = wp1(3) - 10;
            psi_nom = atan2(dy, dx);
            gam_nom = atan2(dz, sqrt(dx^2 + dy^2));
            for i = 1:n_eva
                idx = i - 1;
                cx = mod(idx, n_side);
                cy = mod(floor(idx / n_side), n_side);
                cz = floor(idx / (n_side^2));
                px = (cx - (n_side-1)/2) * grid_spacing;
                py = (cy - (n_side-1)/2) * grid_spacing;
                pz = 10 + (cz - (n_side-1)/2) * grid_spacing;
                params.x0(i,:) = [px, py, pz, psi_nom, gam_nom, params.v_desired(i)];
            end
            % Pursuers on hemisphere behind, alt ~10 km
            for i = (n_eva+1):na
                az = pi + (rand - 0.5) * pi;
                el = (rand - 0.5) * pi/2;
                px = R_pur * cos(el) * cos(az);
                py = R_pur * cos(el) * sin(az);
                pz = 10 + R_pur * sin(el);
                ddx = 0 - px; ddy = 0 - py; ddz = 10 - pz;
                psi_i = atan2(ddy, ddx);
                gam_i = atan2(ddz, sqrt(ddx^2 + ddy^2));
                params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
            end
        end

    elseif strcmp(params.mission, 'formation') || ...
           strcmp(params.mission, 'formation_pursuit') || ...
           strcmp(params.mission, 'formation_pursuit_vis')
        % --- Formation flight ICs ---
        %   Leader (agent 1) at the origin (z_base in 3D), heading toward
        %   first waypoint.  Followers (agents 2..n_eva) at body-frame V
        %   slots evaluated at the leader's IC pose so the team starts
        %   already in formation.  Pursuers (formation_pursuit only) on a
        %   hemisphere behind the formation, heading toward the formation
        %   centroid (origin), exactly as in team_pursuit.
        n_eva = sum(params.evader_mask);
        n_pur = na - n_eva;
        d_form = params.formation_spacing;
        alpha  = params.formation_stagger_angle;
        wp1    = params.waypoints{1}(1,:);

        if params.dim == 2
            params.x0 = zeros(na, 4);
            psi_lead = atan2(wp1(2), wp1(1));
            cps = cos(psi_lead);  sps = sin(psi_lead);
            body_x = [cps, sps];
            body_y = [-sps, cps];

            % Leader at origin
            params.x0(1,:) = [0, 0, psi_lead, params.v_desired(1)];

            % Followers at body-frame V slots (Convention A: agent 2 right,
            % agent 3 left, agent 4 right rank 2, ...)
            for i = 2:n_eva
                k_rk     = i - 1;
                rnk      = ceil(k_rk / 2);
                sd_side  = (-1)^k_rk;       % -1 = right wing, +1 = left wing
                                            % (NB: NOT 'sd' — sd = state_dim
                                            %  in outer scope.)
                ox       = -rnk * d_form * cos(alpha);
                oy       =  sd_side * rnk * d_form * sin(alpha);
                slot     = ox * body_x + oy * body_y;
                params.x0(i,:) = [slot, psi_lead, params.v_desired(i)];
            end

            % Pursuers on hemisphere behind, heading toward origin
            if n_pur > 0
                R_pur = 10 * params.v_desired(1);
                for i = (n_eva+1):na
                    ang = pi + (rand - 0.5) * pi;     % [pi/2, 3*pi/2]
                    px  = R_pur * cos(ang);
                    py  = R_pur * sin(ang);
                    th  = atan2(0 - py, 0 - px);
                    params.x0(i,:) = [px, py, th, params.v_desired(i)];
                end
            end
        else
            params.x0 = zeros(na, 6);
            z_base = 10;
            dx = wp1(1);  dy = wp1(2);  dz = wp1(3) - z_base;
            psi_lead = atan2(dy, dx);
            gam_lead = atan2(dz, sqrt(dx^2 + dy^2));
            cps = cos(psi_lead);  sps = sin(psi_lead);
            cga = cos(gam_lead);  sga = sin(gam_lead);
            % Body axes (yaw + pitch, no roll).  Slots tilt with leader's
            % flight-path angle: leader pitching up -> followers behind-
            % and-below.
            body_x = [cga*cps, cga*sps, sga];
            body_y = [-sps,     cps,     0];

            % Leader at [0, 0, z_base]
            params.x0(1,:) = [0, 0, z_base, psi_lead, gam_lead, params.v_desired(1)];

            % Followers at body-frame V slots (cache slots so vis-mission
            % can locate the two rearmost formation members).
            follower_slots = zeros(max(0, n_eva-1), 3);
            for i = 2:n_eva
                k_rk    = i - 1;
                rnk     = ceil(k_rk / 2);
                sd_side = (-1)^k_rk;        % NB: NOT 'sd' — sd = state_dim
                                            % in outer scope.
                ox      = -rnk * d_form * cos(alpha);
                oy      =  sd_side * rnk * d_form * sin(alpha);
                slot    = [0, 0, z_base] + ox * body_x + oy * body_y;
                follower_slots(i-1, :) = slot;
                params.x0(i,:) = [slot, psi_lead, gam_lead, params.v_desired(i)];
            end

            % Pursuers on hemisphere behind (formation_pursuit* only)
            if n_pur > 0
                if strcmp(params.mission, 'formation_pursuit_vis')
                    % Vis mission: Fibonacci-spiral distribution on the rear
                    % hemisphere centered at the midpoint of the two rearmost
                    % formation members.  Radius v_max*10.
                    R_pur = 10 * params.v_max(n_eva+1);
                    if n_eva - 1 >= 2
                        bxs = follower_slots * body_x';   % body-x projection
                        [~, ord] = sort(bxs, 'ascend');   % most-negative first
                        rear1 = follower_slots(ord(1), :);
                        rear2 = follower_slots(ord(2), :);
                        hemi_center = 0.5 * (rear1 + rear2);
                    elseif n_eva - 1 == 1
                        hemi_center = follower_slots(1, :);
                    else
                        hemi_center = [0, 0, z_base];
                    end
                    rear_dir = -body_x;
                    world_up = [0, 0, 1];
                    side_dir = cross(rear_dir, world_up);
                    if norm(side_dir) < 1e-9
                        side_dir = [0, 1, 0];
                    else
                        side_dir = side_dir / norm(side_dir);
                    end
                    up_dir   = cross(side_dir, rear_dir);
                    up_dir   = up_dir / max(1e-9, norm(up_dir));
                    phi_gr   = pi * (3 - sqrt(5));
                    for i = (n_eva+1):na
                        j       = i - n_eva;
                        z_local = 1 - (j - 0.5) / n_pur;    % (0, 1)
                        r_ring  = sqrt(max(0, 1 - z_local^2));
                        theta   = phi_gr * j;
                        dir_hat = z_local * rear_dir + ...
                                  r_ring * (cos(theta) * side_dir + ...
                                            sin(theta) * up_dir);
                        dir_hat = dir_hat / max(1e-9, norm(dir_hat));
                        p_p = hemi_center + R_pur * dir_hat;
                        ddx = hemi_center(1) - p_p(1);
                        ddy = hemi_center(2) - p_p(2);
                        ddz = hemi_center(3) - p_p(3);
                        psi_i = atan2(ddy, ddx);
                        gam_i = atan2(ddz, sqrt(ddx^2 + ddy^2));
                        params.x0(i,:) = [p_p, psi_i, gam_i, params.v_desired(i)];
                    end
                else
                    % Original formation_pursuit: random hemisphere behind origin
                    R_pur = 10 * params.v_desired(1);
                    for i = (n_eva+1):na
                        az = pi + (rand - 0.5) * pi;
                        el = (rand - 0.5) * pi/2;
                        px = R_pur * cos(el) * cos(az);
                        py = R_pur * cos(el) * sin(az);
                        pz = z_base + R_pur * sin(el);
                        ddx = 0 - px;  ddy = 0 - py;  ddz = z_base - pz;
                        psi_i = atan2(ddy, ddx);
                        gam_i = atan2(ddz, sqrt(ddx^2 + ddy^2));
                        params.x0(i,:) = [px, py, pz, psi_i, gam_i, params.v_desired(i)];
                    end
                end
            end
        end

    else
        % --- Grid-based ICs: agents on a grid centered at the origin,
        %     spacing 5*r so all pairs start at least 5*r apart.
        %     Headings point toward the first waypoint.
        grid_spacing = 5 * params.r_cbf;
        wp1 = params.waypoints{1}(1,:);  % shared first waypoint

        if params.dim == 2
            n_side = ceil(sqrt(na));     % grid side length
            params.x0 = zeros(na, 4);
            theta_nom = atan2(wp1(2), wp1(1));
            for i = 1:na
                row = floor((i-1) / n_side);
                col = mod(i-1, n_side);
                px = (col - (n_side-1)/2) * grid_spacing;
                py = (row - (n_side-1)/2) * grid_spacing;
                params.x0(i,:) = [px, py, theta_nom, params.v_desired(i)];
            end
        else
            n_side = ceil(na^(1/3));     % cube side length
            params.x0 = zeros(na, 6);
            dx = wp1(1); dy = wp1(2); dz = wp1(3) - 10;  % 10 km base altitude
            psi_nom = atan2(dy, dx);
            gam_nom = atan2(dz, sqrt(dx^2 + dy^2));
            for i = 1:na
                idx = i - 1;
                cx = mod(idx, n_side);
                cy = mod(floor(idx / n_side), n_side);
                cz = floor(idx / n_side^2);
                px = (cx - (n_side-1)/2) * grid_spacing;
                py = (cy - (n_side-1)/2) * grid_spacing;
                pz = 10 + (cz - (n_side-1)/2) * grid_spacing;  % 10 km base altitude
                params.x0(i,:) = [px, py, pz, psi_nom, gam_nom, params.v_desired(i)];
            end
        end
    end
end

% Default policy: proportional waypoint tracking
if isempty(params.policy)
    params.policy = @(x_all, id, t, p) waypoint_policy(x_all, id, t, p);
end

% Sticky pursuer-target assignment (used by 'pursuit', 'team_pursuit',
% 'formation_pursuit').  Each pursuer is initially assigned a uniform-random
% evader; reassignment happens at runtime when a pursuer reaches its target.
% Stored as a length-na vector with 0 for non-pursuers.
params.pursuer_target = zeros(na, 1);
if params.pursuit_mode
    if isfield(params, 'evader_mask')
        ev_ids_init = find(params.evader_mask);
        pu_ids_init = find(~params.evader_mask);
    else
        ev_ids_init = 1;            % plain pursuit: single evader
        pu_ids_init = (2:na)';
    end
    n_ev_init = numel(ev_ids_init);
    for pi_ = pu_ids_init(:)'
        params.pursuer_target(pi_) = ev_ids_init(randi(n_ev_init));
    end
end

%% ==================== PREALLOCATE ====================
N_steps   = ceil(params.T_sim / params.dt);
num_pairs = na*(na-1)/2;

t_hist    = zeros(N_steps+1, 1);
x_hist    = zeros(N_steps+1, na, sd);
u_hist    = zeros(N_steps,   na, cd);
% Pre-CBF nominal and post-QP/pre-clamp controls.  Saved unconditionally
% so plotters / diagnostics work for non-pursuit missions too (e.g.
% random_sphere, formation, waypoints_*).
u_nom_hist  = zeros(N_steps, na, cd);
u_safe_hist = zeros(N_steps, na, cd);
if params.compute_ttc
    ttc_hist   = Inf * ones(N_steps+1,   num_pairs);
    sttc1_hist = Inf * ones(N_steps+1,   num_pairs);  % 1-sigma STTC
    sttc2_hist = Inf * ones(N_steps+1,   num_pairs);  % 2-sigma STTC
    ttc2_hist      = Inf * ones(N_steps+1, num_pairs);  % 2nd-order TTC
    sttc2_1sig_hist = Inf * ones(N_steps+1, num_pairs); % 2nd-order 1-sigma STTC
    sttc2_2sig_hist = Inf * ones(N_steps+1, num_pairs); % 2nd-order 2-sigma STTC
end
if params.compute_attc
    attc_hist       = Inf * ones(N_steps+1, num_pairs); % adversarial TTC (forward sim)
end
dist_hist  = zeros(N_steps+1, num_pairs);
h_hist     = zeros(N_steps+1, num_pairs);

% Covariance per agent (for chance constraint / uncertainty tracking)
Sigma = cell(na, 1);
for i = 1:na, Sigma{i} = zeros(sd); end
sigma_pos_hist = zeros(N_steps+1, na);       % position std magnitude
r_eff_hist     = params.r_cbf * ones(N_steps+1, num_pairs);  % effective radius

% Initial state
x_hist(1,:,:) = params.x0;
t_hist(1)     = 0;
x0_mat = reshape(x_hist(1,:,:), na, sd);

% sHOCBF: fixed k for all pairs
if params.use_cbf && strcmp(params.cbf_type, 'shocbf')
    params.scbf_k = 0.5 * ones(na, na);
    fprintf('  sHOCBF: k fixed at %.3f for all pairs\n', 0.5);
end

% saTTC-vraw CBF: fixed k for all pairs
if params.use_cbf && strcmp(params.cbf_type, 'sattc_vraw')
    params.scbf_k = 0.5 * ones(na, na);
    fprintf('  %s-CBF: k fixed at %.3f for all pairs\n', ...
        upper(params.cbf_type), 0.5);
end

[ttc_tmp, dist_hist(1,:), h_hist(1,:)] = compute_all_pairs(x0_mat, params);
u_zero = zeros(na, cd);  % no control at t=0
if params.compute_ttc
    ttc_hist(1,:) = ttc_tmp;
    [sttc1_hist(1,:), sttc2_hist(1,:)] = compute_all_sttc(x0_mat, params, Sigma);
    ttc2_hist(1,:) = compute_all_ttc2(x0_mat, u_zero, params);
    [sttc2_1sig_hist(1,:), sttc2_2sig_hist(1,:)] = ...
        compute_all_sttc2(x0_mat, u_zero, params, Sigma);
end
if params.compute_attc
    attc_hist(1,:) = compute_all_attc(x0_mat, params);
end

%% ==================== MAIN SIMULATION LOOP ====================
% Progress: GUI waitbar if display available, text otherwise
use_text_progress = ~usejava('desktop');
if ~use_text_progress
    try
        wb = waitbar(0, 'Simulating...', 'Name', 'AUSPICE Sim');
    catch
        use_text_progress = true;
    end
end
if use_text_progress
    fprintf('  Progress:   0%%');
    last_pct = 0;
end
for k = 1:N_steps
    if ~use_text_progress
        if mod(k, 100) == 0 || k == N_steps
            waitbar(k / N_steps, wb, sprintf('Simulating... %d / %d  (%.0f%%)', k, N_steps, 100*k/N_steps));
        end
    else
        pct = floor(100 * k / N_steps);
        if pct >= last_pct + 5 || k == N_steps
            fprintf('\b\b\b\b%3d%%', pct);
            last_pct = pct;
        end
        if k == N_steps
            fprintf('\n');
        end
    end
    t      = (k-1) * params.dt;
    x_curr = reshape(x_hist(k,:,:), na, sd);

    % --- Pursuit mode: update pursuer waypoints to fly *through* their target ---
    %     Sticky targeting: each pursuer chases the evader stored in
    %     params.pursuer_target(i).  When that pursuer comes within
    %     capture_dist of its target (= "collision"), pick a new uniform-
    %     random evader other than the just-caught one (if n_ev >= 2).
    %     Plain 'pursuit' has only one evader — re-pick is a no-op.
    if params.pursuit_mode
        if isfield(params, 'evader_mask')
            evader_ids  = find(params.evader_mask);
            pursuer_ids = find(~params.evader_mask);
        else
            evader_ids  = 1;            % plain pursuit: single evader (agent 1)
            pursuer_ids = 2:na;
        end
        evader_ids  = evader_ids(:);    % force column for indexing
        pursuer_ids = pursuer_ids(:)';  % force row for the for-loop
        pdim = params.dim;
        wp_lead = 100 * max(params.v_max);
        n_ev = numel(evader_ids);

        % Capture / collision threshold (matches the post-hoc analysis
        % convention: 2*r_col where r_col = r_ttc if positive else r_cbf).
        if isfield(params, 'r_ttc') && params.r_ttc > 0
            r_col_pur = params.r_ttc;
        else
            r_col_pur = params.r_cbf;
        end
        capture_dist = 2 * r_col_pur;

        for i = pursuer_ids
            pos_i = x_curr(i, 1:pdim);
            tgt   = params.pursuer_target(i);
            % Defensive: if target somehow unset or invalid, pick one now.
            if tgt < 1 || tgt > na || ~ismember(tgt, evader_ids)
                tgt = evader_ids(randi(n_ev));
                params.pursuer_target(i) = tgt;
            end
            pos_t = x_curr(tgt, 1:pdim);
            d_to_tgt = norm(pos_t - pos_i);

            % Capture event → re-pick a new random target excluding current.
            if d_to_tgt < capture_dist && n_ev > 1
                others = evader_ids(evader_ids ~= tgt);
                tgt    = others(randi(numel(others)));
                params.pursuer_target(i) = tgt;
                pos_t  = x_curr(tgt, 1:pdim);
            end

            dir_vec = pos_t - pos_i;
            d       = norm(dir_vec);
            if d > 1e-6
                dir_vec = dir_vec / d;
            else
                dir_vec = randn(1, pdim);
                dir_vec = dir_vec / norm(dir_vec);
            end
            params.waypoints{i}(1,:) = pos_t + wp_lead * dir_vec;
        end
    end

    % --- Formation flight: update follower slot waypoints from leader's pose ---
    %     Slot offset is built in body frame (yaw + pitch, no roll) and
    %     translated by leader's current world position.  Re-evaluated every
    %     step so the chevron rigidly follows the leader's flight path.
    if strcmp(params.mission, 'formation') || ...
       strcmp(params.mission, 'formation_pursuit') || ...
       strcmp(params.mission, 'formation_pursuit_vis')
        evader_ids = find(params.evader_mask);
        evader_ids = evader_ids(:);
        n_ev_f     = numel(evader_ids);
        if n_ev_f >= 2
            leader_id = evader_ids(1);
            pdim_f    = params.dim;
            pos_lead  = x_curr(leader_id, 1:pdim_f);
            d_form    = params.formation_spacing;
            alpha_f   = params.formation_stagger_angle;
            if pdim_f == 2
                psi_l = x_curr(leader_id, 3);
                cps = cos(psi_l);  sps = sin(psi_l);
                bx = [cps, sps];
                by = [-sps, cps];
            else
                psi_l = x_curr(leader_id, 4);
                gam_l = x_curr(leader_id, 5);
                cps = cos(psi_l);  sps = sin(psi_l);
                cga = cos(gam_l);  sga = sin(gam_l);
                bx = [cga*cps, cga*sps, sga];
                by = [-sps,    cps,     0];
            end
            for fi = 2:n_ev_f      % NB: NOT 'k' — outer step loop uses 'k'
                ag   = evader_ids(fi);
                k_rk = fi - 1;
                rnk  = ceil(k_rk / 2);
                sd_f = (-1)^k_rk;     % -1 = right wing, +1 = left wing
                ox   = -rnk * d_form * cos(alpha_f);
                oy   =  sd_f * rnk * d_form * sin(alpha_f);
                slot = pos_lead + ox * bx + oy * by;
                params.waypoints{ag}(1,:) = slot;
            end
        end
    end

    % --- Nominal controls ---
    u_nom = zeros(na, cd);
    for i = 1:na
        u_nom(i,:) = params.policy(x_curr, i, t, params);
    end
    u_nom_hist(k,:,:) = u_nom;

    % --- CBF safety filter (pass Sigma for chance constraint) ---
    if params.use_cbf
        u_applied = auspice_cbf(x_curr, u_nom, Sigma, params);
    else
        u_applied = u_nom;
    end

    % Snapshot post-QP / pre-clamp control so the plotter can decompose
    % (u_safe - u_nom = pure CBF effort) and (u_applied - u_safe = pure clamp).
    u_safe_hist(k,:,:) = u_applied;

    % --- Clamp controls ---
    u_applied = clamp_controls(u_applied, params);
    u_hist(k,:,:) = u_applied;

    % --- Euler integration ---
    x_next = zeros(na, sd);
    for i = 1:na
        dx = dynamics(x_curr(i,:)', u_applied(i,:)', params);
        x_next(i,:) = x_curr(i,:) + params.dt * dx';
    end

    % --- Additive process noise (Euler-Maruyama) ---
    for i = 1:na
        x_next(i,:) = x_next(i,:) + ...
            sqrt(params.dt) * params.noise_std .* randn(1, sd);
    end

    % --- Kalman filter measurement update ---
    if params.use_kalman_filter
        [x_next, Sigma] = auspice_kalman_filter(x_next, Sigma, params);
    end

    % --- Advance waypoint index if within capture radius ---
    if params.sync_waypoints
        % Synchronized: mark arrivals, advance all agents together once
        % every agent has reached its current waypoint.
        % team_pursuit: only EVADERS participate in the sync (pursuers'
        % "waypoints" are dummy targets that get rewritten each timestep).
        if isfield(params, 'evader_mask')
            sync_ids = find(params.evader_mask);
        else
            sync_ids = 1:na;
        end
        for i = sync_ids(:)'
            if ~params.wp_arrived(i)
                idx_wp = params.wp_idx(i);
                wps_i  = params.waypoints{i};
                if params.dim == 2
                    d_wp = norm(x_next(i,1:2) - wps_i(idx_wp,1:2));
                else
                    d_wp = norm(x_next(i,1:3) - wps_i(idx_wp,:));
                end
                if d_wp < params.wp_capture_radius
                    params.wp_arrived(i) = true;
                end
            end
        end
        if all(params.wp_arrived(sync_ids))
            n_wp_1 = size(params.waypoints{sync_ids(1)}, 1);
            new_idx = mod(params.wp_idx(sync_ids(1)), n_wp_1) + 1;
            params.wp_idx(sync_ids) = new_idx;
            params.wp_arrived(sync_ids) = false;
            % Native WP-progress tracking: every sync_id agent cleared a WP
            params.wp_advance_count(sync_ids) = ...
                params.wp_advance_count(sync_ids) + 1;
        end
    else
        % Independent: each agent advances on its own, looping back to WP 1.
        for i = 1:na
            idx_wp = params.wp_idx(i);
            wps_i  = params.waypoints{i};
            n_wp_i = size(wps_i, 1);
            if params.dim == 2
                d_wp = norm(x_next(i,1:2) - wps_i(idx_wp,1:2));
            else
                d_wp = norm(x_next(i,1:3) - wps_i(idx_wp,:));
            end
            if d_wp < params.wp_capture_radius
                params.wp_idx(i) = mod(idx_wp, n_wp_i) + 1;
                % Native WP-progress tracking: agent i cleared a WP
                params.wp_advance_count(i) = params.wp_advance_count(i) + 1;
            end
        end
    end

    % --- Propagate state covariance (linearized dynamics) ---
    if params.use_chance_constraint || params.use_kalman_filter
        Sigma = auspice_kf_predict(x_curr, Sigma, params);
    end

    % Clamp speed (v_max may be scalar or per-agent vector)
    v_max_vec = params.v_max(:);  % ensure column
    if params.dim == 2
        x_next(:,4) = max(0, min(v_max_vec, x_next(:,4)));
    else
        x_next(:,6) = max(0, min(v_max_vec, x_next(:,6)));
    end

    % Store state, TTC, distance, barrier
    t_hist(k+1)     = t + params.dt;
    x_hist(k+1,:,:) = x_next;
    [ttc_tmp, dist_hist(k+1,:), h_hist(k+1,:)] = ...
        compute_all_pairs(x_next, params);
    if params.compute_ttc
        ttc_hist(k+1,:) = ttc_tmp;
        [sttc1_hist(k+1,:), sttc2_hist(k+1,:)] = ...
            compute_all_sttc(x_next, params, Sigma);
        ttc2_hist(k+1,:) = compute_all_ttc2(x_next, u_applied, params);
        [sttc2_1sig_hist(k+1,:), sttc2_2sig_hist(k+1,:)] = ...
            compute_all_sttc2(x_next, u_applied, params, Sigma);
    end
    if params.compute_attc
        attc_hist(k+1,:) = compute_all_attc(x_next, params);
    end

    % Store uncertainty info
    if params.use_chance_constraint || params.use_kalman_filter
        pos_dim = min(params.dim, 3);
        for i = 1:na
            sigma_pos_hist(k+1,i) = sqrt(trace(Sigma{i}(1:pos_dim,1:pos_dim)));
        end
        if params.use_chance_constraint
            pair = 0;
            for i = 1:na
                for j = i+1:na
                    pair = pair + 1;
                    r_eff_hist(k+1,pair) = compute_r_eff(x_next, i, j, Sigma, params);
                end
            end
        end
    end
end
if ~use_text_progress
    close(wb);
end

%% ==================== PACKAGE RESULTS ====================
results.t         = t_hist;
results.x         = x_hist;
results.u         = u_hist;
if params.compute_ttc
    results.ttc       = ttc_hist;
    results.sttc1     = sttc1_hist;
    results.sttc2     = sttc2_hist;
    results.ttc2      = ttc2_hist;
    results.sttc2_1sig = sttc2_1sig_hist;
    results.sttc2_2sig = sttc2_2sig_hist;
end
if params.compute_attc
    results.attc       = attc_hist;
end
results.dist      = dist_hist;
results.h         = h_hist;
results.sigma_pos = sigma_pos_hist;
results.r_eff     = r_eff_hist;
results.u_nom  = u_nom_hist;
results.u_safe = u_safe_hist;
% Native waypoint-progress tracking: per-agent count of how many waypoints
% the agent cleared during the run.  For agents whose waypoint list is a
% single dummy row (pursuers, formation followers), this stays at 0.
results.wp_advance_count = params.wp_advance_count;
results.params    = params;

%% ==================== PLOT ====================

if params.compute_ttc
    fprintf('Simulation complete.  Min TTC = %.3f s | Min dist = %.3f km\n', ...
        min(ttc_hist(:)), min(dist_hist(:)));
else
    fprintf('Simulation complete.  Min dist = %.3f km\n', min(dist_hist(:)));
end
if params.use_cbf
    fprintf('CBF safety filter: ENABLED (%s)\n', upper(params.cbf_type));
else
    fprintf('CBF safety filter: DISABLED\n');
end
end % auspice_sim


%% ================================================================
%%                        DYNAMICS
%% ================================================================

function dxdt = dynamics(x, u, params)
    if params.dim == 2
        dxdt = dubins2d(x, u);
    else
        dxdt = dubins3d(x, u);
    end
end

function dxdt = dubins2d(x, u)
% x = [px; py; theta; v],  u = [omega; a]
    v     = x(4);
    theta = x(3);
    dxdt  = [v*cos(theta); v*sin(theta); u(1); u(2)];
end

function dxdt = dubins3d(x, u)
% x = [px; py; pz; psi; gamma; v],  u = [u_psi; u_gamma; a]
    v     = x(6);
    psi   = x(4);
    gamma = x(5);
    dxdt  = [v*cos(gamma)*cos(psi);
             v*cos(gamma)*sin(psi);
             v*sin(gamma);
             u(1);
             u(2);
             u(3)];   % v_dot = a (acceleration control)
end


%% ================================================================
%%                     WAYPOINT POLICY
%% ================================================================

function u = waypoint_policy(x_all, id, ~, params)
% Proximity-based proportional waypoint tracker.
%   Each agent pursues its current waypoint until within wp_capture_radius,
%   then advances to the next waypoint in the mission.
%   Steering uses proportional heading control; speed uses proportional accel.
    wps  = params.waypoints{id};       % [n_wp x 2] or [n_wp x 3]
    n_wp = size(wps, 1);

    % Current waypoint index (managed in main loop via proximity check)
    idx = min(params.wp_idx(id), n_wp);
    wp  = wps(idx, :);

    K_hdg = 2.5;    % heading gain
    K_spd = 1.5;    % speed gain

    if params.dim == 2
        px = x_all(id,1); py = x_all(id,2);
        theta = x_all(id,3); v = x_all(id,4);

        theta_des = atan2(wp(2)-py, wp(1)-px);
        e_theta   = mod(theta_des - theta + pi, 2*pi) - pi;   % wrap to [-pi,pi]

        omega = K_hdg * e_theta;
        a     = K_spd * (params.v_desired(id) - v);
        u     = [omega, a];
    else
        px = x_all(id,1); py = x_all(id,2); pz = x_all(id,3);
        psi = x_all(id,4); gamma = x_all(id,5); v = x_all(id,6);

        dx = wp(1)-px; dy = wp(2)-py; dz = wp(3)-pz;
        dh = sqrt(dx^2 + dy^2);

        psi_des   = atan2(dy, dx);
        gamma_des = atan2(dz, max(dh, 1e-6));

        e_psi   = mod(psi_des - psi + pi, 2*pi) - pi;
        e_gamma = gamma_des - gamma;

        u_psi   = K_hdg * e_psi;
        u_gamma = K_hdg * e_gamma;
        a       = K_spd * (params.v_desired(id) - v);
        u       = [u_psi, u_gamma, a];
    end
end

function u = waypoint_policy_2(x_all, id, t, params)
% Synchronized waypoint tracker.
%   Identical steering to waypoint_policy — the only difference is that
%   waypoint advancement is handled in the main loop (sync_waypoints mode),
%   so all agents wait at each waypoint until the group has arrived.
    u = waypoint_policy(x_all, id, t, params);
end


%% ================================================================
%%                     TTC COMPUTATION
%% ================================================================

function [ttc_vec, dist_vec, h_vec] = compute_all_pairs(x_curr, params)
% Evaluate TTC, distance, and barrier value for every agent pair.
    na        = params.num_agents;
    num_pairs = na*(na-1)/2;
    ttc_vec   = Inf * ones(1, num_pairs);
    dist_vec  = zeros(1, num_pairs);
    h_vec     = zeros(1, num_pairs);
    pair = 0;
    for i = 1:na
        for j = i+1:na
            pair = pair + 1;
            xi = x_curr(i,:);
            xj = x_curr(j,:);

            if params.dim == 2
                dp = xi(1:2) - xj(1:2);
            else
                dp = xi(1:3) - xj(1:3);
            end
            d = norm(dp);

            dist_vec(pair) = d;
            h_vec(pair)    = d^2 - (2*params.r_cbf)^2;
            ttc_vec(pair)  = compute_ttc_analytic(xi, xj, params);
        end
    end
end

function ttc = compute_ttc_analytic(xi, xj, params)
% Closed-form TTC assuming constant heading & speed (ballistic propagation).
%   Solves  ||dp + dv*tau||^2 = (2r)^2  for the smallest positive tau.
    if params.dim == 2
        dpx = xi(1) - xj(1);
        dpy = xi(2) - xj(2);
        dvx = xi(4)*cos(xi(3)) - xj(4)*cos(xj(3));
        dvy = xi(4)*sin(xi(3)) - xj(4)*sin(xj(3));
        c   = dpx^2 + dpy^2 - (2*params.r_ttc)^2;
        b   = 2*(dpx*dvx + dpy*dvy);
        a   = dvx^2 + dvy^2;
    else
        dpx = xi(1)-xj(1);  dpy = xi(2)-xj(2);  dpz = xi(3)-xj(3);
        vi  = xi(6);  vj = xj(6);
        dvx = vi*cos(xi(5))*cos(xi(4)) - vj*cos(xj(5))*cos(xj(4));
        dvy = vi*cos(xi(5))*sin(xi(4)) - vj*cos(xj(5))*sin(xj(4));
        dvz = vi*sin(xi(5))            - vj*sin(xj(5));
        c   = dpx^2 + dpy^2 + dpz^2 - (2*params.r_ttc)^2; % Distance (tau^2 term)
        b   = 2*(dpx*dvx + dpy*dvy + dpz*dvz); % Cross Term (tau term)
        a   = dvx^2 + dvy^2 + dvz^2; % Velocity Difference (constant term)
    end

    % Regularize: add epsilon to guarantee a > 0, bounding TTC
    a = a + params.epsilon_ttc;

    % Already inside collision zone
    if c <= 0
        ttc = 0;
        return;
    end

    % Parallel / same velocity (only reachable when epsilon_ttc ≈ 0)
    if abs(a) < 1e-12
        if b < -1e-12
            ttc = -c / b;   % linear approach
        else
            ttc = Inf;
        end
        return;
    end

    disc = b^2 - 4*a*c;
    if disc < 0              % implies complex Tau - which is unphysical
        ttc = Inf;           % trajectories never intersect
        return;
    end

    t1 = (-b - sqrt(disc)) / (2*a);
    if t1 > 1e-12
        ttc = t1;            % first future entry time
    else
        ttc = Inf;           % collision in the past; agents diverging
    end
end


%% ================================================================
%%                   STOCHASTIC TTC (STTC)
%% ================================================================
%
%   Combined velocity-perturbation + radius-inflation formulation.
%
%   Three independent uncertainty sources contribute:
%     1. Heading/speed noise (sigma_theta, sigma_v)  -->  velocity
%        uncertainty [km/s].  Propagated via the Jacobian d(vel)/d(state),
%        combined across both agents (RSS), then applied as a pessimistic
%        shift:  dv_adj(k) = dv(k) - sign(dp(k)) * n_sigma * sigma_dv(k)
%
%     2. Position noise (sigma_pos)  -->  Brownian position drift [km/sqrt(s)].
%        Inflates the collision radius over time:  kappa*sqrt(tau)
%        with kappa = n_sigma * sqrt(2) * ||sigma_pos||.
%
%     3. Kalman filter position covariance (Sigma_i, Sigma_j)  -->  current
%        estimation uncertainty [km].  Adds a constant radius inflation:
%        sig_kf = sqrt(trace(Sigma_pos_i) + trace(Sigma_pos_j))
%        When KF is off, Sigma = 0 and this term vanishes.
%
%   Collision criterion (with R0 = 2r + n_sigma*sig_kf):
%       ||dp + dv_adj*tau||  =  R0 + kappa*sqrt(tau)
%
%   Substituting u = sqrt(tau) and squaring yields a quartic in u:
%       a*u^4 + (b - kappa^2)*u^2 - 2*R0*kappa*u + (c - R0^2 + (2r)^2) = 0
%   where a = ||dv_adj||^2,  b = 2*dp.dv_adj,  c = ||dp||^2 - R0^2.

function [sttc1_vec, sttc2_vec] = compute_all_sttc(x_curr, params, Sigma)
% Compute 1-sigma and 2-sigma STTC for every agent pair.
    na        = params.num_agents;
    num_pairs = na*(na-1)/2;
    sttc1_vec = Inf * ones(1, num_pairs);
    sttc2_vec = Inf * ones(1, num_pairs);
    pair = 0;
    for i = 1:na
        for j = i+1:na
            pair = pair + 1;
            sttc1_vec(pair) = compute_sttc_analytic(x_curr(i,:), x_curr(j,:), params, 1, Sigma{i}, Sigma{j});
            sttc2_vec(pair) = compute_sttc_analytic(x_curr(i,:), x_curr(j,:), params, 2, Sigma{i}, Sigma{j});
        end
    end
end

function sttc = compute_sttc_analytic(xi, xj, params, n_sigma, Sigma_i, Sigma_j)
% Stochastic TTC via velocity perturbation + radius inflation.
%   Sigma_i, Sigma_j: state covariance matrices from Kalman filter.
%   When the KF is active, the position block adds a constant radius
%   inflation on top of the sqrt(tau) growth from ongoing process noise.
    if params.dim == 2
        pos_dim = 2;
        dp  = [xi(1)-xj(1), xi(2)-xj(2)];

        thi = xi(3); vi = xi(4);
        thj = xj(3); vj = xj(4);
        dv  = [vi*cos(thi) - vj*cos(thj), ...
               vi*sin(thi) - vj*sin(thj)];

        % --- Velocity uncertainty from heading/speed noise ---
        % Jacobian: d[vx;vy]/d[theta,v] per agent
        sig_th = params.noise_std(3);
        sig_v  = params.noise_std(4);
        % Agent i
        sig_vx_i2 = (vi*sin(thi))^2 * sig_th^2 + cos(thi)^2 * sig_v^2;
        sig_vy_i2 = (vi*cos(thi))^2 * sig_th^2 + sin(thi)^2 * sig_v^2;
        % Agent j
        sig_vx_j2 = (vj*sin(thj))^2 * sig_th^2 + cos(thj)^2 * sig_v^2;
        sig_vy_j2 = (vj*cos(thj))^2 * sig_th^2 + sin(thj)^2 * sig_v^2;
        % Relative velocity std per component (RSS of independent agents)
        sig_dv = [sqrt(sig_vx_i2 + sig_vx_j2), ...
                  sqrt(sig_vy_i2 + sig_vy_j2)];

        sigma_pos = params.noise_std(1:2);
    else
        pos_dim = 3;
        dp  = [xi(1)-xj(1), xi(2)-xj(2), xi(3)-xj(3)];

        psii = xi(4); gami = xi(5); vi = xi(6);
        psij = xj(4); gamj = xj(5); vj = xj(6);
        dv  = [vi*cos(gami)*cos(psii) - vj*cos(gamj)*cos(psij), ...
               vi*cos(gami)*sin(psii) - vj*cos(gamj)*sin(psij), ...
               vi*sin(gami)           - vj*sin(gamj)];

        % --- Velocity uncertainty from heading/speed noise ---
        % Jacobian: d[vx;vy;vz]/d[psi,gamma,v] per agent
        sig_psi = params.noise_std(4);
        sig_gam = params.noise_std(5);
        sig_v   = params.noise_std(6);
        % Agent i
        sig_vx_i2 = (vi*cos(gami)*sin(psii))^2 * sig_psi^2 ...
                   + (vi*sin(gami)*cos(psii))^2 * sig_gam^2 ...
                   + (cos(gami)*cos(psii))^2     * sig_v^2;
        sig_vy_i2 = (vi*cos(gami)*cos(psii))^2 * sig_psi^2 ...
                   + (vi*sin(gami)*sin(psii))^2 * sig_gam^2 ...
                   + (cos(gami)*sin(psii))^2     * sig_v^2;
        sig_vz_i2 = (vi*cos(gami))^2            * sig_gam^2 ...
                   + sin(gami)^2                  * sig_v^2;
        % Agent j
        sig_vx_j2 = (vj*cos(gamj)*sin(psij))^2 * sig_psi^2 ...
                   + (vj*sin(gamj)*cos(psij))^2 * sig_gam^2 ...
                   + (cos(gamj)*cos(psij))^2     * sig_v^2;
        sig_vy_j2 = (vj*cos(gamj)*cos(psij))^2 * sig_psi^2 ...
                   + (vj*sin(gamj)*sin(psij))^2 * sig_gam^2 ...
                   + (cos(gamj)*sin(psij))^2     * sig_v^2;
        sig_vz_j2 = (vj*cos(gamj))^2            * sig_gam^2 ...
                   + sin(gamj)^2                  * sig_v^2;
        % Relative velocity std per component
        sig_dv = [sqrt(sig_vx_i2 + sig_vx_j2), ...
                  sqrt(sig_vy_i2 + sig_vy_j2), ...
                  sqrt(sig_vz_i2 + sig_vz_j2)];

        sigma_pos = params.noise_std(1:3);
    end

    % --- Pessimistic velocity perturbation [km/s] ---
    dv_adj = dv - sign(dp) .* (n_sigma * sig_dv);

    % --- Constant radius inflation from KF position uncertainty [km] ---
    % Relative position covariance: Sigma_rel = Sigma_pos_i + Sigma_pos_j
    sig_kf = sqrt(trace(Sigma_i(1:pos_dim,1:pos_dim)) ...
                + trace(Sigma_j(1:pos_dim,1:pos_dim)));
    R0 = 2*params.r_ttc + n_sigma * sig_kf;

    % --- Position-noise radius inflation [km/sqrt(s)] ---
    kappa = n_sigma * sqrt(2) * norm(sigma_pos);

    % Quadratic coefficients with adjusted velocity
    c_q = dot(dp,dp) - R0^2;
    b_q = 2*dot(dp, dv_adj);
    a_q = dot(dv_adj, dv_adj) + params.epsilon_ttc;

    % Already inside stochastic collision zone
    if c_q <= 0
        sttc = 0;  return;
    end

    % Quartic in u = sqrt(tau):
    %   a_q*u^4 + 0*u^3 + (b_q - kappa^2)*u^2 - 2*R0*kappa*u + c_q = 0
    coeffs = [a_q, 0, b_q - kappa^2, -2*R0*kappa, c_q];

    r_roots = roots(coeffs);

    % Extract smallest positive real u, then tau = u^2
    sttc = Inf;
    for k = 1:length(r_roots)
        u = r_roots(k);
        if abs(imag(u)) > 1e-10, continue; end   % skip complex roots
        u = real(u);
        if u > 1e-6                               % u = sqrt(tau) > 0
            tau = u^2;
            if tau < sttc
                sttc = tau;
            end
        end
    end
end


%% ================================================================
%%            SECOND-ORDER TTC  (TTC2)
%% ================================================================
%
%   Uses a second-order Taylor expansion for position:
%       dp(tau) = dp + dv*tau + 0.5*da*tau^2
%
%   where da is the relative Cartesian acceleration derived from the
%   current control inputs (omega, a_cmd).  Collision criterion:
%       ||dp + dv*tau + 0.5*da*tau^2||^2 = (2r)^2
%
%   Expanding yields a quartic in tau:
%       (||da||^2/4)*tau^4 + (da.dv)*tau^3 + (||dv||^2 + dp.da)*tau^2
%       + 2*(dp.dv)*tau + (||dp||^2 - (2r)^2) = 0

function ttc2_vec = compute_all_ttc2(x_curr, u_curr, params)
% Compute second-order TTC for every agent pair.
    na        = params.num_agents;
    num_pairs = na*(na-1)/2;
    ttc2_vec  = Inf * ones(1, num_pairs);
    pair = 0;
    for i = 1:na
        for j = i+1:na
            pair = pair + 1;
            ttc2_vec(pair) = compute_ttc2_analytic(x_curr(i,:), x_curr(j,:), ...
                                                    u_curr(i,:), u_curr(j,:), params);
        end
    end
end

function ttc2 = compute_ttc2_analytic(xi, xj, ui, uj, params)
% Second-order TTC using quadratic position prediction.
%   ui, uj are the current control inputs for agents i and j.
    if params.dim == 2
        % Relative position
        dp  = [xi(1)-xj(1), xi(2)-xj(2)];
        % Relative Cartesian velocity
        thi = xi(3); vi = xi(4);
        thj = xj(3); vj = xj(4);
        dv  = [vi*cos(thi) - vj*cos(thj), ...
               vi*sin(thi) - vj*sin(thj)];
        % Cartesian acceleration: a = [a_cmd*cos(th) - v*omega*sin(th),
        %                              a_cmd*sin(th) + v*omega*cos(th)]
        omega_i = ui(1); a_cmd_i = ui(2);
        omega_j = uj(1); a_cmd_j = uj(2);
        ai = [a_cmd_i*cos(thi) - vi*omega_i*sin(thi), ...
              a_cmd_i*sin(thi) + vi*omega_i*cos(thi)];
        aj = [a_cmd_j*cos(thj) - vj*omega_j*sin(thj), ...
              a_cmd_j*sin(thj) + vj*omega_j*cos(thj)];
        da = ai - aj;
    else
        % Relative position
        dp  = [xi(1)-xj(1), xi(2)-xj(2), xi(3)-xj(3)];
        % Relative Cartesian velocity
        psii = xi(4); gami = xi(5); vi = xi(6);
        psij = xj(4); gamj = xj(5); vj = xj(6);
        dv  = [vi*cos(gami)*cos(psii) - vj*cos(gamj)*cos(psij), ...
               vi*cos(gami)*sin(psii) - vj*cos(gamj)*sin(psij), ...
               vi*sin(gami)           - vj*sin(gamj)];
        % Cartesian acceleration (constant speed model, dv/dt = 0):
        %   ax = -v*cos(g)*sin(p)*wp - v*sin(g)*cos(p)*wg
        %   ay =  v*cos(g)*cos(p)*wp - v*sin(g)*sin(p)*wg
        %   az =  v*cos(g)*wg
        wp_i = ui(1); wg_i = ui(2);   % psi/gamma rates
        wp_j = uj(1); wg_j = uj(2);
        ai = [-vi*cos(gami)*sin(psii)*wp_i - vi*sin(gami)*cos(psii)*wg_i, ...
               vi*cos(gami)*cos(psii)*wp_i - vi*sin(gami)*sin(psii)*wg_i, ...
               vi*cos(gami)*wg_i];
        aj = [-vj*cos(gamj)*sin(psij)*wp_j - vj*sin(gamj)*cos(psij)*wg_j, ...
               vj*cos(gamj)*cos(psij)*wp_j - vj*sin(gamj)*sin(psij)*wg_j, ...
               vj*cos(gamj)*wg_j];
        da = ai - aj;
    end

    % Quartic coefficients: (||da||^2/4)*t^4 + (da.dv)*t^3
    %   + (||dv||^2 + dp.da)*t^2 + 2*(dp.dv)*t + (||dp||^2 - (2r)^2) = 0
    a4 = dot(da,da) / 4;
    a3 = dot(da, dv);
    a2 = dot(dv,dv) + dot(dp, da);
    a1 = 2 * dot(dp, dv);
    a0 = dot(dp,dp) - (2*params.r_ttc)^2;

    % Already inside collision zone
    if a0 <= 0
        ttc2 = 0;  return;
    end

    % If acceleration is negligible, fall back to linear TTC
    if abs(a4) < 1e-14 && abs(a3) < 1e-14
        % Quadratic: a2*t^2 + a1*t + a0 = 0
        if abs(a2) < 1e-12
            if a1 < -1e-12
                ttc2 = -a0 / a1;
            else
                ttc2 = Inf;
            end
            return;
        end
        disc = a1^2 - 4*a2*a0;
        if disc < 0
            ttc2 = Inf; return;
        end
        t1 = (-a1 - sqrt(disc)) / (2*a2);
        if t1 > 1e-12
            ttc2 = t1;
        else
            ttc2 = Inf;
        end
        return;
    end

    % Solve quartic via companion matrix
    coeffs = [a4, a3, a2, a1, a0];
    r_roots = roots(coeffs);

    % Extract smallest positive real root
    ttc2 = Inf;
    for k = 1:length(r_roots)
        rt = r_roots(k);
        if abs(imag(rt)) > 1e-10, continue; end
        rt = real(rt);
        if rt > 1e-12 && rt < ttc2
            ttc2 = rt;
        end
    end
end


%% ================================================================
%%            SECOND-ORDER STOCHASTIC TTC  (STTC2)
%% ================================================================
%
%   Second-order position prediction with the same three uncertainty
%   sources as STTC (velocity perturbation, KF radius inflation,
%   process-noise radius growth):
%
%       dp(tau) = dp + dv_adj*tau + 0.5*da*tau^2
%
%   Collision criterion:
%       ||dp + dv_adj*tau + 0.5*da*tau^2||  =  R0 + kappa*sqrt(tau)
%
%   Substituting u = sqrt(tau), tau = u^2, the LHS squared becomes a
%   degree-8 polynomial in u.  The full equation is:
%
%       (||da||^2/4)*u^8 + (da.dv_adj)*u^6 + (||dv_adj||^2 + dp.da)*u^4
%       + 2*(dp.dv_adj)*u^2 + ||dp||^2
%       = R0^2 + 2*R0*kappa*u + kappa^2*u^2
%
%   Rearranged as a degree-8 polynomial in u:
%       c8*u^8 + c6*u^6 + c4*u^4 + c2*u^2 - kappa^2*u^2
%       - 2*R0*kappa*u + (||dp||^2 - R0^2) = 0

function [s1_vec, s2_vec] = compute_all_sttc2(x_curr, u_curr, params, Sigma)
% Compute 1-sigma and 2-sigma second-order STTC for every agent pair.
    na        = params.num_agents;
    num_pairs = na*(na-1)/2;
    s1_vec    = Inf * ones(1, num_pairs);
    s2_vec    = Inf * ones(1, num_pairs);
    pair = 0;
    for i = 1:na
        for j = i+1:na
            pair = pair + 1;
            s1_vec(pair) = compute_sttc2_analytic(x_curr(i,:), x_curr(j,:), ...
                u_curr(i,:), u_curr(j,:), params, 1, Sigma{i}, Sigma{j});
            s2_vec(pair) = compute_sttc2_analytic(x_curr(i,:), x_curr(j,:), ...
                u_curr(i,:), u_curr(j,:), params, 2, Sigma{i}, Sigma{j});
        end
    end
end

function sttc2 = compute_sttc2_analytic(xi, xj, ui, uj, params, n_sigma, Sigma_i, Sigma_j)
% Second-order STTC via quadratic position prediction + velocity
% perturbation + radius inflation.
    if params.dim == 2
        pos_dim = 2;
        dp  = [xi(1)-xj(1), xi(2)-xj(2)];

        thi = xi(3); vi = xi(4);
        thj = xj(3); vj = xj(4);
        dv  = [vi*cos(thi) - vj*cos(thj), ...
               vi*sin(thi) - vj*sin(thj)];

        % Cartesian acceleration from current controls
        omega_i = ui(1); a_cmd_i = ui(2);
        omega_j = uj(1); a_cmd_j = uj(2);
        ai = [a_cmd_i*cos(thi) - vi*omega_i*sin(thi), ...
              a_cmd_i*sin(thi) + vi*omega_i*cos(thi)];
        aj = [a_cmd_j*cos(thj) - vj*omega_j*sin(thj), ...
              a_cmd_j*sin(thj) + vj*omega_j*cos(thj)];
        da = ai - aj;

        % Velocity uncertainty from heading/speed noise (Jacobian)
        sig_th = params.noise_std(3);
        sig_v  = params.noise_std(4);
        sig_vx_i2 = (vi*sin(thi))^2 * sig_th^2 + cos(thi)^2 * sig_v^2;
        sig_vy_i2 = (vi*cos(thi))^2 * sig_th^2 + sin(thi)^2 * sig_v^2;
        sig_vx_j2 = (vj*sin(thj))^2 * sig_th^2 + cos(thj)^2 * sig_v^2;
        sig_vy_j2 = (vj*cos(thj))^2 * sig_th^2 + sin(thj)^2 * sig_v^2;
        sig_dv = [sqrt(sig_vx_i2 + sig_vx_j2), ...
                  sqrt(sig_vy_i2 + sig_vy_j2)];

        sigma_pos = params.noise_std(1:2);
    else
        pos_dim = 3;
        dp  = [xi(1)-xj(1), xi(2)-xj(2), xi(3)-xj(3)];

        psii = xi(4); gami = xi(5); vi = xi(6);
        psij = xj(4); gamj = xj(5); vj = xj(6);
        dv  = [vi*cos(gami)*cos(psii) - vj*cos(gamj)*cos(psij), ...
               vi*cos(gami)*sin(psii) - vj*cos(gamj)*sin(psij), ...
               vi*sin(gami)           - vj*sin(gamj)];

        % Cartesian acceleration from current controls
        wp_i = ui(1); wg_i = ui(2);
        wp_j = uj(1); wg_j = uj(2);
        ai = [-vi*cos(gami)*sin(psii)*wp_i - vi*sin(gami)*cos(psii)*wg_i, ...
               vi*cos(gami)*cos(psii)*wp_i - vi*sin(gami)*sin(psii)*wg_i, ...
               vi*cos(gami)*wg_i];
        aj = [-vj*cos(gamj)*sin(psij)*wp_j - vj*sin(gamj)*cos(psij)*wg_j, ...
               vj*cos(gamj)*cos(psij)*wp_j - vj*sin(gamj)*sin(psij)*wg_j, ...
               vj*cos(gamj)*wg_j];
        da = ai - aj;

        % Velocity uncertainty from heading/speed noise (Jacobian)
        sig_psi = params.noise_std(4);
        sig_gam = params.noise_std(5);
        sig_v   = params.noise_std(6);
        sig_vx_i2 = (vi*cos(gami)*sin(psii))^2 * sig_psi^2 ...
                   + (vi*sin(gami)*cos(psii))^2 * sig_gam^2 ...
                   + (cos(gami)*cos(psii))^2     * sig_v^2;
        sig_vy_i2 = (vi*cos(gami)*cos(psii))^2 * sig_psi^2 ...
                   + (vi*sin(gami)*sin(psii))^2 * sig_gam^2 ...
                   + (cos(gami)*sin(psii))^2     * sig_v^2;
        sig_vz_i2 = (vi*cos(gami))^2            * sig_gam^2 ...
                   + sin(gami)^2                  * sig_v^2;
        sig_vx_j2 = (vj*cos(gamj)*sin(psij))^2 * sig_psi^2 ...
                   + (vj*sin(gamj)*cos(psij))^2 * sig_gam^2 ...
                   + (cos(gamj)*cos(psij))^2     * sig_v^2;
        sig_vy_j2 = (vj*cos(gamj)*cos(psij))^2 * sig_psi^2 ...
                   + (vj*sin(gamj)*sin(psij))^2 * sig_gam^2 ...
                   + (cos(gamj)*sin(psij))^2     * sig_v^2;
        sig_vz_j2 = (vj*cos(gamj))^2            * sig_gam^2 ...
                   + sin(gamj)^2                  * sig_v^2;
        sig_dv = [sqrt(sig_vx_i2 + sig_vx_j2), ...
                  sqrt(sig_vy_i2 + sig_vy_j2), ...
                  sqrt(sig_vz_i2 + sig_vz_j2)];

        sigma_pos = params.noise_std(1:3);
    end

    % --- Pessimistic velocity perturbation [km/s] ---
    dv_adj = dv - sign(dp) .* (n_sigma * sig_dv);

    % --- Constant radius inflation from KF position uncertainty [km] ---
    sig_kf = sqrt(trace(Sigma_i(1:pos_dim,1:pos_dim)) ...
                + trace(Sigma_j(1:pos_dim,1:pos_dim)));
    R0 = 2*params.r_ttc + n_sigma * sig_kf;

    % --- Position-noise radius inflation [km/sqrt(s)] ---
    kappa = n_sigma * sqrt(2) * norm(sigma_pos);

    % Coefficients for ||dp + dv_adj*tau + 0.5*da*tau^2||^2 expanded in tau:
    %   e4*tau^4 + e3*tau^3 + e2*tau^2 + e1*tau + e0
    e4 = dot(da,da) / 4;
    e3 = dot(da, dv_adj);
    e2 = dot(dv_adj, dv_adj) + dot(dp, da);
    e1 = 2 * dot(dp, dv_adj);
    e0 = dot(dp, dp);

    % Already inside stochastic collision zone
    if e0 - R0^2 <= 0
        sttc2 = 0;  return;
    end

    % Substitute u = sqrt(tau), tau = u^2:
    %   LHS = e4*u^8 + e3*u^6 + e2*u^4 + e1*u^2 + e0
    %   RHS = (R0 + kappa*u)^2 = R0^2 + 2*R0*kappa*u + kappa^2*u^2
    %
    % Polynomial in u (degree 8):
    %   e4*u^8 + 0*u^7 + e3*u^6 + 0*u^5 + e2*u^4 + 0*u^3
    %   + (e1 - kappa^2)*u^2 - 2*R0*kappa*u + (e0 - R0^2) = 0
    coeffs = [e4, 0, e3, 0, e2, 0, e1 - kappa^2, -2*R0*kappa, e0 - R0^2];

    r_roots = roots(coeffs);

    % Extract smallest positive real u, then tau = u^2
    sttc2 = Inf;
    for k = 1:length(r_roots)
        u = r_roots(k);
        if abs(imag(u)) > 1e-10, continue; end
        u = real(u);
        if u > 1e-6
            tau = u^2;
            if tau < sttc2
                sttc2 = tau;
            end
        end
    end
end



%% ================================================================
%%            ADVERSARIAL TTC  (aTTC)
%% ================================================================
%
%   Forward-simulate both agents with crash-seeking bang-bang controls
%   (max turn rate toward opponent, max acceleration in 2D) and return
%   the time until collision (distance <= 2*r_ttc).

function attc_vec = compute_all_attc(x_curr, params)
% Compute adversarial TTC for every agent pair.
%   Skips pairs separated by more than 2*v_max*attc_horizon (unreachable).
    na        = params.num_agents;
    num_pairs = na*(na-1)/2;
    attc_vec  = Inf * ones(1, num_pairs);

    % Distance gate: skip pairs that can't collide within horizon
    d_gate = 2 * max(params.v_max) * params.attc_horizon;

    pair = 0;
    for i = 1:na
        for j = i+1:na
            pair = pair + 1;
            xi = x_curr(i,:);
            xj = x_curr(j,:);

            if params.dim == 2
                dp = xi(1:2) - xj(1:2);
            else
                dp = xi(1:3) - xj(1:3);
            end

            if norm(dp) > d_gate
                continue;   % unreachable within horizon
            end

            attc_vec(pair) = compute_attc(x_curr, i, j, params);
        end
    end
end


function attc = compute_attc(x_curr, ego_id, other_id, params)
% Adversarial TTC for a single pair via forward simulation.
%
%   attc_type modes:
%     'bang'         : Both agents steer at max rate toward each other (bang-bang)
%                      and accelerate to max speed.
%     'prop'         : Both agents use proportional heading toward each other
%                      and accelerate to max speed.
%     'pursuit_ego'  : Agent i (ego) follows nominal waypoint control; agent j
%                      uses proportional heading toward ego + max acceleration.
%                      One direction only (ego_id = nominal).
%     'pursuit_min'  : Compute both pursuit directions, return min (worst case).
%
%   Returns elapsed time when dist <= 2*r_ttc, or Inf if horizon is reached.
    mode = '';
    if isfield(params, 'attc_type')
        mode = params.attc_type;
    end

    if strcmp(mode, 'pursuit_ego')
        % Asymmetric: ego = nominal, other = pursuer (one direction)
        attc = attc_pursuit_sim(x_curr, ego_id, other_id, params);
    elseif strcmp(mode, 'pursuit_min')
        % Asymmetric: both directions, return minimum
        t1 = attc_pursuit_sim(x_curr, ego_id, other_id, params);
        t2 = attc_pursuit_sim(x_curr, other_id, ego_id, params);
        attc = min(t1, t2);
    else
        % Symmetric (bang / prop)
        attc = attc_adversarial_sim(x_curr(ego_id,:), x_curr(other_id,:), params);
    end
end


function attc = attc_adversarial_sim(xi, xj, params)
% Symmetric adversarial forward sim — both agents crash-seek.
%   Uses adaptive sub-stepping: when agents are within the proximity zone,
%   switches to finer timesteps for tighter heading control.
%   Includes continuous closest-approach check between steps.
    dt_w    = params.attc_dt;
    horizon = params.attc_horizon;
    r_col   = 2 * params.r_ttc;
    r_col2  = r_col^2;
    pd      = 1:min(params.dim, 3);   % position indices (1:2 or 1:3)

    % Sub-stepping parameters
    n_sub      = 10;                   % sub-step factor when close
    prox_zone  = 5 * r_col;           % proximity threshold for sub-stepping
    dt_fine    = dt_w / n_sub;

    si = xi(:)';   % working copies, row vectors
    sj = xj(:)';
    t_elapsed = 0;

    while t_elapsed < horizon
        si_old = si;
        sj_old = sj;

        % Adaptive timestep: fine when close
        dp_now = si(pd) - sj(pd);
        dist_now = sqrt(dp_now * dp_now');
        if dist_now < prox_zone
            dt_step = dt_fine;
        else
            dt_step = dt_w;
        end

        % Crash-seeking controls
        ui = crash_seeking_control(si, sj, params);
        uj = crash_seeking_control(sj, si, params);

        % Euler integration
        dxi = dynamics_row(si, ui, params);
        dxj = dynamics_row(sj, uj, params);
        si  = si + dt_step * dxi;
        sj  = sj + dt_step * dxj;

        % Clamp speed (use global max for symmetric adversarial sim)
        v_max_g = max(params.v_max);
        if params.dim == 2
            si(4) = max(0, min(v_max_g, si(4)));
            sj(4) = max(0, min(v_max_g, sj(4)));
        else
            si(6) = max(0, min(v_max_g, si(6)));
            sj(6) = max(0, min(v_max_g, sj(6)));
        end

        t_elapsed = t_elapsed + dt_step;

        % Continuous closest-approach check between si_old→si, sj_old→sj
        dp0 = si_old(pd) - sj_old(pd);
        ddp = (si(pd) - si_old(pd)) - (sj(pd) - sj_old(pd));
        a_coeff = ddp * ddp';
        b_coeff = dp0 * ddp';

        if a_coeff > 0
            t_star = max(0, min(1, -b_coeff / a_coeff));
        else
            t_star = 0;
        end

        dp_min = dp0 + t_star * ddp;
        if dp_min * dp_min' <= r_col2
            attc = t_elapsed - dt_step + t_star * dt_step;
            return;
        end

        % Also check endpoint
        dp_end = si(pd) - sj(pd);
        if dp_end * dp_end' <= r_col2
            attc = t_elapsed;
            return;
        end
    end

    attc = Inf;
end


function attc = attc_pursuit_sim(x_curr, nom_id, pursuer_id, params)
% Asymmetric pursuit forward sim.
%   nom_id follows nominal waypoint control;
%   pursuer_id uses proportional crash-seeking + max acceleration.
%   Uses continuous closest-approach check between Euler steps.
    dt_w    = params.attc_dt;
    horizon = params.attc_horizon;
    r_col   = 2 * params.r_ttc;
    r_col2  = r_col^2;
    N_w     = ceil(horizon / dt_w);
    K_hdg   = 2.5;  % proportional heading gain (same as waypoint_policy)
    pd      = 1:min(params.dim, 3);   % position indices

    % Working copies — use full x_curr so waypoint_policy can reference it
    x_sim   = x_curr;
    s_nom   = x_curr(nom_id,:);
    s_pur   = x_curr(pursuer_id,:);

    for step = 1:N_w
        s_nom_old = s_nom;
        s_pur_old = s_pur;

        % Nominal agent: waypoint-tracking control
        x_sim(nom_id,:)     = s_nom;
        x_sim(pursuer_id,:) = s_pur;
        u_nom = waypoint_policy(x_sim, nom_id, 0, params);

        % Pursuer: proportional heading toward nominal agent + max acceleration
        u_pur = pursuit_control(s_pur, s_nom, K_hdg, params);

        % Euler integration
        dx_nom = dynamics_row(s_nom, u_nom, params);
        dx_pur = dynamics_row(s_pur, u_pur, params);
        s_nom  = s_nom + dt_w * dx_nom;
        s_pur  = s_pur + dt_w * dx_pur;

        % Clamp speed (per-agent v_max)
        vm = params.v_max(:);
        vm_nom = vm(min(nom_id, length(vm)));
        vm_pur = vm(min(pursuer_id, length(vm)));
        if params.dim == 2
            s_nom(4) = max(0, min(vm_nom, s_nom(4)));
            s_pur(4) = max(0, min(vm_pur, s_pur(4)));
        else
            s_nom(6) = max(0, min(vm_nom, s_nom(6)));
            s_pur(6) = max(0, min(vm_pur, s_pur(6)));
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
% Pursuit control: proportional heading toward target + max acceleration.
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
% Crash-seeking control to steer s_self toward s_target.
%   Mode 'bang': bang-bang (max turn rate toward target)
%   Mode 'prop': proportional (K_hdg * error, clamped to max rate)
%   2D: u = [omega, a_max]                  — turn rate + max accel
%   3D: u = [u_psi, u_gamma, a_max]         — yaw & pitch rates + max accel
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

        % Yaw toward target
        psi_des = atan2(dy, dx);
        psi_err = mod(psi_des - s_self(4) + pi, 2*pi) - pi;

        % Pitch toward target
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


function dxdt = dynamics_row(x_row, u_row, params)
% Row-vector wrapper around dynamics (avoids column-vector reshape).
    if params.dim == 2
        v     = x_row(4);
        theta = x_row(3);
        dxdt  = [v*cos(theta), v*sin(theta), u_row(1), u_row(2)];
    else
        v     = x_row(6);
        psi   = x_row(4);
        gamma = x_row(5);
        dxdt  = [v*cos(gamma)*cos(psi), v*cos(gamma)*sin(psi), ...
                 v*sin(gamma), u_row(1), u_row(2), u_row(3)];
    end
end


%% ==================== UTILITIES ====================

function u = clamp_controls(u, params)
    na = size(u,1);
    for i = 1:na
        u(i,1) = max(-params.omega_max, min(params.omega_max, u(i,1)));
        if params.dim == 2
            u(i,2) = max(-params.a_max, min(params.a_max, u(i,2)));
        else
            u(i,2) = max(-params.nu_max, min(params.nu_max, u(i,2)));
            u(i,3) = max(-params.a_max,  min(params.a_max,  u(i,3)));
        end
    end
end


function [ic, wps] = random_sphere_wp_sequence(R_sphere, dim, n_wp)
%RANDOM_SPHERE_WP_SEQUENCE  Build one agent's random_sphere start point
%   and length-n_wp waypoint chain.
%
%   First WP is antipodal to the start; subsequent WPs are random points
%   on the hemisphere opposite the previous WP.  Mirrors the per-agent
%   loop used by mission='random_sphere' and the evader half of
%   mission='random_sphere_pursuit'.
%
%   Returns:
%     ic   1 x dim     start point on the sphere
%     wps  n_wp x dim  waypoint sequence
    ic  = random_sphere_point(R_sphere, dim);
    wps = zeros(n_wp, dim);
    if dim == 2
        prev_ang = atan2(ic(2), ic(1));
        for w = 1:n_wp
            if w == 1
                new_ang = prev_ang + pi;                          % antipodal
            else
                new_ang = prev_ang + pi + (rand - 0.5) * pi;      % opposite semicircle
            end
            wps(w,:) = R_sphere * [cos(new_ang), sin(new_ang)];
            prev_ang = new_ang;
        end
    else
        prev_dir = ic / R_sphere;
        for w = 1:n_wp
            if w == 1
                new_dir = -prev_dir;                              % antipodal
            else
                new_dir = randn(1, 3);
                new_dir = new_dir / norm(new_dir);
                if dot(new_dir, prev_dir) > 0
                    new_dir = -new_dir;                           % flip to opposite hemisphere
                end
            end
            wps(w,:) = R_sphere * new_dir;
            prev_dir = new_dir;
        end
    end
end


function p = random_sphere_point(R_sphere, dim)
%RANDOM_SPHERE_POINT  One uniform random point on the sphere of radius R_sphere.
    if dim == 2
        theta = 2*pi * rand;
        p = R_sphere * [cos(theta), sin(theta)];
    else
        d = randn(1, 3);
        d = d / norm(d);
        p = R_sphere * d;
    end
end


function wps = generate_random_waypoint_grid(n_wp, dim, spread)
%GENERATE_RANDOM_WAYPOINT_GRID  Uniform random waypoints in a box of
%   X full-width = `spread`, Y full-width = 0.8*spread (centered on 0),
%   and (3D only) Z full-width = 0.4*spread (centered on 0).
%
%   Aspect ratio matches the historic hardcoded spreads in auspice_sim:
%     spread=250 -> [0,250] x [-100,100] x [-50,50]    (waypoints_random[_sync])
%     spread=50  -> [0,50]  x [-20,20]   x [-10,10]    (team_pursuit, formation*)
%
%   RNG-call order matches the original blocks (x then y then z) so that
%   passing the historic spread reproduces the exact prior waypoints under
%   the same seed.
    x = spread     * rand(n_wp, 1);
    y = spread*0.8 * rand(n_wp, 1) - spread*0.4;     % [-0.4*spread, 0.4*spread]
    if dim == 2
        wps = [x, y];
    else
        z = spread*0.4 * rand(n_wp, 1) - spread*0.2; % [-0.2*spread, 0.2*spread]
        wps = [x, y, z];
    end
end
