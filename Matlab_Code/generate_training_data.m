function generate_training_data(v_desired_in, out_tag_in)
%GENERATE_TRAINING_DATA  Generate labeled aTTC training data for the NN.
%
%   generate_training_data()               % full sweep (4 v_desired values)
%   generate_training_data(0.25)           % single v_desired
%   generate_training_data(0.25, 'data1')  % custom output tag
%
%   For each v_desired this script:
%     1. Runs six 2-agent scenario simulations (waypoints_random_sync,
%        random_sphere, head_on_ping_pong, and pursuit at advantage
%        0.9 / 1.5 / 0.5), T_sim/6 seconds each, chosen to expose the NN
%        to a wide variety of relative positions and velocities.
%     2. Concatenates the trajectories.
%     3. Labels every timestep with the adversarial time-to-collision
%        (aTTC) oracle at five assumed pursuer speeds
%        v_max = [0.10, 0.30, 0.50, 0.70, 0.90] km/s
%        via compute_attc_trajectory (forward integration of the
%        pursuit dynamics — this is the expensive step).
%     4. Saves one .mat per (v_desired, v_max) pair to ../Results/:
%        training_data_<tag>_3D_2A_T50000_vdes<V>_vmax<V>.mat
%
%   The full sweep (4 v_desired x 5 v_max = 20 files) is what
%   Python_Code/run_ml_training_vraw.py expects as input.  Runtime is
%   substantial: each v_desired takes hours (dominated by the aTTC
%   labeling).  On a cluster, launch one job per v_desired:
%       matlab -batch "generate_training_data(0.15)"
%       matlab -batch "generate_training_data(0.25)"
%       matlab -batch "generate_training_data(0.35)"
%       matlab -batch "generate_training_data(0.50)"
%
%   Inputs (both optional):
%     v_desired   scalar km/s, or [] for the full default sweep
%     out_tag     char, tag baked into output filenames (default 'data1')

if nargin < 1, v_desired_in = []; end
if nargin < 2 || isempty(out_tag_in), out_tag_in = 'data1'; end

if isempty(v_desired_in)
    v_desired_sweep = [0.15, 0.25, 0.35, 0.50];
    fprintf(['No v_desired given — running the full %d-value sweep.\n' ...
             'This takes several hours per value; consider one job per ' ...
             'value on a cluster.\n\n'], numel(v_desired_sweep));
else
    v_desired_sweep = double(v_desired_in);
end

for v_des = v_desired_sweep(:)'
    generate_one_vdes(v_des, char(out_tag_in));
end
end


function generate_one_vdes(v_desired, out_tag)
%GENERATE_ONE_VDES  All missions + aTTC labeling for one v_desired value.

%% --- Fixed configuration (the "data1" recipe used in the paper) ---
dim        = 3;
num_agents = 2;
T_sim      = 50000;

% aTTC oracle parameters
v_max_list   = [0.10, 0.30, 0.50, 0.70, 0.90];   % assumed pursuer speeds
attc_type    = 'pursuit_ego';
r_ttc        = 0.1;      % collision radius [km]
attc_dt      = 0.2;      % oracle forward-integration timestep [s]
attc_horizon = 120;      % oracle horizon [s]; aTTC > horizon -> Inf

% Simulation v_max for trajectory generation
sim_v_max     = 0.55;
crossing_time = 100;     % sets the random_sphere radius (= v_des*ct/2)

%% --- Mission list ---
missions           = {'waypoints_random_sync', 'random_sphere', ...
                      'head_on_ping_pong', 'pursuit', 'pursuit', 'pursuit'};
pursuit_advantages = {0, 0, 0, 0.9, 1.5, 0.5};
n_missions = length(missions);

% Each mission contributes ~equally: T_each = T_sim / n_missions.
T_each     = T_sim / n_missions;
noise_mult = 10;

% Output lands in ../Results (sibling of Matlab_Code/).
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Results');

fprintf('========================================\n');
fprintf('  aTTC training-data generator\n');
fprintf('  v_desired         = %.3f km/s\n', v_desired);
fprintf('  sim v_max         = %.3f km/s\n', sim_v_max);
fprintf('  T_sim             = %d s   (%d x %.0f s missions)\n', T_sim, n_missions, T_each);
fprintf('  crossing_time     = %d s\n', crossing_time);
fprintf('  R_sphere          = %.2f km (= v_desired * crossing_time / 2)\n', ...
        v_desired * crossing_time / 2);
fprintf('  missions          = %s\n', strjoin(missions, ', '));
fprintf('  pursuit_advantages = [%s]\n', ...
        num2str(cell2mat(pursuit_advantages), '%.2f '));
fprintf('  output tag        = %s\n', out_tag);
fprintf('========================================\n\n');

%% --- Run simulations at this v_desired ---
runs = cell(1, n_missions);
for m = 1:n_missions
    fprintf('=== Running mission %d/%d: %s (%.0f s, v_des=%.2f, adv=%.2f) ===\n', ...
        m, n_missions, missions{m}, T_each, v_desired, pursuit_advantages{m});
    if strcmp(missions{m}, 'pursuit')
        cbf_type_m = 'hocbf';   % evader defends itself so pursuit lasts
    else
        cbf_type_m = 'none';
    end
    tic;
    runs{m} = auspice_sim( ...
        'dim',               dim, ...
        'num_agents',        num_agents, ...
        'mission',           missions{m}, ...
        'T_sim',             T_each, ...
        'v_desired',         v_desired, ...
        'v_max',             sim_v_max, ...
        'cbf_type',          cbf_type_m, ...
        'pursuit_advantage', pursuit_advantages{m}, ...
        'crossing_time',     crossing_time, ...
        'use_kalman_filter', false, ...
        'compute_ttc',       false, ...
        'compute_attc',      false, ...
        'noise_std_mult',    noise_mult);
    fprintf('  Done in %.1f s  (%d timesteps)\n', toc, size(runs{m}.x, 1));
end

%% --- Concatenate trajectories ---
fprintf('\nConcatenating %d runs...\n', n_missions);

t_offset = 0;
t_all    = [];
x_all    = [];
dist_all = [];

for m = 1:n_missions
    r = runs{m};
    t_all    = [t_all;    r.t + t_offset];       %#ok<AGROW>
    x_all    = [x_all;    r.x];                  %#ok<AGROW>
    dist_all = [dist_all; r.dist];               %#ok<AGROW>
    t_offset = t_all(end) + r.t(2) - r.t(1);
end

results_base.t      = t_all;
results_base.x      = x_all;
results_base.dist   = dist_all;
results_base.params = runs{1}.params;
results_base.params.T_sim              = T_sim;
results_base.params.missions           = missions;
results_base.params.v_desired          = v_desired;
results_base.params.sim_v_max          = sim_v_max;
results_base.params.crossing_time      = crossing_time;
results_base.params.pursuit_advantages = pursuit_advantages;

fprintf('Combined: %d timesteps, x = [%s]\n', ...
    size(results_base.x, 1), num2str(size(results_base.x)));

%% --- Compute aTTC labels for each assumed pursuer v_max ---
n_v = length(v_max_list);
fprintf('\nComputing aTTC for %d v_max values (type: %s)\n', n_v, attc_type);

for iv = 1:n_v
    v_val = v_max_list(iv);
    fprintf('  [%d/%d] v_max = %.3f km/s (v_des=%.2f) ...', iv, n_v, v_val, v_desired);
    tic;

    res_copy = results_base;
    res_copy.params.v_max = v_val;
    if isfield(res_copy, 'attc')
        res_copy = rmfield(res_copy, 'attc');
    end

    res_copy = compute_attc_trajectory(res_copy, r_ttc, ...
        'attc_dt',      attc_dt, ...
        'attc_horizon', attc_horizon, ...
        'attc_type',    attc_type, ...
        'verbose',      false);

    results = res_copy;
    results.params.v_max_attc = v_val;
    results.params.v_desired  = v_desired;
    results.params.attc_type  = attc_type;

    out_name = sprintf('training_data_%s_%dD_%dA_T%d_vdes%.2f_vmax%.2f.mat', ...
        out_tag, dim, num_agents, T_sim, v_desired, v_val);
    save(fullfile(out_dir, out_name), 'results', '-v7.3');
    fprintf(' done (%.1f s). Saved %s\n', toc, out_name);
end

fprintf('\nAll done for v_desired=%.3f (tag=%s).\n', v_desired, out_tag);
end
