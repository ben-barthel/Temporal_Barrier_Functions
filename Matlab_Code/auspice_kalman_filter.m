function [x_next, Sigma] = auspice_kalman_filter(x_next, Sigma, params)
%AUSPICE_KALMAN_FILTER  Kalman filter measurement update for all agents.
%
%   [x_next, Sigma] = auspice_kalman_filter(x_next, Sigma, params)
%
%   For each agent:
%     1. Generate a noisy position measurement of the true state.
%     2. Compute the Kalman gain from the predicted covariance.
%     3. Correct the state estimate using the innovation.
%     4. Update the covariance via the Joseph form (numerically stable).
%
%   Inputs:
%     x_next  - [na x sd] state matrix (post-dynamics, post-noise)
%     Sigma   - {na x 1} cell of [sd x sd] covariance matrices
%     params  - simulation parameter struct (must include H, R_obs,
%               obs_noise_std, state_dim)
%
%   Outputs:
%     x_next  - corrected state matrix
%     Sigma   - updated covariance cell array

    na = size(x_next, 1);
    sd = params.state_dim;

    for i = 1:na
        % Noisy position measurement of the true (post-noise) state
        z = params.H * x_next(i,:)' + ...
            params.obs_noise_std(:) .* randn(size(params.H,1),1);

        % Kalman gain
        S = params.H * Sigma{i} * params.H' + params.R_obs;
        K = Sigma{i} * params.H' / S;

        % State correction
        innovation = z - params.H * x_next(i,:)';
        x_next(i,:) = x_next(i,:) + (K * innovation)';

        % Covariance correction (Joseph form for numerical stability)
        IKH = eye(sd) - K * params.H;
        Sigma{i} = IKH * Sigma{i} * IKH' + K * params.R_obs * K';
    end
end
