function [attc, grad_raw, H_features] = attc_nn_forward_hessian(z_raw, nn, grad_clip, v_max)
%ATTC_NN_FORWARD_HESSIAN  Forward + backward + feature-space Hessian for aTTC NN.
%
%   [attc, grad_raw, H_features] = attc_nn_forward_hessian(z_raw, nn, grad_clip, v_max)
%
%   Same inputs/outputs as attc_nn_forward, plus:
%   H_features : (6x6) Hessian of attc w.r.t. raw features z_raw.
%
%   For a ReLU network with softplus (or softplus+log) output, all curvature
%   comes from the scalar output activation.  The feature-space Hessian is
%   therefore rank-1:
%
%       H_features = C_hess * g_unc * g_unc'
%
%   where g_unc is the UNCLIPPED gradient w.r.t. z_raw and:
%       Without log-transform:  C_hess = (1 - sigma(a4)) / sigma(a4)
%       With    log-transform:  C_hess = 1 / (exp(f_nn) * sigma(a4))
%
%   The clipped grad_raw is returned for use in CBF gradient computations;
%   the unclipped version is used for the Hessian so it is not distorted
%   by the clipping operation.

    use_film = isfield(nn, 'film') && nn.film > 0.5;
    if use_film && (nargin < 4 || isempty(v_max))
        error('v_max required for FiLM model');
    end

    % --- Normalize ---
    z = (z_raw - nn.feat_mean) ./ nn.feat_std;

    % --- FiLM conditioning ---
    if use_film
        c = (v_max - nn.v_max_min) / (nn.v_max_max - nn.v_max_min);
        ca1 = nn.cond_W1' * c + nn.cond_b1;
        ch1 = max(ca1, 0);
        ca2 = nn.cond_W2' * ch1 + nn.cond_b2;
        ch2 = max(ca2, 0);
        ca3 = nn.cond_W3' * ch2 + nn.cond_b3;
        sizes = nn.film_sizes(:)';
        n_layers = length(sizes);
        gamma = cell(1, n_layers);
        beta  = cell(1, n_layers);
        idx = 0;
        for k = 1:n_layers
            gamma{k} = ca3(idx+1 : idx+sizes(k));
            idx = idx + sizes(k);
            beta{k}  = ca3(idx+1 : idx+sizes(k));
            idx = idx + sizes(k);
        end
    end

    % --- Forward pass ---
    a1 = nn.W1' * z + nn.b1;
    if use_film; a1_mod = gamma{1} .* a1 + beta{1}; else; a1_mod = a1; end
    h1 = max(a1_mod, 0);

    a2 = nn.W2' * h1 + nn.b2;
    if use_film; a2_mod = gamma{2} .* a2 + beta{2}; else; a2_mod = a2; end
    h2 = max(a2_mod, 0);

    a3 = nn.W3' * h2 + nn.b3;
    if use_film; a3_mod = gamma{3} .* a3 + beta{3}; else; a3_mod = a3; end
    h3 = max(a3_mod, 0);

    a4   = nn.W4' * h3 + nn.b4;
    % NOTE (2026-05-06): numerically stable softplus.  Identical to
    % `log(1 + exp(a4))` for any finite a4; avoids overflow at |a4| > ~709.
    % To revert: change to  f_nn = log(1 + exp(a4));
    f_nn = max(a4, 0) + log(1 + exp(-abs(a4)));    % numerically stable softplus(a4)

    use_log = isfield(nn, 'log_transform') && nn.log_transform > 0.5;
    if use_log
        attc = exp(f_nn) - 1;
    else
        attc = f_nn;
    end

    % --- Backward pass ---
    sigma_a4 = 1 / (1 + exp(-a4));   % sigmoid(a4)
    if use_log
        d4 = exp(f_nn) * sigma_a4;
    else
        d4 = sigma_a4;
    end

    if use_film
        d3 = (nn.W4 * d4) .* (gamma{3} .* double(a3_mod > 0));
        d2 = (nn.W3 * d3) .* (gamma{2} .* double(a2_mod > 0));
        d1 = (nn.W2 * d2) .* (gamma{1} .* double(a1_mod > 0));
    else
        d3 = (nn.W4 * d4) .* double(a3_mod > 0);
        d2 = (nn.W3 * d3) .* double(a2_mod > 0);
        d1 = (nn.W2 * d2) .* double(a1_mod > 0);
    end

    grad_norm        = nn.W1 * d1;
    grad_raw_unclip  = grad_norm ./ nn.feat_std;   % unclipped — used for Hessian

    % --- Hessian of attc w.r.t. z_raw  (rank-1 trick) ---
    % H_features = C_hess * grad_raw_unclip * grad_raw_unclip'
    %   Without log:  G(a4) = softplus → C = (1-sigma)/sigma
    %   With log:     G = exp(softplus)-1 → C = 1/d4
    if use_log
        C_hess = 1 / d4;
    else
        C_hess = (1 - sigma_a4) / sigma_a4;
    end

    H_features = C_hess * (grad_raw_unclip * grad_raw_unclip');

    % --- Clip gradient for output ---
    g_norm = norm(grad_raw_unclip);
    if g_norm > grad_clip
        grad_raw = grad_raw_unclip * (grad_clip / g_norm);
    else
        grad_raw = grad_raw_unclip;
    end
end
