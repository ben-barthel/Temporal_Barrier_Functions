function [attc, grad_raw] = attc_nn_forward(z_raw, nn, grad_clip, v_max)
%ATTC_NN_FORWARD  Standalone forward + backward pass for aTTC NN surrogate.
%
%   [attc, grad_raw] = attc_nn_forward(z_raw, nn, grad_clip, v_max)
%
%   z_raw     : (6x1) raw features [dp; dv] (unnormalized)
%   nn        : struct with W1..W4, b1..b4, feat_mean, feat_std
%               For FiLM: also cond_W1..3, cond_b1..3, film_sizes,
%               v_max_min, v_max_max, film
%   grad_clip : max gradient norm (scalar)
%   v_max     : (optional) scalar v_max for FiLM conditioning
%
%   Returns:
%   attc      : predicted aTTC (scalar, seconds)
%   grad_raw  : (6x1) gradient w.r.t. raw features (clipped)
%
%   This is a standalone copy of the local function in auspice_cbf.m,
%   callable from any file (e.g. auspice_sim.m for saTTC initialisation).

    use_film = isfield(nn, 'film') && nn.film > 0.5;
    if use_film && (nargin < 4 || isempty(v_max))
        error('v_max required for FiLM model');
    end

    % --- Normalize state features ---
    z = (z_raw - nn.feat_mean) ./ nn.feat_std;

    % --- FiLM conditioning pathway ---
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

    a4 = nn.W4' * h3 + nn.b4;
    % NOTE (2026-05-06): the softplus below uses the numerically stable form
    %   max(a4,0) + log(1 + exp(-|a4|))   ==   log(1 + exp(a4))
    % These are mathematically identical for any finite a4.  The stable form
    % avoids overflowing to Inf when |a4| > ~709 (which has been observed on
    % OOD states in 6+ agent pursuit sims).  To revert to the textbook form,
    % replace the next line with:  f_nn = log(1 + exp(a4));
    f_nn = max(a4, 0) + log(1 + exp(-abs(a4)));    % numerically stable softplus(a4)

    use_log = isfield(nn, 'log_transform') && nn.log_transform > 0.5;
    if use_log
        attc = exp(f_nn) - 1;
    else
        attc = f_nn;
    end

    % --- Backward pass ---
    d4 = 1 / (1 + exp(-a4));        % d(softplus)/d(a4) = sigmoid(a4)
    if use_log
        d4 = exp(f_nn) * d4;        % chain through exp(f_nn) - 1
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

    grad_norm = nn.W1 * d1;
    grad_raw  = grad_norm ./ nn.feat_std;

    % Clip gradient norm
    g_norm = norm(grad_raw);
    if g_norm > grad_clip
        grad_raw = grad_raw * (grad_clip / g_norm);
    end
end
