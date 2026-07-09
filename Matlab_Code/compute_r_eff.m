function r_eff = compute_r_eff(x_curr, i, j, Sigma, params)
%COMPUTE_R_EFF  Inflate collision radius to satisfy P(collision) <= epsilon.
%   Projects combined positional uncertainty onto the line of approach
%   and inflates by kappa * sigma_d.
    pos_dim = min(params.dim, 3);  % 2D: indices 1:2, 3D: indices 1:3
    Sig_rel = Sigma{i}(1:pos_dim,1:pos_dim) + Sigma{j}(1:pos_dim,1:pos_dim);
    dp      = x_curr(i,1:pos_dim) - x_curr(j,1:pos_dim);
    d       = norm(dp);

    if d < 1e-6
        sigma_d = sqrt(max(eig(Sig_rel)));   % worst-case direction
    else
        n_hat   = dp(:) / d;
        sigma_d = sqrt(n_hat' * Sig_rel * n_hat);
    end

    r_eff = params.r_cbf + params.kappa * sigma_d;
end
