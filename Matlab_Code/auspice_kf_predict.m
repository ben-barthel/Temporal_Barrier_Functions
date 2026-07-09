function Sigma = auspice_kf_predict(x_curr, Sigma, params)
%AUSPICE_KF_PREDICT  Propagate state covariance via linearized dynamics.
%
%   Sigma = auspice_kf_predict(x_curr, Sigma, params)
%
%   Discrete-time covariance prediction using the Jacobian of the
%   continuous-time Dubins dynamics:
%       A_d = I + dt * df/dx
%       Sigma_k+1|k = A_d * Sigma_k|k * A_d' + dt * Q
%
%   Inputs:
%     x_curr  - [na x sd] current state matrix
%     Sigma   - {na x 1} cell of [sd x sd] covariance matrices
%     params  - simulation parameter struct (must include dt, Q, dim,
%               state_dim)
%
%   Outputs:
%     Sigma   - predicted covariance cell array

    na = size(x_curr, 1);
    sd = params.state_dim;

    for i = 1:na
        Jac = dynamics_jacobian(x_curr(i,:)', params);
        A_d = eye(sd) + params.dt * Jac;
        Sigma{i} = A_d * Sigma{i} * A_d' + params.dt * params.Q;
    end
end


function Jac = dynamics_jacobian(x, params)
% Jacobian df/dx of Dubins dynamics (for covariance propagation).
    if params.dim == 2
        v = x(4); theta = x(3);
        Jac = [0, 0, -v*sin(theta), cos(theta);
               0, 0,  v*cos(theta), sin(theta);
               0, 0,  0,            0;
               0, 0,  0,            0];
    else
        v = x(6); psi = x(4); gamma = x(5);
        Jac = zeros(6);
        Jac(1,4) = -v*cos(gamma)*sin(psi);
        Jac(1,5) = -v*sin(gamma)*cos(psi);
        Jac(1,6) =    cos(gamma)*cos(psi);
        Jac(2,4) =  v*cos(gamma)*cos(psi);
        Jac(2,5) = -v*sin(gamma)*sin(psi);
        Jac(2,6) =    cos(gamma)*sin(psi);
        Jac(3,5) =  v*cos(gamma);
        Jac(3,6) =    sin(gamma);
    end
end
