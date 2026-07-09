function [r, c] = min_enclosing_ball_radius(P, varargin)
%MIN_ENCLOSING_BALL_RADIUS  Approximate minimum-enclosing-ball radius and
%   center for a d-dimensional point cloud.
%
%   [r, c] = MIN_ENCLOSING_BALL_RADIUS(P) returns the radius r and the
%   center c (1 x d) of an approximately minimum-enclosing ball of the
%   rows of P (n x d).  Points are treated as ordered by row.
%
%   Algorithm: Frank-Wolfe iteration on the MEB formulation with the
%   harmonic step size 1/(t+1).  This is *approximate* — convergence is
%   O(1/T) in the primal gap.  With the default max_iter = 200 the
%   relative radius error is typically well below 0.5% for the small
%   point clouds encountered here (n <~ 20).  Not to be used for
%   worst-case-critical guarantees; use Welzl's algorithm if you need
%   exactness.
%
%   Bias direction: the returned r is max‖P - c_est‖ from the *estimated*
%   center, not the true optimal center.  When c_est is imperfect r
%   *over*-estimates the true MEB radius by roughly ‖c_est - c_true‖.
%   For validation harnesses this manifests as r slightly above R for
%   uniformly-sampled sphere clouds (typical +0.2 - 0.5% at n = 200).
%
%   Cost per iteration: O(n * d).  Total: O(max_iter * n * d).
%
%   Name-value options:
%     'max_iter'  int    default 200
%     'tol'       double default 1e-6
%           Early exit when the radius stops changing:
%             (r_prev - r_cur) / max(r_cur, eps) < tol
%
%   Degenerate inputs:
%     empty P     -> r = 0, c = zero-vector of size (1, size(P,2)) if P has
%                    columns, otherwise (1, 0)
%     single row  -> r = 0, c = P
%
%   Example
%     P = randn(20, 3);
%     [r, c] = min_enclosing_ball_radius(P);
%
%   See also: VALIDATE_MIN_ENCLOSING_BALL_RADIUS.

    ip = inputParser;
    addParameter(ip, 'max_iter', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(ip, 'tol',      1e-6, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    parse(ip, varargin{:});
    max_iter = double(ip.Results.max_iter);
    tol      = double(ip.Results.tol);

    if isempty(P)
        d = size(P, 2);
        c = zeros(1, d);
        r = 0;
        return;
    end

    n = size(P, 1);
    d = size(P, 2);

    if n == 1
        c = P;
        r = 0;
        return;
    end

    c = mean(P, 1);
    r_prev = inf;
    for it = 1:max_iter
        d2 = sum((P - c).^2, 2);
        [d2max, kmax] = max(d2);
        r_cur = sqrt(d2max);
        if abs(r_prev - r_cur) < tol * max(r_cur, eps)
            break;
        end
        r_prev = r_cur;
        c = c + (P(kmax, :) - c) / (it + 1);
    end
    r = sqrt(max(sum((P - c).^2, 2)));
end
