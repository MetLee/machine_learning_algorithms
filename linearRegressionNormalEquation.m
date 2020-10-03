function theta = linearRegressionNormalEquation(X, y, lambda)

    % Compute the theta using normal equation with regularization.

    if nargin == 2
        lambda = 0;
    end

    n_plus_1 = size(X, 2);
    
    reg = eye(n_plus_1);
    reg(1) = 0;
    theta = pinv(X' * X + lambda * reg) * X' * y;

end