function [J, grad] = linearRegressionCostReg(X, y, theta, lambda)

    % Compute the cost and the gradient of linear regression algorithm with regularization.

    m = length(y);
    n_plus_1 = size(X, 2);

    h = X * theta;
    costMatrix = h - y;
    theta_without_theta0 = theta(2:n_plus_1); % do not regularize theta(1)
    regMatrix = 1 / 2 * lambda * theta_without_theta0 .^ 2;
    J = 1 / (2 * m) * ((costMatrix' * costMatrix) + sum(regMatrix));

    X_without_X0 = X(:, [2:n_plus_1]);
    grad0 = 1 / m * ((X * theta - y)' * X)';
    grad_rest = 1 / m * (((h - y)' * X_without_X0)' + lambda * theta_without_theta0);
    grad = [grad0; grad_rest];

end