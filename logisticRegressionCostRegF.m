function [J, grad] = logisticRegressionCostRegF(X, y, theta, lambda)

    % Compute the cost and the gradient of logistic regression algorithm with regularization.

    m = length(y);
    n_plus_1 = size(X, 2);
    
    z = X * theta;
    h = sigmoid(z);
    costMatrix = - y .* log(h) - (1 - y) .* log(1 - h);
    theta_without_theta0 = theta(2:n_plus_1); % do not regularize theta(1)
    regMatrix = 1 / 2 * lambda * theta_without_theta0 .^ 2;
    J = 1 / m * (sum(costMatrix) + sum(regMatrix));

    grad0 = 1 / m * ((h - y)' * X(:, 1))';
    X_without_X0 = X(:, [2:n_plus_1]);
    grad_rest = 1 / m * (((h - y)' * X_without_X0)' + lambda * theta_without_theta0);
    grad = [grad0; grad_rest];

end